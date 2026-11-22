"""Streamlit Web UI for Text-to-SQL with QLoRA & AST Safety Guard.

Run locally:
    streamlit run app.py
"""
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.exec_eval import get_schema, open_readonly, run_query
from src.prompting import build_messages, extract_sql
from src.sql_guard import check_sql

# Page configuration
st.set_page_config(
    page_title="Text-to-SQL with QLoRA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_sample_db() -> Path:
    """Create a sample company.db if it doesn't already exist."""
    db_path = Path("company.db")
    if not db_path.exists():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT NOT NULL,
            location TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            emp_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dept_id INTEGER,
            salary REAL NOT NULL,
            hire_year INTEGER NOT NULL,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        );
        """)
        cur.executemany("INSERT INTO departments VALUES (?, ?, ?)", [
            (1, "Engineering", "San Francisco"),
            (2, "Sales", "New York"),
            (3, "Marketing", "Chicago"),
            (4, "Human Resources", "Austin"),
        ])
        cur.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", [
            (101, "Alice Chen", 1, 145000, 2021),
            (102, "Bob Smith", 1, 130000, 2022),
            (103, "Charlie Brown", 2, 95000, 2020),
            (104, "Diana Prince", 3, 110000, 2019),
            (105, "Evan Wright", 1, 160000, 2018),
            (106, "Fiona Gallagher", 2, 105000, 2023),
            (107, "George Miller", 4, 85000, 2022),
        ])
        conn.commit()
        conn.close()
    return db_path


@st.cache_resource(show_spinner="Loading model and tokenizer...")
def load_cached_model(base_model_id: str, adapter_path: str | None, load_4bit: bool):
    tok = AutoTokenizer.from_pretrained(base_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {}
    if load_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


# --- Header & Overview ---
st.title("🔍 Text-to-SQL with QLoRA & AST Safety Guard")
st.markdown("""
Convert natural language questions into executable SQLite queries using **Qwen2.5-1.5B-Instruct** fine-tuned with **QLoRA**.
Every generated query passes through an **AST-based SQL Safety Guard** before executing on a **read-only** database connection.
""")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")

# Model configuration
model_mode = st.sidebar.selectbox(
    "Model Selection",
    [
        "Local Fine-Tuned Adapter (outputs/qwen2.5-1.5b-text2sql-lora)",
        "Base Model Only (Qwen/Qwen2.5-1.5B-Instruct)",
        "Hugging Face Hub (GPTpasta/qwen2.5-1.5b-text2sql-lora)",
    ]
)

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
if "Local Fine-Tuned" in model_mode:
    adapter_path = "outputs/qwen2.5-1.5b-text2sql-lora" if Path("outputs/qwen2.5-1.5b-text2sql-lora").exists() else None
elif "Hugging Face Hub" in model_mode:
    adapter_path = "GPTpasta/qwen2.5-1.5b-text2sql-lora"
else:
    adapter_path = None

use_4bit = st.sidebar.checkbox("Load 4-bit (NF4 Quantization)", value=torch.cuda.is_available())

# Database selection
st.sidebar.subheader("🗄️ Database Source")
db_choice = st.sidebar.radio(
    "Choose Database",
    ["Sample Company DB", "Spider Database", "Upload Custom SQLite (.db/.sqlite)"]
)

sample_db_file = ensure_sample_db()
active_db_path = sample_db_file

if db_choice == "Sample Company DB":
    active_db_path = sample_db_file
elif db_choice == "Spider Database":
    spider_databases = list(Path("data/spider/database").glob("*/*.sqlite")) if Path("data/spider/database").exists() else []
    if spider_databases:
        db_names = [f"{p.parent.name}/{p.name}" for p in spider_databases]
        selected_db_str = st.sidebar.selectbox("Select Spider DB", db_names)
        active_db_path = Path("data/spider/database") / selected_db_str
    else:
        st.sidebar.warning("Spider database directory not found. Using Sample Company DB.")
        active_db_path = sample_db_file
elif db_choice == "Upload Custom SQLite (.db/.sqlite)":
    uploaded_file = st.sidebar.file_uploader("Upload SQLite file", type=["db", "sqlite", "sqlite3"])
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        temp_db_path = Path(temp_dir) / uploaded_file.name
        temp_db_path.write_bytes(uploaded_file.getbuffer())
        active_db_path = temp_db_path
    else:
        st.sidebar.info("Upload a file or using sample company DB.")
        active_db_path = sample_db_file

# Inspect Schema
conn = open_readonly(active_db_path)
db_schema = get_schema(conn)
conn.close()

with st.sidebar.expander("📋 View Database Schema", expanded=False):
    st.code(db_schema, language="sql")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Hardware:** {'🟢 NVIDIA GPU (CUDA)' if torch.cuda.is_available() else '🟡 CPU Mode'}")
st.sidebar.markdown("[⭐ View GitHub Repo](https://github.com/SEASIL/text2sql-qlora)")
st.sidebar.markdown("[🤗 Hugging Face Model](https://huggingface.co/GPTpasta/qwen2.5-1.5b-text2sql-lora)")

# --- Main Interaction ---
st.subheader("💬 Ask a Question")

# Preset sample questions
default_questions = {
    "Sample Company DB": [
        "Show all employees with a salary greater than 120000 ordered by salary descending.",
        "How many employees are in each department?",
        "What is the average salary in the Engineering department?",
        "Who is the highest-paid employee and what is their salary?",
    ],
    "Spider Database": [
        "How many singers are there?",
        "Show name, country, and age of all singers ordered by age descending.",
        "What are all car makers from USA?",
    ],
    "Upload Custom SQLite (.db/.sqlite)": [
        "Show the first 5 rows from each table.",
        "Count the total records in the primary table.",
    ]
}

current_presets = default_questions.get(db_choice, default_questions["Sample Company DB"])
selected_preset = st.selectbox("Quick Sample Questions (or type custom below):", ["-- Select a preset or type below --"] + current_presets)

custom_question = st.text_input(
    "Enter question in plain English:",
    value="" if selected_preset.startswith("--") else selected_preset,
    placeholder="e.g. List all employees hired after 2020 in San Francisco"
)

col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("🚀 Run Text-to-SQL", type="primary", use_container_width=True)

if generate_btn and custom_question.strip():
    with st.spinner("Generating SQL query..."):
        try:
            tok, model = load_cached_model(base_model_id, adapter_path, use_4bit)
            messages = build_messages(db_schema, custom_question.strip())
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            start_t = time.monotonic()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
            gen_time = (time.monotonic() - start_t) * 1000

            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            raw_output = tok.decode(new_tokens[0], skip_special_tokens=True)
            sql = extract_sql(raw_output)

        except Exception as e:
            st.error(f"Generation error: {e}")
            sql = None

    if sql:
        st.markdown("### 1. Generated SQL Query")
        st.code(sql, language="sql")
        st.caption(f"Inference Latency: **{gen_time:.1f} ms**")

        # SQL Guard Check
        st.markdown("### 2. AST Safety Guard Inspection")
        guard_result = check_sql(sql)

        if guard_result.allowed:
            st.success("✅ **ALLOWED:** Query passed AST safety checks (single read-only SELECT / WITH statement).")

            # Execute Query
            st.markdown("### 3. Execution Results")
            exec_conn = open_readonly(active_db_path)
            exec_res = run_query(exec_conn, sql)
            exec_conn.close()

            if exec_res.ok:
                if exec_res.rows:
                    df = pd.DataFrame(exec_res.rows)
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"Returned **{len(exec_res.rows)} rows** successfully.")
                else:
                    st.info("Query executed successfully but returned 0 rows.")
            else:
                st.error(f"Execution Error: {exec_res.error}")
        else:
            st.error(f"🛑 **BLOCKED BY SAFETY GUARD:** {guard_result.reason}")
            st.warning("Destructive statements (DROP, DELETE, UPDATE, INSERT, ALTER) or multi-statement injections are forbidden.")
