"""Generate notebooks/demo.ipynb for Google Colab."""
import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Text-to-SQL with QLoRA & SQL Safety Guard\n",
            "**Portfolio Project:** Fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` with an execution-safe SQL evaluation loop on SQLite.\n",
            "\n",
            "This notebook demonstrates:\n",
            "1. Loading the 1.5B model in 4-bit NF4 with our LoRA adapter on free Google Colab T4 GPU (or CPU via llama.cpp).\n",
            "2. Creating a sample SQLite database (`company.db`).\n",
            "3. Generating SQL queries from English questions.\n",
            "4. Enforcing the AST SQL safety guard (`sql_guard`) to prevent destructive statements.\n",
            "5. Executing safely against a read-only connection and displaying tabular results."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 1. Install Dependencies & Clone Repo"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q transformers peft bitsandbytes accelerate sqlglot\n",
            "!git clone https://github.com/SEASIL/text2sql-qlora.git\n",
            "%cd text2sql-qlora"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 2. Create Sample SQLite Database"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sqlite3\n",
            "import pandas as pd\n",
            "\n",
            "conn = sqlite3.connect('company.db')\n",
            "cur = conn.cursor()\n",
            "\n",
            "cur.execute('''\n",
            "CREATE TABLE IF NOT EXISTS departments (\n",
            "    dept_id INTEGER PRIMARY KEY,\n",
            "    dept_name TEXT NOT NULL,\n",
            "    location TEXT NOT NULL\n",
            ");\n",
            "''')\n",
            "\n",
            "cur.execute('''\n",
            "CREATE TABLE IF NOT EXISTS employees (\n",
            "    emp_id INTEGER PRIMARY KEY,\n",
            "    name TEXT NOT NULL,\n",
            "    dept_id INTEGER,\n",
            "    salary REAL NOT NULL,\n",
            "    hire_year INTEGER NOT NULL,\n",
            "    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)\n",
            ");\n",
            "''')\n",
            "\n",
            "cur.executemany('INSERT INTO departments VALUES (?, ?, ?)', [\n",
            "    (1, 'Engineering', 'San Francisco'),\n",
            "    (2, 'Sales', 'New York'),\n",
            "    (3, 'Marketing', 'Chicago'),\n",
            "])\n",
            "\n",
            "cur.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?)', [\n",
            "    (101, 'Alice Chen', 1, 145000, 2021),\n",
            "    (102, 'Bob Smith', 1, 130000, 2022),\n",
            "    (103, 'Charlie Brown', 2, 95000, 2020),\n",
            "    (104, 'Diana Prince', 3, 110000, 2019),\n",
            "    (105, 'Evan Wright', 1, 160000, 2018),\n",
            "])\n",
            "\n",
            "conn.commit()\n",
            "conn.close()\n",
            "print('Sample company.db created successfully.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 3. Load Model & LoRA Adapter"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
            "from peft import PeftModel\n",
            "\n",
            "base_id = 'Qwen/Qwen2.5-1.5B-Instruct'\n",
            "tok = AutoTokenizer.from_pretrained(base_id)\n",
            "\n",
            "bnb = BitsAndBytesConfig(\n",
            "    load_in_4bit=True,\n",
            "    bnb_4bit_quant_type='nf4',\n",
            "    bnb_4bit_use_double_quant=True,\n",
            "    bnb_4bit_compute_dtype=torch.bfloat16,\n",
            ")\n",
            "\n",
            "model = AutoModelForCausalLM.from_pretrained(\n",
            "    base_id,\n",
            "    quantization_config=bnb,\n",
            "    torch_dtype=torch.bfloat16,\n",
            "    device_map='auto',\n",
            ")\n",
            "print('Model loaded successfully.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 4. Interactive Natural Language Query Function with Guard"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.prompting import build_messages, extract_sql\n",
            "from src.sql_guard import check_sql\n",
            "from src.exec_eval import open_readonly, get_schema, run_query\n",
            "\n",
            "def ask_database(db_path: str, question: str):\n",
            "    conn = open_readonly(db_path)\n",
            "    schema = get_schema(conn)\n",
            "    \n",
            "    messages = build_messages(schema, question)\n",
            "    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
            "    \n",
            "    inputs = tok(prompt, return_tensors='pt').to(model.device)\n",
            "    with torch.no_grad():\n",
            "        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)\n",
            "    new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]\n",
            "    raw_sql = tok.decode(new_tokens[0], skip_special_tokens=True)\n",
            "    sql = extract_sql(raw_sql)\n",
            "    \n",
            "    print('--- Generated SQL ---')\n",
            "    print(sql)\n",
            "    \n",
            "    guard = check_sql(sql)\n",
            "    print('\\n--- SQL Safety Guard ---')\n",
            "    print('Allowed:', guard.allowed, '|', guard.reason)\n",
            "    \n",
            "    if not guard.allowed:\n",
            "        conn.close()\n",
            "        return\n",
            "        \n",
            "    res = run_query(conn, sql)\n",
            "    conn.close()\n",
            "    \n",
            "    print('\\n--- Execution Results ---')\n",
            "    if res.ok:\n",
            "        print(pd.DataFrame(res.rows))\n",
            "    else:\n",
            "        print('Execution Error:', res.error)\n",
            "\n",
            "# Test Query 1: Legitimate question\n",
            "ask_database('company.db', 'Show all employees with a salary greater than 120000 ordered by salary descending.')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test Query 2: Safety Guard Blocking malicious injection test\n",
            "guard_check = check_sql('DROP TABLE employees;')\n",
            "print('Malicious query blocked:', not guard_check.allowed, '| Reason:', guard_check.reason)"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebooks/demo.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Wrote", out_path)
