---
language:
- en
license: apache-2.0
tags:
- text-to-sql
- qlora
- qwen
- llama.cpp
- gguf
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
---

# Qwen2.5-1.5B Text-to-SQL (QLoRA & GGUF Q4_K_M)

This repository provides a lightweight Text-to-SQL model based on `Qwen/Qwen2.5-1.5B-Instruct`, fine-tuned using **QLoRA** on an 8GB NVIDIA GeForce RTX 4060 Laptop GPU, and quantized to **GGUF Q4_K_M** for CPU inference via llama.cpp.

## Model Highlights

- **Base Architecture:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters)
- **Quantized GGUF Size:** **934.7 MB** (Q4_K_M, 5.08 BPW)
- **CPU Inference Speed:** **262 tokens/sec** prompt eval, **21.9 tokens/sec** generation (~45.6 ms/token on 4 CPU threads)
- **Safety Guard:** Built-in SQL parser ensuring queries are strictly single read-only `SELECT`/`WITH` statements (blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, multiple statements, and administrative pragmas).

---

## Benchmark & Empirical Findings

Evaluated on the full **Spider dev benchmark** (1,034 complex multi-table queries):

| Configuration | Execution Accuracy | Valid-SQL Rate | Memory Footprint |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (Base, 4-bit) | 45.65% (472/1034) | 71.66% (741/1034) | ~2.2 GB VRAM |
| + QLoRA Fine-Tune (`b-mc2/sql-create-context`) | 15.76% (163/1034) | 49.71% (514/1034) | ~3.1 GB VRAM |
| GGUF Q4_K_M (CPU) | Matched fp16 | Matched fp16 | **< 1.2 GB RAM** |

### The Value of Honest ML Engineering (Cross-Dataset Negative Transfer)
Fine-tuning on 20,000 samples from `b-mc2/sql-create-context` caused domain shift on multi-table relational benchmarks like Spider:
- `sql-create-context` primarily consists of WikiSQL single-table synthetic schemas.
- While the adapter successfully learned to enforce markdown-free raw SQL formatting, it overfitted to synthetic `T1/T2` alias conventions, causing alias misattribution on multi-table DDL schemas.
- For full error taxonomy, failure cases, and mitigation strategies, see `docs/error_analysis.md`.

---

## Quickstart

### 1. Python (Transformers + PEFT)

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "your-username/qwen2.5-1.5b-text2sql-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)

prompt = """<|im_start|>system
You are a careful SQL assistant. Given a database schema and a question, write one read-only SQL SELECT query that answers the question. Return only the SQL, with no explanation.<|im_end|>
<|im_start|>user
Schema:
CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);

Question: Who earns the highest salary in the Engineering department?<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### 2. llama.cpp (GGUF CPU Execution)

```bash
# Run locally with pure CPU on any computer (sub-1GB memory footprint)
./llama-cli -m model-Q4_K_M.gguf -p "Schema:\nCREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);\n\nQuestion: Who earns the highest salary in Engineering?" -n 64
```
