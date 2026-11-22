# Text-to-SQL with QLoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEASIL/text2sql-qlora/blob/main/notebooks/demo.ipynb)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GPTpasta%2Fqwen2.5--1.5b--text2sql--lora-blue)](https://huggingface.co/GPTpasta/qwen2.5-1.5b-text2sql-lora)
[![CI](https://github.com/SEASIL/text2sql-qlora/actions/workflows/ci.yml/badge.svg)](https://github.com/SEASIL/text2sql-qlora/actions/workflows/ci.yml)

Fine-tune a small open LLM (`Qwen/Qwen2.5-1.5B-Instruct`) to turn English questions into SQL,
measure it honestly against the untuned base model, and ship it with end-to-end evaluation, guardrails, and CPU quantization.

## What this project proves
- **Fine-tuning:** QLoRA (4-bit base + LoRA adapters) trained on a single 8GB RTX 4060 GPU (peak VRAM: 3.1 GB).
- **Evaluation:** Execution accuracy on Spider dev (run the SQL, compare result sets), not just string matching.
- **Safety:** Every generated query passes an AST-based SQL guard and executes on a read-only SQLite connection.
- **Shipping:** Merged + quantized GGUF (`Q4_K_M`, 934 MB), Hugging Face Hub release, Google Colab demo, Streamlit web app, and GitHub Actions CI.

## How it works
```
 train:   sql-create-context (schema + question -> SQL)
              | src/data_prep.py  (format with src/prompting.py, drop Spider-dev overlap)
              v
          QLoRA fine-tune  (src/train_qlora.py, tracked in MLflow)
              v
          LoRA adapter  ->  merge  ->  GGUF (Q4_K_M)  ->  HF Hub / Colab demo

 eval:    Spider dev (real SQLite DBs)
              | schema is read from each .sqlite file
              v
          model writes SQL -> sql_guard -> run on read-only DB -> compare rows with gold
              v
          execution accuracy + per-example error types (error analysis)
```
Training data and evaluation data are deliberately different (cross-dataset), so the
reported number reflects generalization to unseen schemas, not memorization.

## Layout
| Path | Purpose |
|---|---|
| `src/prompting.py` | Single source of truth for the prompt format (train = eval = inference) |
| `src/data_prep.py` | Builds `train.jsonl` / `val.jsonl`, with Spider-dev decontamination |
| `src/train_qlora.py` | QLoRA training (TRL SFTTrainer, MLflow logging) |
| `src/evaluate.py` | Execution-accuracy eval + per-example predictions |
| `src/exec_eval.py` | Read-only SQLite execution, timeout, result-set comparison |
| `src/sql_guard.py` | Allows only a single SELECT/WITH; blocks everything else |
| `src/infer.py` | CLI: ask a question about a `.sqlite` file |
| `scripts/merge_adapter.py` | Merge adapter into base model for export |
| `configs/` | Training hyperparameters |
| `tests/` | Unit tests for the guard and the executor |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -m pytest -q
```

## Run it
```bash
# 1. Get the Spider dataset (official release) and unzip so you have:
#    data/spider/dev.json   and   data/spider/database/<db_id>/<db_id>.sqlite

# 2. Prepare training data (decontaminated against Spider dev)
python -m src.data_prep --spider-dev data/spider/dev.json

# 3. Baseline: evaluate the untuned model
python -m src.evaluate --label base --spider-dir data/spider

# 4. Train
python -m src.train_qlora --config configs/qwen2.5-1.5b-qlora.yaml

# 5. Evaluate the fine-tuned model
python -m src.evaluate --label finetuned --spider-dir data/spider \
    --adapter outputs/qwen2.5-1.5b-text2sql-lora

# 6. Try it (CLI or Web UI)
# CLI query:
python -m src.infer --db data/spider/database/concert_singer/concert_singer.sqlite \
    --adapter outputs/qwen2.5-1.5b-text2sql-lora --question "How many singers are there?"

# Interactive Streamlit Web App:
streamlit run app.py

# MLflow UI for training curves
mlflow ui
```

## Export for free hosting
```bash
python scripts/merge_adapter.py --adapter outputs/qwen2.5-1.5b-text2sql-lora --out outputs/merged-fp16
# Convert + quantize with llama.cpp (clone ggml-org/llama.cpp):
python convert_hf_to_gguf.py outputs/merged-fp16 --outfile outputs/model-f16.gguf
./llama-quantize outputs/model-f16.gguf outputs/model-Q4_K_M.gguf Q4_K_M
```
## Free Public Hosting & Demo

- **Hugging Face Hub Repository:** [GPTpasta/qwen2.5-1.5b-text2sql-lora](https://huggingface.co/GPTpasta/qwen2.5-1.5b-text2sql-lora) (contains the LoRA adapter safetensors and the quantized `model-Q4_K_M.gguf` file).
- **Interactive Google Colab Notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb) (runs in free-tier Colab with T4 GPU or CPU).
- **In-Depth Error Analysis:** [`docs/error_analysis.md`](docs/error_analysis.md) (comprehensive taxonomy and failure case studies).

## Results (fill in with MEASURED numbers only)
| Model | Execution accuracy (Spider dev) | Valid-SQL rate | Notes |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 45.65% (472/1034) | 71.66% (741/1034) | 4-bit NF4, commit `ac7acb2` |
| + QLoRA fine-tune | 15.76% (163/1034) | 49.71% (514/1034) | 4-bit NF4 + LoRA r=16, commit `7f44940` (cross-dataset degradation; see Error Analysis) |
| Hosted API baseline (optional) | TBD | TBD | |

## Known limitations (keep this section honest)
- Spider is SQLite-flavoured; queries for other dialects may need adjusting.
- Result comparison ignores column order, which is slightly lenient.
- Cross-dataset eval: some Spider-style questions may still be near-duplicates of training data
  even after exact-match decontamination.
