# Text-to-SQL with QLoRA

Fine-tune a small open LLM (Qwen2.5-1.5B-Instruct) to turn English questions into SQL,
measure it honestly against the untuned base model, and ship it so anyone can try it.

> **Status: base scaffold.** Code is written but training/eval have not been run yet.
> The results table below is intentionally empty. Fill it only with numbers you measured.

## What this project proves
- **Fine-tuning:** QLoRA (4-bit base + LoRA adapters) trained on a single 8GB GPU.
- **Evaluation:** execution accuracy on Spider dev (run the SQL, compare result sets),
  not just string matching. Base vs fine-tuned, same prompt, same settings.
- **Safety:** every generated query passes a read-only SQL guard and runs on a
  read-only database connection.
- **Shipping:** merged + quantized GGUF, model on Hugging Face Hub, Colab demo, CI.

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

# 6. Try it
python -m src.infer --db data/spider/database/concert_singer/concert_singer.sqlite \
    --adapter outputs/qwen2.5-1.5b-text2sql-lora --question "How many singers are there?"

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
Upload the adapter and GGUF to the Hugging Face Hub (public repos are free), then link a
Google Colab notebook and a short recorded demo from this README.
Note: Hugging Face Spaces now requires a paid plan for Gradio/Docker apps, so do not
depend on it for the live demo.

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
