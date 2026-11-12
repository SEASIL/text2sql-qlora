"""Execution-accuracy evaluation on the Spider dev set.

Expects the official Spider release unzipped at --spider-dir:
    <spider-dir>/dev.json
    <spider-dir>/database/<db_id>/<db_id>.sqlite

Run it once per model so the comparison is apples to apples:
    python -m src.evaluate --label base      --spider-dir data/spider
    python -m src.evaluate --label finetuned --spider-dir data/spider --adapter outputs/qwen2.5-1.5b-text2sql-lora

Writes outputs/eval_<label>.jsonl (every prediction, for error analysis) and
outputs/eval_<label>_summary.json.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.exec_eval import (get_schema, gold_has_order_by, open_readonly,
                           results_match, run_query)
from src.prompting import build_messages, extract_sql
from src.sql_guard import check_sql


def load_model(base: str, adapter: str | None, load_4bit: bool):
    tok = AutoTokenizer.from_pretrained(base, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def generate_batch(tok, model, prompts: list[str], max_new_tokens: int) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--spider-dir", default="data/spider")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--load-4bit", action="store_true", help="match the 4-bit training setup")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    spider = Path(args.spider_dir)
    examples = json.loads((spider / "dev.json").read_text(encoding="utf-8"))
    if args.limit:
        examples = examples[: args.limit]

    tok, model = load_model(args.base_model, args.adapter, args.load_4bit)

    # Build prompts with the schema read straight from each SQLite file.
    schema_cache: dict[str, str] = {}
    prompts = []
    for ex in examples:
        db = ex["db_id"]
        if db not in schema_cache:
            conn = open_readonly(spider / "database" / db / f"{db}.sqlite")
            schema_cache[db] = get_schema(conn)
            conn.close()
        msgs = build_messages(schema_cache[db], ex["question"])
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    records, counts = [], Counter()
    for start in tqdm(range(0, len(examples), args.batch_size)):
        batch_ex = examples[start: start + args.batch_size]
        outputs = generate_batch(tok, model, prompts[start: start + args.batch_size],
                                 args.max_new_tokens)
        for ex, raw in zip(batch_ex, outputs):
            pred = extract_sql(raw)
            db = ex["db_id"]
            conn = open_readonly(spider / "database" / db / f"{db}.sqlite")
            gold = run_query(conn, ex["query"])
            guard = check_sql(pred)
            if not gold.ok:
                error_type = "gold_failed"
                pred_res = None
            elif not guard.allowed:
                error_type = "blocked_by_guard"
                pred_res = None
            else:
                pred_res = run_query(conn, pred)
                if not pred_res.ok:
                    error_type = "invalid_sql"
                elif results_match(gold.rows, pred_res.rows, gold_has_order_by(ex["query"])):
                    error_type = "match"
                else:
                    error_type = "wrong_result"
            conn.close()
            counts[error_type] += 1
            records.append({
                "db_id": db, "question": ex["question"], "gold": ex["query"],
                "pred": pred, "raw_output": raw, "error_type": error_type,
                "error_detail": (pred_res.error if pred_res and not pred_res.ok else guard.reason),
            })

    scored = len(records) - counts["gold_failed"]
    summary = {
        "label": args.label, "base_model": args.base_model, "adapter": args.adapter,
        "n_examples": len(records), "n_scored": scored,
        "execution_accuracy": counts["match"] / scored if scored else 0.0,
        "valid_sql_rate": (counts["match"] + counts["wrong_result"]) / scored if scored else 0.0,
        "counts": dict(counts),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"eval_{args.label}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / f"eval_{args.label}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
