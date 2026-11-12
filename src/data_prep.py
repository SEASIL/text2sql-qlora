"""Build train/val JSONL files from b-mc2/sql-create-context.

Each line: {"messages": [system, user, assistant]} using src.prompting.

Leakage control: sql-create-context is (as far as I know) derived partly from Spider
and WikiSQL, so we optionally drop any training question that also appears in the
Spider dev set, which is what we evaluate on. Pass --spider-dev to enable it.
ALWAYS run with --spider-dev before reporting results.
"""
import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from src.prompting import build_messages


def norm_question(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="b-mc2/sql-create-context")
    ap.add_argument("--num-train", type=int, default=20000)
    ap.add_argument("--num-val", type=int, default=500)
    ap.add_argument("--spider-dev", default=None, help="path to spider dev.json for decontamination")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="train")
    rows = [dict(r) for r in ds]
    print(f"loaded {len(rows)} rows")

    blocked = set()
    if args.spider_dev:
        dev = json.loads(Path(args.spider_dev).read_text(encoding="utf-8"))
        blocked = {norm_question(x["question"]) for x in dev}
        before = len(rows)
        rows = [r for r in rows if norm_question(r["question"]) not in blocked]
        print(f"decontamination removed {before - len(rows)} rows overlapping Spider dev questions")
    else:
        print("WARNING: no --spider-dev given, results may be contaminated")

    random.Random(args.seed).shuffle(rows)
    val = rows[: args.num_val]
    train = rows[args.num_val : args.num_val + args.num_train]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in split:
                msgs = build_messages(r["context"], r["question"], r["answer"])
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        print(f"wrote {len(split)} rows -> {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
