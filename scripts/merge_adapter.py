"""Merge the LoRA adapter into the base model (needed before GGUF conversion).

    python scripts/merge_adapter.py --adapter outputs/qwen2.5-1.5b-text2sql-lora \
        --out outputs/merged-fp16
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.out)
    print("merged model saved to", args.out)


if __name__ == "__main__":
    main()
