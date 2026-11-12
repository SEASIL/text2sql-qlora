"""QLoRA fine-tuning with TRL's SFTTrainer.

Usage:  python -m src.train_qlora --config configs/qwen2.5-1.5b-qlora.yaml

NOTE: TRL/transformers change argument names between releases (e.g. max_length vs
max_seq_length, processing_class vs tokenizer). If a call below errors, check the
installed version's docs and adapt, then pin versions in requirements.txt.
"""
import argparse
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns")

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-1.5b-qlora.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(example):
        return {"text": tok.apply_chat_template(example["messages"], tokenize=False)}

    data = load_dataset("json", data_files={
        "train": cfg["data"]["train_file"], "val": cfg["data"]["val_file"]})
    data = data.map(to_text, remove_columns=data["train"].column_names)

    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb,
        torch_dtype=getattr(torch, q["compute_dtype"]), device_map={"": 0})
    model.config.use_cache = False

    l = cfg["lora"]
    peft_config = LoraConfig(
        r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
        target_modules=l["target_modules"], bias="none", task_type="CAUSAL_LM")

    t = cfg["training"]
    sft_args = SFTConfig(
        output_dir=t["output_dir"], run_name=t["run_name"], seed=cfg["seed"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"], max_length=t["max_length"],
        logging_steps=t["logging_steps"], eval_strategy="steps", eval_steps=t["eval_steps"],
        save_steps=t["save_steps"], save_total_limit=t["save_total_limit"],
        optim=t["optim"], bf16=True, gradient_checkpointing=True,
        dataset_text_field="text", packing=False, report_to=t["report_to"])

    # SFTTrainer prepares the 4-bit model for k-bit training when a peft_config is passed.
    trainer = SFTTrainer(
        model=model, args=sft_args, train_dataset=data["train"], eval_dataset=data["val"],
        peft_config=peft_config, processing_class=tok)
    trainer.train()
    trainer.save_model(t["output_dir"])
    tok.save_pretrained(t["output_dir"])
    print("saved adapter to", t["output_dir"])


if __name__ == "__main__":
    main()
