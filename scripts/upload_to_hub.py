"""Upload fine-tuned LoRA adapter and quantized GGUF models to Hugging Face Hub.

Usage:
    python scripts/upload_to_hub.py --repo-id <hf-username>/qwen2.5-1.5b-text2sql-lora --upload-gguf
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF Hub repository name (e.g. username/text2sql-qlora)")
    parser.add_argument("--adapter-dir", default="outputs/qwen2.5-1.5b-text2sql-lora")
    parser.add_argument("--gguf-path", default="outputs/model-Q4_K_M.gguf")
    parser.add_argument("--model-card", default="MODEL_CARD.md")
    parser.add_argument("--upload-gguf", action="store_true", help="Upload GGUF Q4_K_M file alongside adapter")
    args = parser.parse_args()

    api = HfApi()
    print(f"Ensuring repository exists: {args.repo_id}")
    create_repo(repo_id=args.repo_id, exist_ok=True, repo_type="model")

    print(f"Uploading adapter files from {args.adapter_dir}...")
    api.upload_folder(
        folder_path=args.adapter_dir,
        repo_id=args.repo_id,
        ignore_patterns=["checkpoint-*", "*.bin"],
    )

    if Path(args.model_card).exists():
        print(f"Uploading model card {args.model_card} as README.md...")
        api.upload_file(
            path_or_fileobj=args.model_card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
        )

    if args.upload_gguf and Path(args.gguf_path).exists():
        print(f"Uploading GGUF file {args.gguf_path}...")
        api.upload_file(
            path_or_fileobj=args.gguf_path,
            path_in_repo=Path(args.gguf_path).name,
            repo_id=args.repo_id,
        )

    print(f"\nUpload complete! View your model at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
