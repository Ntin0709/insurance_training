"""LoRA fine-tuning entrypoint for a small tool-calling LLM.

Install optional dependencies first:

python3 -m pip install "transformers>=4.41" "datasets>=2.19" "trl>=0.9" "peft>=0.11" accelerate bitsandbytes

Example:

python3 scripts/export_llm_dataset.py
python3 scripts/train_small_llm_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset data/banking_tool_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_optional_training_stack():
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing LLM training dependencies. Install with:\n"
            'python3 -m pip install "transformers>=4.41" "datasets>=2.19" '
            '"trl>=0.9" "peft>=0.11" accelerate bitsandbytes'
        ) from exc
    return load_dataset, LoraConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, SFTTrainer


def format_record(record: dict) -> str:
    chunks = []
    for message in record["messages"]:
        role = message["role"]
        content = message["content"]
        chunks.append(f"<|{role}|>\n{content}")
    return "\n".join(chunks) + "\n<|end|>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="data/banking_tool_sft.jsonl")
    parser.add_argument("--output-dir", default="outputs/banking-tool-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}. Run scripts/export_llm_dataset.py first.")

    load_dataset, LoraConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, SFTTrainer = load_optional_training_stack()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(lambda row: {"text": format_record(row)})

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    Path(args.output_dir, "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
