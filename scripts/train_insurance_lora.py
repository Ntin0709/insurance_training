"""Gemma 3 LoRA SFT trainer for the Indian insurance advisor."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.training import DEFAULT_GEMMA3_MODEL, apply_config_defaults, model_torch_dtype, resolve_precision, tokenize_sft_row, write_json


def optional_stack():
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Install training dependencies first:\n"
            'python3 -m pip install "transformers>=4.51" "datasets>=2.19" '
            '"peft>=0.11" "accelerate>=0.30" bitsandbytes'
        ) from exc
    return torch, load_dataset, LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_len - len(label)))
        import torch

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_GEMMA3_MODEL)
    parser.add_argument("--train-file", default="data/insurance/sdg_rules_50k_balanced/sft_train.jsonl")
    parser.add_argument("--validation-file", default="data/insurance/sdg_rules_50k_balanced/sft_val.jsonl")
    parser.add_argument("--output-dir", default="outputs/gemma3-insurance-sft-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = apply_config_defaults(parser)

    train_path = Path(args.train_file)
    validation_path = Path(args.validation_file)
    if not train_path.exists():
        raise SystemExit(f"Train file missing: {train_path}")
    if not validation_path.exists():
        validation_path = train_path

    (
        torch,
        load_dataset,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    ) = optional_stack()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_torch_dtype(torch, args.precision),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_torch_dtype(torch, args.precision),
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    data_files = {"train": str(train_path), "validation": str(validation_path)}
    dataset = load_dataset("json", data_files=data_files)
    tokenized = dataset.map(
        lambda row: tokenize_sft_row(tokenizer, row, args.max_seq_length),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing SFT rows with assistant-only labels",
    )

    training_args = TrainingArguments(**_accepted_training_args(TrainingArguments, args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_json(Path(args.output_dir) / "training_config.json", vars(args))
    print(json.dumps({"output_dir": args.output_dir, "train_file": str(train_path), "validation_file": str(validation_path)}, indent=2))


def _accepted_training_args(training_args_cls, args: argparse.Namespace) -> dict[str, Any]:
    precision = resolve_precision(__import__("torch"), args.precision)
    values: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "tf32": precision["bf16"] or precision["fp16"],
        "logging_steps": args.logging_steps,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_total_limit": 3,
        "report_to": [],
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
    }
    signature = inspect.signature(training_args_cls.__init__)
    return {key: value for key, value in values.items() if key in signature.parameters}


if __name__ == "__main__":
    main()
