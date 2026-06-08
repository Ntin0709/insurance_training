"""Gemma 3 DPO trainer for insurance advisor preference data."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.training import DEFAULT_GEMMA3_MODEL, apply_config_defaults, format_dpo_row, model_torch_dtype, resolve_precision, write_json


def optional_stack():
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install DPO dependencies first:\n"
            'python3 -m pip install "transformers>=4.51" "datasets>=2.19" '
            '"trl>=0.12" "peft>=0.11" "accelerate>=0.30" bitsandbytes'
        ) from exc
    try:
        from trl import DPOConfig
    except ImportError:
        from transformers import TrainingArguments as DPOConfig
    return torch, load_dataset, LoraConfig, PeftModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DPOTrainer, DPOConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_GEMMA3_MODEL)
    parser.add_argument("--init-adapter", default="", help="Optional PEFT adapter from the previous SFT stage.")
    parser.add_argument("--train-file", default="data/insurance/sdg_rules_50k_balanced/dpo_train.jsonl")
    parser.add_argument("--validation-file", default="")
    parser.add_argument("--output-dir", default="outputs/gemma3-insurance-dpo-lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = apply_config_defaults(parser)

    train_path = Path(args.train_file)
    if not train_path.exists():
        raise SystemExit(f"Train file missing: {train_path}")
    validation_path = Path(args.validation_file) if args.validation_file else train_path

    torch, load_dataset, LoraConfig, PeftModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DPOTrainer, DPOConfig = optional_stack()

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
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    model.config.use_cache = False

    data_files = {"train": str(train_path), "validation": str(validation_path)}
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(lambda row: format_dpo_row(tokenizer, row), remove_columns=dataset["train"].column_names)

    peft_config = None if args.init_adapter else LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dpo_args = DPOConfig(**_accepted_dpo_config_kwargs(DPOConfig, args))

    trainer_kwargs = {
        "model": model,
        "args": dpo_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    signature = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "beta" in signature.parameters:
        trainer_kwargs["beta"] = args.beta
    if "max_length" in signature.parameters:
        trainer_kwargs["max_length"] = args.max_length
    if "max_prompt_length" in signature.parameters:
        trainer_kwargs["max_prompt_length"] = args.max_prompt_length

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_json(Path(args.output_dir) / "training_config.json", vars(args))
    print(json.dumps({"output_dir": args.output_dir, "train_file": str(train_path)}, indent=2))


def _accepted_dpo_config_kwargs(config_cls, args: argparse.Namespace) -> dict[str, object]:
    precision = resolve_precision(__import__("torch"), args.precision)
    values: dict[str, object] = {
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
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_total_limit": 3,
        "report_to": [],
        "remove_unused_columns": False,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_steps": args.max_steps,
    }
    signature = inspect.signature(config_cls.__init__)
    return {key: value for key, value in values.items() if key in signature.parameters}


if __name__ == "__main__":
    main()
