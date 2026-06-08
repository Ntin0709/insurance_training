"""Gemma 3 GRPO/RLVR trainer for insurance tool-use rewards."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.training import DEFAULT_GEMMA3_MODEL, apply_config_defaults, batch_completion_rewards, format_grpo_row, model_torch_dtype, resolve_precision, write_json


def optional_stack():
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install GRPO dependencies first:\n"
            'python3 -m pip install "transformers>=4.51" "datasets>=2.19" '
            '"trl>=0.16" "peft>=0.11" "accelerate>=0.30" bitsandbytes'
        ) from exc
    try:
        from trl import GRPOConfig
    except ImportError as exc:
        raise SystemExit('Your TRL version does not include GRPOConfig. Install "trl>=0.16".') from exc
    return torch, load_dataset, LoraConfig, PeftModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GRPOTrainer, GRPOConfig


def insurance_reward_func(completions: list[Any], reference: list[dict[str, Any]], reward_spec: list[dict[str, float]], **_: Any) -> list[float]:
    texts = [_completion_to_text(completion) for completion in completions]
    return batch_completion_rewards(texts, reference, reward_spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_GEMMA3_MODEL)
    parser.add_argument("--init-adapter", default="", help="Optional PEFT adapter from the previous DPO stage.")
    parser.add_argument("--train-file", default="data/insurance/sdg_rules_50k_balanced/rlvr_train.jsonl")
    parser.add_argument("--validation-file", default="data/insurance/sdg_rules_50k_balanced/rlvr_val.jsonl")
    parser.add_argument("--output-dir", default="outputs/gemma3-insurance-grpo-lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
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

    torch, load_dataset, LoraConfig, PeftModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GRPOTrainer, GRPOConfig = optional_stack()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    dataset = dataset.map(lambda row: format_grpo_row(tokenizer, row), remove_columns=dataset["train"].column_names)

    peft_config = None if args.init_adapter else LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    grpo_args = GRPOConfig(**_accepted_grpo_config_kwargs(GRPOConfig, args))

    trainer_kwargs = {
        "model": model,
        "reward_funcs": [insurance_reward_func],
        "args": grpo_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_json(Path(args.output_dir) / "training_config.json", vars(args))
    print(json.dumps({"output_dir": args.output_dir, "train_file": str(train_path)}, indent=2))


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", json.dumps(completion, ensure_ascii=False)))
    return str(completion)


def _accepted_grpo_config_kwargs(config_cls, args: argparse.Namespace) -> dict[str, object]:
    precision = resolve_precision(__import__("torch"), args.precision)
    values: dict[str, object] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "tf32": precision["bf16"] or precision["fp16"],
        "logging_steps": 10,
        "save_steps": 250,
        "save_total_limit": 3,
        "report_to": [],
        "remove_unused_columns": False,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
    }
    signature = inspect.signature(config_cls.__init__)
    return {key: value for key, value in values.items() if key in signature.parameters}


if __name__ == "__main__":
    main()
