"""Gemma 3 preference trainer for insurance advisor DPO data.

This script intentionally avoids TRL's DPOTrainer preprocessing path. In
TRL 0.26, GemmaTokenizerFast can be misclassified as a processor and TRL then
looks for ``tokenizer.tokenizer``. The custom trainer below works directly on
tokenized prompt/chosen/rejected strings and keeps the release bundle stable.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.training import DEFAULT_GEMMA3_MODEL, apply_config_defaults, format_dpo_row, model_torch_dtype, resolve_precision, write_json


def optional_stack():
    try:
        import torch
        import torch.nn.functional as F
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Install DPO dependencies first:\n"
            'python3 -m pip install "transformers>=4.51" "datasets>=2.19" '
            '"peft>=0.11" "accelerate>=0.30" bitsandbytes'
        ) from exc
    return (
        torch,
        F,
        load_dataset,
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )


@dataclass
class PreferenceCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        chosen = [{"input_ids": row["chosen_input_ids"], "attention_mask": row["chosen_attention_mask"]} for row in features]
        rejected = [{"input_ids": row["rejected_input_ids"], "attention_mask": row["rejected_attention_mask"]} for row in features]
        chosen_labels = [row["chosen_labels"] for row in features]
        rejected_labels = [row["rejected_labels"] for row in features]

        chosen_batch = self.tokenizer.pad(chosen, padding=True, return_tensors="pt")
        rejected_batch = self.tokenizer.pad(rejected, padding=True, return_tensors="pt")
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "chosen_labels": _pad_labels(chosen_labels, chosen_batch["input_ids"].shape[1]),
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "rejected_labels": _pad_labels(rejected_labels, rejected_batch["input_ids"].shape[1]),
        }


def build_preference_trainer(base_trainer_cls: Any, torch: Any, F: Any, beta: float) -> type:
    class InsurancePreferenceTrainer(base_trainer_cls):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any) -> Any:
            chosen_logps = _sequence_logps(
                model,
                inputs["chosen_input_ids"],
                inputs["chosen_attention_mask"],
                inputs["chosen_labels"],
            )
            rejected_logps = _sequence_logps(
                model,
                inputs["rejected_input_ids"],
                inputs["rejected_attention_mask"],
                inputs["rejected_labels"],
            )
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = None

            disable_adapter = getattr(model, "disable_adapter", None)
            if callable(disable_adapter):
                with torch.no_grad():
                    with disable_adapter():
                        ref_chosen_logps = _sequence_logps(
                            model,
                            inputs["chosen_input_ids"],
                            inputs["chosen_attention_mask"],
                            inputs["chosen_labels"],
                        )
                        ref_rejected_logps = _sequence_logps(
                            model,
                            inputs["rejected_input_ids"],
                            inputs["rejected_attention_mask"],
                            inputs["rejected_labels"],
                        )
                ref_logratios = ref_chosen_logps - ref_rejected_logps

            logits = pi_logratios if ref_logratios is None else pi_logratios - ref_logratios
            loss = -F.logsigmoid(beta * logits).mean()
            if return_outputs:
                return loss, {"chosen_logps": chosen_logps.detach(), "rejected_logps": rejected_logps.detach()}
            return loss

    return InsurancePreferenceTrainer


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

    (
        torch,
        F,
        load_dataset,
        LoraConfig,
        PeftModel,
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
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
    model.config.use_cache = False

    data_files = {"train": str(train_path), "validation": str(validation_path)}
    dataset = load_dataset("json", data_files=data_files)
    tokenized = dataset.map(
        lambda row: tokenize_dpo_row(tokenizer, row, args.max_prompt_length, args.max_length),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing DPO rows",
    )

    training_args = TrainingArguments(**_accepted_training_args(TrainingArguments, args))
    trainer_cls = build_preference_trainer(Trainer, torch, F, args.beta)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=PreferenceCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_json(Path(args.output_dir) / "training_config.json", vars(args))
    print(json.dumps({"output_dir": args.output_dir, "train_file": str(train_path)}, indent=2))


def tokenize_dpo_row(tokenizer: Any, row: dict[str, Any], max_prompt_length: int, max_length: int) -> dict[str, list[int]]:
    formatted = format_dpo_row(tokenizer, row)
    prompt_ids = tokenizer(
        formatted["prompt"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    )["input_ids"]
    chosen_ids = tokenizer(str(formatted["chosen"]), add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(str(formatted["rejected"]), add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        chosen_ids = chosen_ids + [eos_id]
        rejected_ids = rejected_ids + [eos_id]
    chosen = _truncate_pair(prompt_ids, chosen_ids, max_length)
    rejected = _truncate_pair(prompt_ids, rejected_ids, max_length)
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "chosen_labels": chosen["labels"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "rejected_labels": rejected["labels"],
    }


def _truncate_pair(prompt_ids: list[int], answer_ids: list[int], max_length: int) -> dict[str, list[int]]:
    max_answer_len = max(1, max_length - len(prompt_ids))
    answer_ids = answer_ids[:max_answer_len]
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
    answer_start = max(0, len(input_ids) - len(answer_ids))
    labels = [-100] * answer_start + input_ids[answer_start:]
    if labels and all(label == -100 for label in labels):
        labels[-1] = input_ids[-1]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _pad_labels(labels: list[list[int]], max_len: int) -> Any:
    import torch

    padded = [row + [-100] * (max_len - len(row)) for row in labels]
    return torch.tensor(padded, dtype=torch.long)


def _sequence_logps(model: Any, input_ids: Any, attention_mask: Any, labels: Any) -> Any:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    token_logps = logits.log_softmax(dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * loss_mask).sum(dim=-1)


def _accepted_training_args(training_args_cls: Any, args: argparse.Namespace) -> dict[str, Any]:
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
        "logging_steps": 10,
        "max_steps": args.max_steps,
        "save_steps": 500,
        "eval_steps": 500,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_total_limit": 3,
        "report_to": [],
        "remove_unused_columns": False,
    }
    signature = inspect.signature(training_args_cls.__init__)
    return {key: value for key, value in values.items() if key in signature.parameters}


if __name__ == "__main__":
    main()
