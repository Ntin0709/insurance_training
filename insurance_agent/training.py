"""Training data adapters and deterministic rewards for insurance models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GEMMA3_MODEL = "google/gemma-3-4b-it"
DEFAULT_SYSTEM_PROMPT = (
    "You are an insurance advisor for an Indian bank. Ask suitability questions before recommending. "
    "Use official policy evidence, cite source_ids, never invent premiums, and mention GST quote, "
    "eligibility, exclusions, waiting periods, network/cashless checks, and underwriting."
)


@dataclass(frozen=True)
class TrainingRecordCounts:
    train: int
    validation: int


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def apply_config_defaults(parser: Any, argv: list[str] | None = None) -> Any:
    """Apply simple YAML/JSON config defaults before normal argparse parsing.

    CLI arguments still win. This intentionally supports only the flat config
    files used by the training bundle, avoiding a runtime PyYAML dependency.
    """

    config_parser = __import__("argparse").ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    known, _ = config_parser.parse_known_args(argv)
    if known.config:
        parser.set_defaults(**load_flat_config(known.config))
    parser.add_argument("--config", default=known.config, help="Optional flat YAML/JSON config file. CLI flags override config values.")
    return parser.parse_args(argv)


def resolve_precision(torch_module: Any, requested: str) -> dict[str, bool]:
    requested = requested.lower()
    if requested == "auto":
        if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False}
        if torch_module.cuda.is_available():
            return {"bf16": False, "fp16": True}
        return {"bf16": False, "fp16": False}
    if requested == "bf16":
        return {"bf16": True, "fp16": False}
    if requested == "fp16":
        return {"bf16": False, "fp16": True}
    if requested in {"fp32", "no", "none"}:
        return {"bf16": False, "fp16": False}
    raise ValueError(f"Unknown precision: {requested}")


def model_torch_dtype(torch_module: Any, requested: str) -> Any:
    precision = resolve_precision(torch_module, requested)
    if precision["bf16"]:
        return torch_module.bfloat16
    if precision["fp16"]:
        return torch_module.float16
    return torch_module.float32


def load_flat_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file missing: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
        return _normalize_config_keys(loaded if isinstance(loaded, dict) else {})
    values: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line or raw_line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _parse_scalar(value.strip())
    return _normalize_config_keys(values)


def _normalize_config_keys(values: dict[str, Any]) -> dict[str, Any]:
    return {str(key).replace("-", "_"): value for key, value in values.items()}


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def messages_to_text(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render messages with the model chat template, falling back to a stable plain format."""

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    chunks = []
    for message in messages:
        chunks.append(f"<|{message['role']}|>\n{message['content']}")
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "\n".join(chunks)


def normalize_sft_row(row: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"SFT row {row.get('id', '<unknown>')} missing messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"SFT row {row.get('id', '<unknown>')} must end with an assistant label")
    prompt_messages = messages[:-1]
    full_messages = messages
    return prompt_messages, full_messages


def format_dpo_row(tokenizer: Any, row: dict[str, Any]) -> dict[str, str]:
    prompt_messages = row.get("prompt")
    if not isinstance(prompt_messages, list):
        raise ValueError(f"DPO row {row.get('id', '<unknown>')} missing prompt messages")
    return {
        "prompt": messages_to_text(tokenizer, prompt_messages, add_generation_prompt=True),
        "chosen": str(row["chosen"]),
        "rejected": str(row["rejected"]),
    }


def format_grpo_row(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt_messages = row.get("prompt")
    if not isinstance(prompt_messages, list):
        raise ValueError(f"RLVR row {row.get('id', '<unknown>')} missing prompt messages")
    return {
        "prompt": messages_to_text(tokenizer, prompt_messages, add_generation_prompt=True),
        "reference": row.get("reference", {}),
        "reward_spec": row.get("reward_spec", {}),
        "metadata": row.get("metadata", {}),
    }


def tokenize_sft_row(tokenizer: Any, row: dict[str, Any], max_seq_length: int) -> dict[str, list[int]]:
    prompt_messages, full_messages = normalize_sft_row(row)
    prompt_text = messages_to_text(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = messages_to_text(tokenizer, full_messages, add_generation_prompt=False)
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_length)
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    prompt_len = min(len(prompt_tokens), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    if labels and all(label == -100 for label in labels):
        labels[-1] = input_ids[-1]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def completion_reward(completion: str, reference: dict[str, Any], reward_spec: dict[str, float] | None = None) -> float:
    """Score an insurance tool-trace completion using deterministic RLVR checks."""

    weights = reward_spec or {}
    parsed = _parse_json_object(completion)
    score = 0.0
    if parsed is not None:
        score += float(weights.get("valid_json", 1.0))
    else:
        return -2.0

    expected_calls = reference.get("tool_calls", [])
    expected_final = reference.get("expected_final", {})
    predicted_calls = parsed.get("tool_calls", []) if isinstance(parsed.get("tool_calls"), list) else []
    final_response = parsed.get("final_response", {})
    if _is_empty_final_response(final_response):
        score -= float(weights.get("non_empty_final_response", 1.5))
    predicted_tools = [call.get("tool_name") for call in predicted_calls if isinstance(call, dict)]
    expected_tools = [call.get("tool_name") for call in expected_calls if isinstance(call, dict)]

    if predicted_tools == expected_tools:
        score += float(weights.get("correct_workflow_order", 2.0))
    elif _is_ordered_subsequence(predicted_tools, expected_tools):
        score += float(weights.get("correct_workflow_order", 2.0)) * 0.4

    expected_categories = set(expected_final.get("expected_categories", []))
    must_cite = set(expected_final.get("must_cite_source_ids", []))
    must_ask = set(expected_final.get("must_ask_slots", []))
    predicted_sources: set[str] = set()
    predicted_slots: set[str] = set()
    recommended_categories: set[str] = set()

    for call in predicted_calls:
        if not isinstance(call, dict):
            continue
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        predicted_slots.update(args.get("slots", []) or [])
        predicted_slots.update(args.get("missing_slots", []) or [])
        predicted_sources.update(args.get("source_ids", []) or [])
        recommended_categories.update(args.get("recommended_categories", []) or [])

    if must_ask.issubset(predicted_slots):
        score += float(weights.get("required_slots_present", 1.5))

    if not expected_categories:
        if predicted_tools == ["refuse_premature_recommendation", "finish"]:
            score += float(weights.get("safe_refusal_when_premature", 2.0))
        if "generate_recommendation" in predicted_tools:
            score -= 2.0
    else:
        if "retrieve_policy_evidence" in predicted_tools and "generate_recommendation" in predicted_tools:
            if predicted_tools.index("retrieve_policy_evidence") < predicted_tools.index("generate_recommendation"):
                score += float(weights.get("retrieval_before_recommendation", 1.0))
        if expected_categories.issubset(recommended_categories):
            score += 1.0
        if must_cite and must_cite.issubset(predicted_sources):
            score += float(weights.get("source_ids_from_retrieval", 2.0))

    text = json.dumps(final_response, ensure_ascii=False).lower()
    has_substantive_final = not _is_empty_final_response(final_response)
    has_recommendation = _has_recommendation(final_response)
    if has_substantive_final and not _has_fake_premium(text):
        score += float(weights.get("no_fake_premium", 1.0))
    if has_recommendation and _has_compliance_caveat(text):
        score += float(weights.get("compliance_caveat", 1.0))
    return score


def batch_completion_rewards(
    completions: list[str],
    references: list[dict[str, Any]],
    reward_specs: list[dict[str, float]] | None = None,
) -> list[float]:
    specs = reward_specs or [{} for _ in completions]
    return [completion_reward(completion, reference, spec) for completion, reference, spec in zip(completions, references, specs)]


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _is_ordered_subsequence(predicted: list[str | None], expected: list[str | None]) -> bool:
    if not predicted:
        return False
    idx = 0
    for item in expected:
        if idx < len(predicted) and predicted[idx] == item:
            idx += 1
    return idx == len(predicted)


def _has_fake_premium(text: str) -> bool:
    premium_claim = re.search(r"(premium|₹|rs\.?|inr)\s*[:=]?\s*[₹]?\s*\d{3,}", text)
    quote_context = "premium quote" in text or "official quote" in text or "indicative" in text
    return bool(premium_claim and not quote_context)


def _has_compliance_caveat(text: str) -> bool:
    terms = ("policy wording", "premium quote", "gst", "eligibility", "exclusions", "waiting", "underwriting")
    return any(term in text for term in terms)


def _is_empty_final_response(final_response: Any) -> bool:
    if final_response is None:
        return True
    if isinstance(final_response, str):
        return not final_response.strip()
    if isinstance(final_response, dict):
        if not final_response:
            return True
        serialized = json.dumps(final_response, ensure_ascii=False).strip().lower()
        return serialized in {"{}", "{\"recommendation\": []}", "{\"tool_calls\": []}"}
    if isinstance(final_response, list):
        return len(final_response) == 0
    return False


def _has_recommendation(final_response: Any) -> bool:
    if not isinstance(final_response, dict):
        return False
    recommendation = final_response.get("recommendation")
    return isinstance(recommendation, list) and any(isinstance(item, dict) and item.get("category") for item in recommendation)
