"""Deterministic quality gates for insurance synthetic training data."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXACT_PREMIUM_RE = re.compile(r"(premium|₹|rs\.?|inr)\s*[:=]?\s*[₹]?\s*\d{3,}", re.IGNORECASE)


def quality_report(
    scenarios: list[Any],
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    dpo_rows: list[dict[str, Any]],
    rlvr_rows: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]] | None = None,
    heldout_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    messages = [case["messages"][-1]["content"] for case in cases]
    normalized_messages = [_normalize_text(message) for message in messages]
    heldout_messages = _heldout_messages(heldout_paths or [])
    overlap = sorted(set(normalized_messages) & heldout_messages)
    case_ids = [case["case_id"] for case in cases]
    pred_ids = [row["case_id"] for row in predictions]
    normal_cases = [case for case in cases if case["task"] != "reject_premature_recommendation"]
    refusal_cases = [case for case in cases if case["task"] == "reject_premature_recommendation"]
    category_counts = Counter(category for case in cases for category in case["expected_final"]["expected_categories"])
    language_counts = Counter(case["language"] for case in cases)
    task_counts = Counter(case["task"] for case in cases)
    scenario_type_counts = Counter(case.get("scenario_type", "unknown") for case in cases)
    risk_tag_counts = Counter(tag for case in cases for tag in case.get("risk_tags", []))
    split_leakage = _split_leakage(splits or {})

    errors: list[str] = []
    warnings: list[str] = []
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate_case_ids")
    if set(case_ids) != set(pred_ids):
        errors.append("prediction_case_id_mismatch")
    if not (len(cases) == len(predictions) == len(sft_rows) == len(dpo_rows) == len(rlvr_rows)):
        errors.append("export_count_mismatch")
    if any(not case["expected_final"]["must_ask_slots"] for case in cases):
        errors.append("missing_required_slots")
    if any(not case["expected_final"]["must_cite_source_ids"] for case in normal_cases):
        errors.append("ungrounded_non_refusal_case")
    if any(case["expected_final"]["must_cite_source_ids"] for case in refusal_cases):
        errors.append("refusal_case_has_sources")
    if split_leakage:
        errors.append("split_message_leakage")
    if overlap:
        errors.append("heldout_benchmark_message_overlap")

    if cases and len(set(normalized_messages)) / len(cases) < 0.985:
        warnings.append("low_message_uniqueness")
    if cases and len(refusal_cases) / len(cases) < 0.08:
        warnings.append("low_refusal_coverage")
    if cases and len(refusal_cases) / len(cases) > 0.25:
        warnings.append("high_refusal_coverage")
    if cases and category_counts and max(category_counts.values()) / sum(category_counts.values()) > 0.35:
        warnings.append("category_distribution_imbalanced")
    if cases and language_counts and max(language_counts.values()) / len(cases) > 0.30:
        warnings.append("language_distribution_imbalanced")
    if any(len(message) < 35 for message in messages):
        warnings.append("short_user_messages_present")

    return {
        "total": len(cases),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "scenario_count": len(scenarios),
        "sft_count": len(sft_rows),
        "dpo_count": len(dpo_rows),
        "rlvr_count": len(rlvr_rows),
        "normal_case_count": len(normal_cases),
        "refusal_case_count": len(refusal_cases),
        "unique_user_messages": len(set(normalized_messages)),
        "duplicate_user_message_count": len(cases) - len(set(normalized_messages)),
        "heldout_overlap_count": len(overlap),
        "heldout_overlap_examples": overlap[:10],
        "split_leakage": split_leakage,
        "language_counts": dict(sorted(language_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "scenario_type_counts": dict(sorted(scenario_type_counts.items())),
        "risk_tag_counts": dict(sorted(risk_tag_counts.items())),
        "exact_premium_request_count": sum(bool(EXACT_PREMIUM_RE.search(message)) for message in messages),
    }


def assert_quality(report: dict[str, Any], allow_warnings: bool = False) -> None:
    if report["errors"]:
        raise ValueError(f"SDG quality errors: {report['errors']}")
    if report["warnings"] and not allow_warnings:
        raise ValueError(f"SDG quality warnings: {report['warnings']}")


def _heldout_messages(paths: list[str | Path]) -> set[str]:
    messages = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "messages" in row and row["messages"]:
                    messages.add(_normalize_text(row["messages"][-1]["content"]))
                elif "user_message" in row:
                    messages.add(_normalize_text(row["user_message"]))
    return messages


def _split_leakage(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    split_messages = {
        name: {_normalize_text(row["messages"][-1]["content"]) for row in rows}
        for name, rows in splits.items()
    }
    leakage = {}
    names = sorted(split_messages)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            count = len(split_messages[left] & split_messages[right])
            if count:
                leakage[f"{left}_vs_{right}"] = count
    return leakage


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def _normalize_text(text: str) -> str:
    return normalize_text(text)
