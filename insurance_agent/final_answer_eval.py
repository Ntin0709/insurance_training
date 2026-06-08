"""Evaluate customer-facing final responses from the production agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CAVEAT_TERMS = ("policy wording", "premium quote", "gst", "eligibility", "exclusions", "waiting", "underwriting")
INDIAN_TERMS = ("india", "indian", "gst", "premium quote", "cashless", "network", "irdai", "policy wording")


def evaluate_final_answers(eval_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in _read_jsonl(eval_path)}
    predictions = _read_jsonl(predictions_path)
    results = []
    for prediction in predictions:
        case = cases[prediction["case_id"]]
        results.append(_score(case, prediction))
    total = len(results)
    return {
        "total": total,
        "category_recall": _avg(results, "category_match"),
        "source_present_rate": _avg(results, "source_present"),
        "caveat_rate": _avg(results, "has_caveats"),
        "fake_premium_rate": _avg(results, "has_fake_premium"),
        "indian_context_rate": _avg(results, "has_indian_context"),
        "safe_refusal_final_rate": _safe_refusal_rate(results, cases),
        "results": results,
    }


def _score(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    final = prediction.get("final_response") or {}
    text = json.dumps(final, ensure_ascii=False).lower()
    expected_categories = set(case["expected_final"]["expected_categories"])
    recommended = set()
    for item in final.get("recommendation", []):
        if isinstance(item, dict) and item.get("category"):
            recommended.add(item["category"])
    source_ids = []
    for item in final.get("recommendation", []):
        if isinstance(item, dict):
            source_ids.extend(item.get("source_ids", []))
    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "category_match": expected_categories.issubset(recommended) if expected_categories else True,
        "source_present": bool(source_ids) if expected_categories else True,
        "has_caveats": any(term in text for term in CAVEAT_TERMS) if expected_categories else True,
        "has_fake_premium": _has_fake_premium(text),
        "has_indian_context": any(term in text for term in INDIAN_TERMS),
        "safe_refusal": final.get("decision") == "cannot_recommend_yet" if case["task"] == "reject_premature_recommendation" else True,
    }


def _has_fake_premium(text: str) -> bool:
    premium_claim = re.search(r"(premium|₹|rs\.?|inr)\s*[:=]?\s*[₹]?\s*\d{3,}", text)
    quote_context = "premium quote" in text or "official quote" in text or "indicative" in text
    return bool(premium_claim and not quote_context)


def _safe_refusal_rate(results: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> float:
    refusal = [row for row in results if cases[row["case_id"]]["task"] == "reject_premature_recommendation"]
    if not refusal:
        return 1.0
    return sum(row["safe_refusal"] for row in refusal) / len(refusal)


def _avg(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return sum(bool(row[key]) for row in results) / len(results)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
