"""Rule-based evaluation for insurance advisor outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from insurance_agent.dataset import QUESTION_SLOTS


CAVEAT_TERMS = (
    "policy wording",
    "eligibility",
    "premium",
    "premium quote",
    "underwriting",
    "exclusion",
    "waiting",
    "gst",
    "irdai",
)

INDIAN_CONTEXT_TERMS = (
    "india",
    "indian",
    "inr",
    "₹",
    "gst",
    "irdai",
    "policy wording",
    "premium quote",
    "waiting period",
    "pre-existing",
    "cashless",
    "network hospital",
    "upi",
    "third-party",
    "third party",
    "own damage",
    "family floater",
)


def evaluate_response(case: dict[str, Any], response_text: str) -> dict[str, Any]:
    parsed = _parse_json(response_text)
    lower = response_text.lower()
    expected = set(case["expected_categories"])
    avoid = set(case.get("avoid_categories", []))
    must_slots = set(case.get("must_ask_slots", QUESTION_SLOTS))
    must_sources = {str(source_id).lower() for source_id in case.get("must_cite_source_ids", [])}

    asked_slots = set(_extract_asked_slots(parsed, lower))
    recommended_categories = set(_extract_categories(parsed, lower))
    cited_sources = {source_id.lower() for source_id in _extract_source_ids(parsed, lower)}

    return {
        "case_id": case["case_id"],
        "ask_slot_recall": len(asked_slots & must_slots) / len(must_slots) if must_slots else 1.0,
        "expected_category_recall": len(recommended_categories & expected) / len(expected) if expected else 1.0,
        "unsafe_category_count": len(recommended_categories & avoid),
        "has_compliance_caveat": any(term in lower for term in CAVEAT_TERMS),
        "has_indian_context": any(term in lower for term in INDIAN_CONTEXT_TERMS),
        "has_fake_exact_premium": _has_fake_exact_premium(lower),
        "source_recall": len(cited_sources & must_sources) / len(must_sources) if must_sources else 1.0,
        "refused_premature_recommendation": _detect_refusal(parsed, lower),
        "is_json": parsed is not None,
        "recommended_categories": sorted(recommended_categories),
        "asked_slots": sorted(asked_slots),
        "cited_sources": sorted(cited_sources),
    }


def evaluate_predictions(eval_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in _read_jsonl(eval_path)}
    predictions = _read_jsonl(predictions_path)
    results = []
    for prediction in predictions:
        case = cases[prediction["case_id"]]
        response_text = prediction.get("response", prediction.get("raw_response", ""))
        results.append(evaluate_response(case, response_text))

    total = len(results)
    return {
        "total": total,
        "avg_ask_slot_recall": sum(item["ask_slot_recall"] for item in results) / total if total else 0.0,
        "avg_expected_category_recall": sum(item["expected_category_recall"] for item in results) / total if total else 0.0,
        "unsafe_category_rate": sum(item["unsafe_category_count"] > 0 for item in results) / total if total else 0.0,
        "json_rate": sum(item["is_json"] for item in results) / total if total else 0.0,
        "caveat_rate": sum(item["has_compliance_caveat"] for item in results) / total if total else 0.0,
        "indian_context_rate": sum(item["has_indian_context"] for item in results) / total if total else 0.0,
        "fake_premium_rate": sum(item["has_fake_exact_premium"] for item in results) / total if total else 0.0,
        "avg_source_recall": sum(item["source_recall"] for item in results) / total if total else 0.0,
        "premature_refusal_rate": _premature_refusal_rate(results, cases),
        "results": results,
    }


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_asked_slots(parsed: Any, lower: str) -> list[str]:
    slots = []
    slot_terms = {
        "age": ("age", "वय", "umar", "வயது", "వయస్సు"),
        "city": ("city", "शहर", "nagar", "நகரம்", "నగరం"),
        "family_members": ("family", "परिवार", "குடும்ப", "family size"),
        "dependents": ("dependent", "आश्रित", "dependents"),
        "existing_cover": ("existing", "current cover", "पहले", "cover"),
        "budget_band": ("budget", "premium range", "बजट"),
        "primary_need": ("need", "health", "motor", "travel", "cyber", "home", "accident"),
    }
    if isinstance(parsed, dict) and "ask_slots" in parsed:
        slots.extend(str(slot) for slot in parsed["ask_slots"])
    for slot, terms in slot_terms.items():
        if any(term.lower() in lower for term in terms):
            slots.append(slot)
    return sorted(set(slots))


def _extract_categories(parsed: Any, lower: str) -> list[str]:
    categories = {"health", "motor", "travel", "cyber", "home", "personal_accident", "life", "term"}
    found = set()
    for value in _walk_json(parsed):
        if isinstance(value, dict) and "category" in value:
            found.add(str(value["category"]))
        if isinstance(value, str):
            normalized = value.lower()
            for category in categories:
                if normalized == category or normalized == category.replace("_", " "):
                    found.add(category)
    for category in categories:
        if category.replace("_", " ") in lower or category in lower:
            found.add(category)
    return sorted(found)


def _extract_source_ids(parsed: Any, lower: str) -> list[str]:
    found = set()
    for value in _walk_json(parsed):
        if isinstance(value, dict):
            for key in ("source_ids", "must_cite_source_ids"):
                values = value.get(key, [])
                if isinstance(values, list):
                    found.update(str(item) for item in values)
        elif isinstance(value, str) and value.lower().startswith("pdf_"):
            found.add(value)
    for match in re.findall(r"pdf_[A-Za-z0-9_]+", lower):
        found.add(match)
    return sorted(found)


def _walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)
    else:
        yield value


def _detect_refusal(parsed: Any, lower: str) -> bool:
    refusal_terms = ("cannot_recommend_yet", "cannot recommend", "need suitability", "answer questions", "more information")
    if isinstance(parsed, dict) and str(parsed.get("decision", "")).lower() in {"cannot_recommend_yet", "refuse"}:
        return True
    return any(term in lower for term in refusal_terms)


def _has_fake_exact_premium(lower: str) -> bool:
    import re

    premium_claim = re.search(r"(premium|₹|rs\.?|inr)\s*[:=]?\s*[₹]?\s*\d{3,}", lower)
    quote_context = "premium quote" in lower or "official quote" in lower or "indicative" in lower
    return bool(premium_claim and not quote_context)


def _premature_refusal_rate(results: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> float:
    premature = [item for item in results if cases[item["case_id"]].get("task") == "reject_premature_recommendation"]
    if not premature:
        return 1.0
    return sum(item["refused_premature_recommendation"] for item in premature) / len(premature)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
