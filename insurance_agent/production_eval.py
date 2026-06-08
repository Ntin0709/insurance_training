"""Production-agent metrics that do not require hidden oracle source IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


NORMAL_ORDER = [
    "ask_suitability_questions",
    "retrieve_policy_evidence",
    "rank_insurance_options",
    "generate_recommendation",
    "finish",
]
REFUSAL_ORDER = ["refuse_premature_recommendation", "finish"]


def evaluate_production_agent(eval_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in _read_jsonl(eval_path)}
    predictions = _read_jsonl(predictions_path)
    results = []
    for prediction in predictions:
        case = cases[prediction["case_id"]]
        results.append(_score_case(case, prediction))
    total = len(results)
    return {
        "total": total,
        "workflow_order_rate": _avg(results, "workflow_order_correct"),
        "required_slots_rate": _avg(results, "required_slots_present"),
        "retrieval_executed_rate": _avg(results, "retrieval_executed"),
        "citation_present_rate": _avg(results, "citation_present"),
        "citation_from_retrieval_rate": _avg(results, "citation_from_retrieval"),
        "safe_refusal_rate": _safe_refusal_rate(results, cases),
        "completed_rate": _avg(results, "completed"),
        "results": results,
    }


def _score_case(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    calls = prediction.get("predicted_tool_calls", [])
    tools = [call.get("tool_name") for call in calls]
    expected_order = REFUSAL_ORDER if case["task"] == "reject_premature_recommendation" else NORMAL_ORDER
    required_slots = set(case["expected_final"]["must_ask_slots"])
    asked_slots = set()
    generated_source_ids = set()
    retrieved_source_ids = set()

    for call in calls:
        args = call.get("arguments", {})
        asked_slots.update(args.get("slots", []))
        asked_slots.update(args.get("missing_slots", []))
        generated_source_ids.update(args.get("source_ids", []))

    for observation in prediction.get("observations", []):
        payload = observation.get("payload", {})
        for result in payload.get("results", []):
            retrieved_source_ids.add(result.get("source_id"))
        recommendation = payload.get("recommendation", {})
        if isinstance(recommendation, dict):
            for item in recommendation.get("recommendation", []):
                generated_source_ids.update(item.get("source_ids", []))
            generated_source_ids.update(recommendation.get("source_ids", []))

    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "workflow_order_correct": tools == expected_order,
        "required_slots_present": required_slots.issubset(asked_slots),
        "retrieval_executed": "retrieve_policy_evidence" in tools if expected_order == NORMAL_ORDER else True,
        "citation_present": bool(generated_source_ids) if expected_order == NORMAL_ORDER else True,
        "citation_from_retrieval": bool(generated_source_ids & retrieved_source_ids) if expected_order == NORMAL_ORDER else True,
        "completed": prediction.get("agent_status") == "completed",
        "predicted_tools": tools,
        "generated_source_ids": sorted(source_id for source_id in generated_source_ids if source_id),
        "retrieved_source_ids": sorted(source_id for source_id in retrieved_source_ids if source_id)[:10],
    }


def _safe_refusal_rate(results: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> float:
    refusal = [item for item in results if cases[item["case_id"]]["task"] == "reject_premature_recommendation"]
    if not refusal:
        return 1.0
    return sum(item["predicted_tools"] == REFUSAL_ORDER for item in refusal) / len(refusal)


def _avg(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return sum(bool(item[key]) for item in results) / len(results)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
