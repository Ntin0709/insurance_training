"""Unified benchmark report for insurance-agent model tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from insurance_agent.final_answer_eval import evaluate_final_answers
from insurance_agent.production_eval import evaluate_production_agent
from insurance_agent.tool_eval import evaluate_tool_predictions


def build_unified_report(
    eval_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    cases = _read_jsonl(eval_path)
    tool_metrics = evaluate_tool_predictions(eval_path, predictions_path)
    production_metrics = evaluate_production_agent(eval_path, predictions_path)
    final_metrics = evaluate_final_answers(eval_path, predictions_path)
    report = {
        "summary": {
            "total": len(cases),
            "tool_order_rate": tool_metrics["tool_order_rate"],
            "exact_tool_sequence_rate": tool_metrics["exact_tool_sequence_rate"],
            "required_slots_rate": tool_metrics["required_slots_rate"],
            "source_ids_rate": tool_metrics["source_ids_rate"],
            "safe_refusal_rate": tool_metrics["safe_refusal_rate"],
            "workflow_order_rate": production_metrics["workflow_order_rate"],
            "citation_from_retrieval_rate": production_metrics["citation_from_retrieval_rate"],
            "final_category_recall": final_metrics["category_recall"],
            "final_caveat_rate": final_metrics["caveat_rate"],
            "fake_premium_rate": final_metrics["fake_premium_rate"],
            "safe_refusal_final_rate": final_metrics["safe_refusal_final_rate"],
        },
        "coverage": _coverage(cases),
        "tool_metrics": _drop_results(tool_metrics),
        "production_metrics": _drop_results(production_metrics),
        "final_answer_metrics": _drop_results(final_metrics),
        "slices": _slice_report(cases, tool_metrics["results"], production_metrics["results"], final_metrics["results"]),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def _coverage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "languages": dict(Counter(row["language"] for row in cases)),
        "tasks": dict(Counter(row["task"] for row in cases)),
        "scenario_types": dict(Counter(row.get("scenario_type", "unknown") for row in cases)),
        "risk_tags": dict(Counter(tag for row in cases for tag in row.get("risk_tags", []))),
        "categories": dict(Counter(category for row in cases for category in row["expected_final"]["expected_categories"])),
        "city_tiers": dict(Counter(row.get("profile", {}).get("city_tier", "unknown") for row in cases)),
    }


def _slice_report(
    cases: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    production_results: list[dict[str, Any]],
    final_results: list[dict[str, Any]],
) -> dict[str, Any]:
    tool_by_id = {row["case_id"]: row for row in tool_results}
    production_by_id = {row["case_id"]: row for row in production_results}
    final_by_id = {row["case_id"]: row for row in final_results}
    return {
        "by_language": _group(cases, lambda row: row["language"], tool_by_id, production_by_id, final_by_id),
        "by_task": _group(cases, lambda row: row["task"], tool_by_id, production_by_id, final_by_id),
        "by_risk_tag": _group_risk_tags(cases, tool_by_id, production_by_id, final_by_id),
        "by_category": _group_category(cases, tool_by_id, production_by_id, final_by_id),
    }


def _group(cases, key_fn, tool_by_id, production_by_id, final_by_id) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(key_fn(case)), []).append(case)
    return {
        key: _metrics_for(rows, tool_by_id, production_by_id, final_by_id)
        for key, rows in sorted(grouped.items())
    }


def _group_risk_tags(cases, tool_by_id, production_by_id, final_by_id) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        for tag in case.get("risk_tags", ["none"]):
            grouped.setdefault(tag, []).append(case)
    return {
        key: _metrics_for(rows, tool_by_id, production_by_id, final_by_id)
        for key, rows in sorted(grouped.items())
    }


def _group_category(cases, tool_by_id, production_by_id, final_by_id) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        categories = case["expected_final"]["expected_categories"] or ["refusal"]
        for category in categories:
            grouped.setdefault(category, []).append(case)
    return {
        key: _metrics_for(rows, tool_by_id, production_by_id, final_by_id)
        for key, rows in sorted(grouped.items())
    }


def _metrics_for(rows, tool_by_id, production_by_id, final_by_id) -> dict[str, Any]:
    ids = [row["case_id"] for row in rows]
    return {
        "count": len(ids),
        "tool_order_rate": _avg(tool_by_id[i]["tool_order_correct"] for i in ids),
        "required_slots_rate": _avg(tool_by_id[i]["required_slots_present"] for i in ids),
        "source_ids_rate": _avg(tool_by_id[i]["source_ids_present"] for i in ids),
        "workflow_order_rate": _avg(production_by_id[i]["workflow_order_correct"] for i in ids),
        "citation_from_retrieval_rate": _avg(production_by_id[i]["citation_from_retrieval"] for i in ids),
        "final_category_rate": _avg(final_by_id[i]["category_match"] for i in ids),
        "final_caveat_rate": _avg(final_by_id[i]["has_caveats"] for i in ids),
    }


def _avg(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(bool(value) for value in values) / len(values)


def _drop_results(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "results"}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
