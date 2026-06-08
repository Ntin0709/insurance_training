"""Benchmark-grade reporting for insurance advisor tool-calling models."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from insurance_agent.tool_eval import evaluate_tool_predictions


def create_benchmark_split(
    eval_path: str | Path,
    output_path: str | Path,
    sample_size: int = 120,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(eval_path)
    rng = random.Random(seed)
    selected = _stratified_sample(rows, sample_size, rng)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return selected


def build_benchmark_report(
    eval_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    rows = _read_jsonl(eval_path)
    predictions = {row["case_id"]: row for row in _read_jsonl(predictions_path)}
    metrics = evaluate_tool_predictions(eval_path, predictions_path)
    row_by_id = {row["case_id"]: row for row in rows}
    result_by_id = {row["case_id"]: row for row in metrics["results"]}

    report = {
        "summary": _summary(metrics),
        "coverage": _coverage(rows),
        "slices": {
            "by_language": _slice_metrics(rows, result_by_id, lambda row: row["language"]),
            "by_task": _slice_metrics(rows, result_by_id, lambda row: row["task"]),
            "by_primary_category": _slice_metrics(rows, result_by_id, _primary_category),
            "by_city_tier": _slice_metrics(rows, result_by_id, lambda row: row.get("profile", {}).get("city_tier", "unknown")),
            "by_occupation": _slice_metrics(rows, result_by_id, lambda row: row.get("profile", {}).get("occupation", "unknown")),
        },
        "failures": _failure_examples(rows, predictions, result_by_id),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def _stratified_sample(rows: list[dict[str, Any]], sample_size: int, rng: random.Random) -> list[dict[str, Any]]:
    if sample_size >= len(rows):
        return rows
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["language"], _primary_category(row), row["task"])].append(row)

    selected = []
    bucket_items = list(buckets.values())
    rng.shuffle(bucket_items)
    for bucket in bucket_items:
        rng.shuffle(bucket)
        selected.append(bucket[0])
        if len(selected) >= sample_size:
            return _ensure_task_diversity(selected, rows, sample_size, rng)

    remaining = [row for bucket in bucket_items for row in bucket[1:]]
    rng.shuffle(remaining)
    selected.extend(remaining[: sample_size - len(selected)])
    return _ensure_task_diversity(selected, rows, sample_size, rng)


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total",
        "exact_tool_sequence_rate",
        "tool_order_rate",
        "required_slots_rate",
        "source_ids_rate",
        "safe_refusal_rate",
    )
    return {key: metrics[key] for key in keys}


def _ensure_task_diversity(
    selected: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected_by_id = {row["case_id"]: row for row in selected}
    tasks_present = {row["task"] for row in selected}
    all_tasks = sorted({row["task"] for row in all_rows})
    missing_tasks = [task for task in all_tasks if task not in tasks_present]
    if not missing_tasks:
        return selected[:sample_size]

    selected_list = list(selected)
    replaceable_indexes = [
        idx for idx, row in enumerate(selected_list)
        if sum(item["task"] == row["task"] for item in selected_list) > 1
    ]
    rng.shuffle(replaceable_indexes)
    for task in missing_tasks:
        candidates = [row for row in all_rows if row["task"] == task and row["case_id"] not in selected_by_id]
        if not candidates:
            continue
        candidate = rng.choice(candidates)
        if len(selected_list) < sample_size:
            selected_list.append(candidate)
        elif replaceable_indexes:
            idx = replaceable_indexes.pop()
            selected_list[idx] = candidate
        selected_by_id[candidate["case_id"]] = candidate
    return selected_list[:sample_size]


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "languages": dict(Counter(row["language"] for row in rows)),
        "tasks": dict(Counter(row["task"] for row in rows)),
        "primary_categories": dict(Counter(_primary_category(row) for row in rows)),
        "city_tiers": dict(Counter(row.get("profile", {}).get("city_tier", "unknown") for row in rows)),
        "occupations": dict(Counter(row.get("profile", {}).get("occupation", "unknown") for row in rows)),
    }


def _slice_metrics(rows: list[dict[str, Any]], result_by_id: dict[str, dict[str, Any]], key_fn) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(result_by_id[row["case_id"]])
    return {key: _aggregate_results(items) for key, items in sorted(grouped.items())}


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if not total:
        return {}
    return {
        "total": total,
        "tool_order_rate": sum(item["tool_order_correct"] for item in results) / total,
        "required_slots_rate": sum(item["required_slots_present"] for item in results) / total,
        "source_ids_rate": sum(item["source_ids_present"] for item in results) / total,
        "exact_tool_sequence_rate": sum(item["exact_tool_sequence"] for item in results) / total,
    }


def _failure_examples(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    limit: int = 25,
) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        result = result_by_id[row["case_id"]]
        if result["tool_order_correct"] and result["required_slots_present"] and result["source_ids_present"]:
            continue
        prediction = predictions.get(row["case_id"], {})
        failures.append(
            {
                "case_id": row["case_id"],
                "language": row["language"],
                "task": row["task"],
                "primary_category": _primary_category(row),
                "expected_tools": result["expected_tools"],
                "predicted_tools": result["predicted_tools"],
                "first_error": result["first_error"],
                "raw_response_preview": str(prediction.get("raw_response", ""))[:1200],
            }
        )
        if len(failures) >= limit:
            break
    return failures


def _primary_category(row: dict[str, Any]) -> str:
    categories = row.get("expected_final", {}).get("expected_categories", [])
    if categories:
        return categories[0]
    return "none"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
