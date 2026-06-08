"""Rebuild edge benchmark artifacts from an existing edge_scenarios.jsonl file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.edge_benchmark import benchmark_quality_report
from insurance_agent.rag_index import InsuranceRAGIndex
from insurance_agent.synthetic_data import ScenarioCard, build_benchmark_cases, generate_fast_oracle_traces


REFUSAL_BUCKETS = {
    "out_of_scope_life_or_investment",
    "premature_recommendation_pressure",
    "prompt_injection_ignore_safety",
}
CATEGORIES = ["health", "motor", "travel", "cyber", "home", "personal_accident"]
SLOTS = ["age", "city", "family_members", "dependents", "existing_cover", "budget_band", "primary_need"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    scenarios = [_scenario_from_row(row, idx) for idx, row in enumerate(_read_jsonl(args.scenarios))]
    index = InsuranceRAGIndex.from_chunks_file(args.chunks)
    cases = build_benchmark_cases(scenarios, index)
    predictions = generate_fast_oracle_traces(cases)
    report = benchmark_quality_report(scenarios, cases, predictions)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "scenarios": output / "edge_scenarios.jsonl",
        "benchmark": output / "edge_benchmark.jsonl",
        "oracle_predictions": output / "edge_oracle_predictions.jsonl",
        "report": output / "edge_report.json",
    }
    _write_jsonl(paths["scenarios"], [scenario.to_dict() for scenario in scenarios])
    _write_jsonl(paths["benchmark"], cases)
    _write_jsonl(paths["oracle_predictions"], predictions)
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    for name, path in paths.items():
        print(f"{name}={path}")


def _scenario_from_row(row: dict, idx: int) -> ScenarioCard:
    scenario_type = row["scenario_type"]
    expected = list(row.get("expected_categories", []))
    avoid = list(row.get("avoid_categories", []))
    missing = list(row.get("missing_slots", []))
    task = row.get("task", "grounded_personalized_recommendation")
    if scenario_type in REFUSAL_BUCKETS:
        expected = []
        avoid = CATEGORIES
        missing = SLOTS
        task = "reject_premature_recommendation"
    elif len(expected) > 1:
        task = "compare_suitable_categories"
    else:
        task = "grounded_personalized_recommendation"
    return ScenarioCard(
        scenario_id=f"edge_{idx:06d}",
        scenario_type=scenario_type,
        language=row["language"],
        user_message=row["user_message"],
        profile=row["profile"],
        expected_categories=expected,
        avoid_categories=avoid,
        missing_slots=missing,
        risk_tags=row.get("risk_tags", []),
        task=task,
    )


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
