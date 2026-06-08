"""Rebuild DPO files with same-shape hard negatives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.synthetic_data import export_dpo_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/insurance/sdg_rules_50k_balanced/synthetic_benchmark.jsonl")
    parser.add_argument("--predictions", default="data/insurance/sdg_rules_50k_balanced/synthetic_oracle_predictions.jsonl")
    parser.add_argument("--output-dir", default="data/insurance/sdg_rules_50k_balanced")
    parser.add_argument("--train", default="data/insurance/sdg_rules_50k_balanced/benchmark_train.jsonl")
    parser.add_argument("--val", default="data/insurance/sdg_rules_50k_balanced/benchmark_val.jsonl")
    parser.add_argument("--test", default="data/insurance/sdg_rules_50k_balanced/benchmark_test.jsonl")
    args = parser.parse_args()

    cases = _read_jsonl(args.benchmark)
    predictions = _read_jsonl(args.predictions)
    rows = export_dpo_rows(cases, predictions)
    output = Path(args.output_dir)
    _write_jsonl(output / "synthetic_dpo.jsonl", rows)
    by_case = {row["id"].split(":")[0]: row for row in rows}
    for split_name, split_path in (("train", args.train), ("val", args.val), ("test", args.test)):
        split_cases = _read_jsonl(split_path)
        split_rows = [by_case[case["case_id"]] for case in split_cases]
        _write_jsonl(output / f"dpo_{split_name}.jsonl", split_rows)
    report = _dpo_shape_report(rows)
    (output / "dpo_hard_negative_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def _dpo_shape_report(rows: list[dict]) -> dict:
    negative_types = {}
    chosen_lengths = []
    rejected_lengths = []
    for row in rows:
        rejected = json.loads(row["rejected"])
        negative_type = rejected.get("negative_type", "unknown")
        negative_types[negative_type] = negative_types.get(negative_type, 0) + 1
        chosen_lengths.append(len(row["chosen"]))
        rejected_lengths.append(len(row["rejected"]))
    ratios = [rejected / max(chosen, 1) for chosen, rejected in zip(chosen_lengths, rejected_lengths)]
    ratios_sorted = sorted(ratios)
    return {
        "rows": len(rows),
        "negative_types": negative_types,
        "median_chosen_chars": sorted(chosen_lengths)[len(chosen_lengths) // 2] if chosen_lengths else 0,
        "median_rejected_chars": sorted(rejected_lengths)[len(rejected_lengths) // 2] if rejected_lengths else 0,
        "median_rejected_to_chosen_ratio": ratios_sorted[len(ratios_sorted) // 2] if ratios_sorted else 0,
    }


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
