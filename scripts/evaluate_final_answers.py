"""Evaluate final customer-facing responses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.final_answer_eval import evaluate_final_answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="data/insurance/final_answer_report.json")
    args = parser.parse_args()

    report = evaluate_final_answers(args.eval, args.predictions)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
