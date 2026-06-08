"""Build a benchmark report from tool-call predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.benchmark import build_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="data/insurance/benchmark_report.json")
    args = parser.parse_args()

    report = build_benchmark_report(args.eval, args.predictions, args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
