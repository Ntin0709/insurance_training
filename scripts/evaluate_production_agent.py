"""Evaluate controlled production agent outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.production_eval import evaluate_production_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="data/insurance/production_agent_report.json")
    args = parser.parse_args()

    metrics = evaluate_production_agent(args.eval, args.predictions)
    Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    summary = {k: metrics[k] for k in metrics if k != "results"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
