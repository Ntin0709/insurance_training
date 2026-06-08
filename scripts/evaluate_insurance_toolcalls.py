"""Evaluate multi-turn insurance advisor tool-call predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.tool_eval import evaluate_tool_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/insurance/insurance_india_toolcall_eval.jsonl")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    metrics = evaluate_tool_predictions(args.eval, args.predictions)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
