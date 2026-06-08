"""Build multi-turn tool-calling eval data from grounded insurance eval cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.tool_eval import build_tool_eval_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-eval", default="data/insurance/insurance_india_grounded_eval.jsonl")
    parser.add_argument("--output", default="data/insurance/insurance_india_toolcall_eval.jsonl")
    args = parser.parse_args()

    rows = build_tool_eval_data(args.grounded_eval, args.output)
    print(f"wrote_toolcall_eval={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
