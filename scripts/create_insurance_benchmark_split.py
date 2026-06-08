"""Create a stratified benchmark split from the full tool-call eval set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.benchmark import create_benchmark_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/insurance/insurance_india_toolcall_eval.jsonl")
    parser.add_argument("--output", default="data/insurance/insurance_benchmark_120.jsonl")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = create_benchmark_split(args.eval, args.output, args.sample_size, args.seed)
    print(json.dumps({"output": args.output, "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
