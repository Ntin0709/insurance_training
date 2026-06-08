"""Build evidence-backed insurance advisor eval data from scraped PDF chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.eval_data import build_eval_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--output", default="data/insurance/insurance_india_grounded_eval.jsonl")
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    cases = build_eval_data(args.chunks, args.output, args.per_category, args.seed)
    print(f"wrote_eval_cases={len(cases)} output={args.output}")


if __name__ == "__main__":
    main()
