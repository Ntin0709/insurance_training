"""Build SFT and eval JSONL files for the insurance advisor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.dataset import export_eval, export_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-output", default="data/insurance/insurance_sft.jsonl")
    parser.add_argument("--eval-output", default="data/insurance/insurance_eval.jsonl")
    args = parser.parse_args()

    sft = export_sft(args.sft_output)
    eval_rows = export_eval(args.eval_output)
    print(f"wrote_sft={len(sft)} output={args.sft_output}")
    print(f"wrote_eval={len(eval_rows)} output={args.eval_output}")


if __name__ == "__main__":
    main()
