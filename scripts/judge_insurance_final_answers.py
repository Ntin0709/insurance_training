"""Run optional LLM-as-judge evaluation on final insurance-agent answers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.llm_judge import judge_final_answers
from insurance_agent.vllm_testbed import VLLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="data/insurance/llm_judge_report.json")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "local-model"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = VLLMConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=1,
    )
    report = judge_final_answers(args.eval, args.predictions, args.output, config, limit=args.limit)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
