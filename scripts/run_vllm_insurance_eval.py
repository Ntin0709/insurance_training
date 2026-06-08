"""Run insurance tool-calling eval against a local vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.vllm_testbed import VLLMConfig, run_vllm_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/insurance/insurance_india_toolcall_eval.jsonl")
    parser.add_argument("--predictions", default="data/insurance/vllm_toolcall_predictions.jsonl")
    parser.add_argument("--metrics", default="data/insurance/vllm_toolcall_metrics.json")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "local-model"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-mode", choices=["head", "evenly"], default="head")
    parser.add_argument("--dry-run-oracle", action="store_true")
    args = parser.parse_args()

    config = VLLMConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    metrics = run_vllm_eval(
        eval_path=args.eval,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        config=config,
        limit=args.limit,
        sample_mode=args.sample_mode,
        dry_run_oracle=args.dry_run_oracle,
    )
    summary_keys = [
        "total",
        "exact_tool_sequence_rate",
        "tool_order_rate",
        "required_slots_rate",
        "source_ids_rate",
        "safe_refusal_rate",
    ]
    print(json.dumps({key: metrics[key] for key in summary_keys}, indent=2, sort_keys=True))
    print(f"predictions={args.predictions}")
    print(f"metrics={args.metrics}")


if __name__ == "__main__":
    main()
