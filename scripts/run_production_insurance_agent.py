"""Run the controlled production insurance agent benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.production_agent import run_production_agent_benchmark
from insurance_agent.tool_eval import evaluate_tool_predictions
from insurance_agent.vllm_testbed import VLLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/insurance/insurance_benchmark_40.jsonl")
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--predictions", default="data/insurance/production_agent_predictions.jsonl")
    parser.add_argument("--metrics", default="data/insurance/production_agent_metrics.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use-llm-final", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "local-model"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    config = (
        VLLMConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=1,
        )
        if args.use_llm_final
        else None
    )
    run_production_agent_benchmark(
        eval_path=args.eval,
        chunks_path=args.chunks,
        predictions_path=args.predictions,
        limit=args.limit,
        config=config,
        use_llm_final=args.use_llm_final,
    )
    metrics = evaluate_tool_predictions(args.eval, args.predictions)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["total", "tool_order_rate", "required_slots_rate", "source_ids_rate", "safe_refusal_rate"]}, indent=2))
    print(f"predictions={args.predictions}")
    print(f"metrics={args.metrics}")


if __name__ == "__main__":
    main()
