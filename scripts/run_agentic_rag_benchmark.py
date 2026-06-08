"""Run the production-shaped agentic RAG benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.agentic_rag import run_agentic_rag_benchmark
from insurance_agent.tool_eval import evaluate_tool_predictions
from insurance_agent.vllm_testbed import VLLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/insurance/insurance_benchmark_40.jsonl")
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--predictions", default="data/insurance/agentic_rag_predictions.jsonl")
    parser.add_argument("--metrics", default="data/insurance/agentic_rag_metrics.json")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "local-model"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    config = VLLMConfig(base_url=args.base_url, model=args.model, api_key=args.api_key, timeout=args.timeout, retries=1)
    run_agentic_rag_benchmark(args.eval, args.chunks, args.predictions, config, args.limit)
    metrics = evaluate_tool_predictions(args.eval, args.predictions)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["total", "tool_order_rate", "required_slots_rate", "source_ids_rate", "safe_refusal_rate"]}, indent=2))
    print(f"predictions={args.predictions}")
    print(f"metrics={args.metrics}")


if __name__ == "__main__":
    main()
