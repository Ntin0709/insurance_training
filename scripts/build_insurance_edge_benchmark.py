"""Build a hard, LLM-generated benchmark for insurance-agent evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.edge_benchmark import build_edge_benchmark_pipeline
from insurance_agent.vllm_testbed import VLLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--output-dir", default="data/insurance/edge_benchmark")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--fallback-rules", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "local-model"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    config = None
    if not args.fallback_rules:
        config = VLLMConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=1,
        )

    result = build_edge_benchmark_pipeline(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        llm_config=config,
        llm_batch_size=args.llm_batch_size,
        fallback_rules=args.fallback_rules,
    )
    print(json.dumps(result["report"], indent=2, ensure_ascii=False, sort_keys=True))
    for name, path in result["paths"].items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
