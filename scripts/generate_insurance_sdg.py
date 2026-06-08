"""Run the synthetic data generation pipeline for insurance-agent training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.synthetic_data import generate_sdg_pipeline
from insurance_agent.vllm_testbed import VLLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--output-dir", default="data/insurance/synthetic")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--scenario-source", choices=["rules", "llm", "hybrid"], default="rules")
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "local-model"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--production-oracle",
        action="store_true",
        help="Use the full production agent for oracle traces. Slower; useful for small validation runs.",
    )
    args = parser.parse_args()
    llm_config = None
    if args.scenario_source in {"llm", "hybrid"}:
        llm_config = VLLMConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=1,
        )

    result = generate_sdg_pipeline(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        fast_oracle=not args.production_oracle,
        scenario_source=args.scenario_source,
        llm_config=llm_config,
        llm_batch_size=args.llm_batch_size,
    )
    print(json.dumps(result["report"], indent=2, ensure_ascii=False, sort_keys=True))
    for name, path in result["paths"].items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
