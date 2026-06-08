"""Generate large-scale insurance SDG data with deterministic quality gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.sdg_scale import generate_scaled_sdg_pipeline
from insurance_agent.vllm_testbed import VLLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--scenario-source", choices=["llm", "hybrid", "rules"], default="hybrid")
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-warnings", action="store_true")
    parser.add_argument("--heldout", action="append", default=[])
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "local-model"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    config = None
    if args.scenario_source in {"llm", "hybrid"}:
        config = VLLMConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=1,
        )

    result = generate_scaled_sdg_pipeline(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        scenario_source=args.scenario_source,
        llm_config=config,
        llm_batch_size=args.llm_batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        heldout_paths=args.heldout,
        allow_warnings=args.allow_warnings,
    )
    print(json.dumps(result["report"], indent=2, ensure_ascii=False, sort_keys=True))
    for name, path in result["paths"].items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
