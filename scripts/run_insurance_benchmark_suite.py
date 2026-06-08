"""Run the insurance benchmark suite against model or oracle predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.benchmark_report import build_unified_report
from insurance_agent.llm_judge import judge_final_answers
from insurance_agent.production_agent import run_production_agent_benchmark
from insurance_agent.tool_eval import evaluate_tool_predictions
from insurance_agent.vllm_testbed import VLLMConfig, run_vllm_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["tool-plan-model", "production-oracle", "both"], default="both")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "local-model"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-mode", choices=["head", "evenly"], default="head")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--judge-limit", type=int, default=25)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = VLLMConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=1,
    )
    manifest: dict[str, str | dict] = {
        "eval": args.eval,
        "mode": args.mode,
        "model": args.model,
        "base_url": args.base_url,
    }

    if args.mode in {"tool-plan-model", "both"}:
        tool_predictions = output / "model_tool_predictions.jsonl"
        tool_metrics = output / "model_tool_metrics.json"
        metrics = run_vllm_eval(
            eval_path=args.eval,
            predictions_path=tool_predictions,
            metrics_path=tool_metrics,
            config=config,
            limit=args.limit,
            sample_mode=args.sample_mode,
        )
        manifest["model_tool_predictions"] = str(tool_predictions)
        manifest["model_tool_metrics"] = str(tool_metrics)
        manifest["model_tool_summary"] = _summary(metrics)

    if args.mode in {"production-oracle", "both"}:
        production_predictions = output / "production_predictions.jsonl"
        run_production_agent_benchmark(
            eval_path=args.eval,
            chunks_path=args.chunks,
            predictions_path=production_predictions,
            limit=args.limit,
        )
        unified_report = output / "production_unified_report.json"
        report = build_unified_report(args.eval, production_predictions, unified_report)
        manifest["production_predictions"] = str(production_predictions)
        manifest["production_unified_report"] = str(unified_report)
        manifest["production_summary"] = report["summary"]
        if args.llm_judge:
            judge_report = output / "production_llm_judge_report.json"
            judge = judge_final_answers(args.eval, production_predictions, judge_report, config, limit=args.judge_limit)
            manifest["production_llm_judge_report"] = str(judge_report)
            manifest["production_llm_judge_summary"] = {key: value for key, value in judge.items() if key != "results"}

    manifest_path = output / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"manifest={manifest_path}")


def _summary(metrics: dict) -> dict:
    keys = (
        "total",
        "exact_tool_sequence_rate",
        "tool_order_rate",
        "required_slots_rate",
        "source_ids_rate",
        "safe_refusal_rate",
    )
    return {key: metrics[key] for key in keys}


if __name__ == "__main__":
    main()
