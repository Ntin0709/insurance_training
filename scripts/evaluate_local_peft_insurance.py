"""Evaluate a local base model plus optional PEFT adapter on insurance benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.benchmark_report import build_unified_report
from insurance_agent.training import model_torch_dtype
from insurance_agent.vllm_testbed import _messages_for_case, parse_tool_calls


def optional_stack():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install eval dependencies first:\n"
            'python3 -m pip install "transformers>=4.51" "peft>=0.11" "accelerate>=0.30"'
        ) from exc
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model path/name.")
    parser.add_argument("--adapter", default="", help="Optional PEFT adapter path.")
    parser.add_argument("--eval", default="data/insurance/edge_benchmark_llm_300/edge_benchmark.jsonl")
    parser.add_argument("--predictions", default="outputs/local_peft_eval/predictions.jsonl")
    parser.add_argument("--report", default="outputs/local_peft_eval/report.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = optional_stack()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_torch_dtype(torch, args.precision),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    cases = _read_jsonl(args.eval)
    if args.limit:
        cases = cases[: args.limit]
    rows = []
    for idx, case in enumerate(cases, start=1):
        messages = _messages_for_case(case)
        prompt = _messages_to_text(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        final_response = _extract_final_response(raw)
        rows.append(
            {
                "case_id": case["case_id"],
                "predicted_tool_calls": parse_tool_calls(raw),
                "observations": [],
                "final_response": final_response,
                "agent_status": "completed",
                "raw_response": raw,
            }
        )
        print(f"completed={idx}/{len(cases)} case_id={case['case_id']}", flush=True)

    _write_jsonl(args.predictions, rows)
    report = build_unified_report(args.eval, args.predictions, args.report)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False, sort_keys=True))


def _messages_to_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return "\n".join(f"<|{message['role']}|>\n{message['content']}" for message in messages) + "\n<|assistant|>\n"


def _extract_final_response(raw: str) -> Any:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed.get("final_response", parsed)
    return {}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
