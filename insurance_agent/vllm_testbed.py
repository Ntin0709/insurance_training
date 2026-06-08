"""vLLM/OpenAI-compatible testbed for insurance advisor tool-calling eval."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from insurance_agent.tool_eval import evaluate_tool_predictions
from insurance_agent.tool_eval import _expected_tool_calls, _grounded_case_from_benchmark


DEFAULT_BASE_URL = "http://localhost:8000/v1"


@dataclass(frozen=True)
class VLLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = "local-model"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 120
    retries: int = 2


def run_vllm_eval(
    eval_path: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
    config: VLLMConfig,
    limit: int = 0,
    sample_mode: str = "head",
    dry_run_oracle: bool = False,
) -> dict[str, Any]:
    cases = _read_jsonl(eval_path)
    if limit:
        cases = _sample_cases(cases, limit, sample_mode)

    predictions = []
    for index, case in enumerate(cases, start=1):
        if dry_run_oracle:
            tool_calls = case.get("expected_tool_calls") or _expected_tool_calls(_grounded_case_from_benchmark(case))
            raw_response = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
        else:
            raw_response = call_vllm(case, config)
            tool_calls = parse_tool_calls(raw_response)
        predictions.append(
            {
                "case_id": case["case_id"],
                "predicted_tool_calls": tool_calls,
                "raw_response": raw_response,
            }
        )
        print(f"completed={index}/{len(cases)} case_id={case['case_id']} calls={len(tool_calls)}", flush=True)

    _write_jsonl(predictions_path, predictions)
    metrics = evaluate_tool_predictions(eval_path, predictions_path)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return metrics


def call_vllm(case: dict[str, Any], config: VLLMConfig) -> str:
    payload = {
        "model": config.model,
        "messages": _messages_for_case(case),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"].get("content") or ""
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < config.retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"vLLM request failed after {config.retries + 1} attempts: {exc}") from exc
    raise RuntimeError(f"vLLM request failed: {last_error}")


def call_vllm_messages(messages: list[dict[str, str]], config: VLLMConfig) -> str:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"}
    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"].get("content") or ""
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < config.retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"vLLM request failed after {config.retries + 1} attempts: {exc}") from exc
    raise RuntimeError(f"vLLM request failed: {last_error}")


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse model output into `expected_tool_calls`-compatible records."""
    parsed = _extract_json(text)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        calls = parsed
    elif isinstance(parsed, dict):
        calls = parsed.get("tool_calls") or parsed.get("expected_tool_calls") or parsed.get("actions") or []
    else:
        return []

    normalized = []
    for idx, call in enumerate(calls, start=1):
        if not isinstance(call, dict):
            continue
        tool_name = call.get("tool_name") or call.get("tool") or call.get("name")
        arguments = call.get("arguments") or call.get("args") or {}
        if not tool_name:
            continue
        normalized.append(
            {
                "turn": int(call.get("turn", idx)),
                "role": call.get("role", "assistant"),
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )
    return normalized


def _messages_for_case(case: dict[str, Any]) -> list[dict[str, str]]:
    tools_compact = [
        {
            "name": tool["name"],
            "required": tool.get("parameters", {}).get("required", []),
            "description": tool.get("description", ""),
        }
        for tool in case["tools"]
    ]
    expected_shape = {
        "tool_calls": [
            {
                "turn": 1,
                "role": "assistant",
                "tool_name": "ask_suitability_questions",
                "arguments": {"slots": ["age"], "language": "English", "questions": ["..."]},
            }
        ]
    }
    user_context = {
        "case_id": case["case_id"],
        "task": case["task"],
        "language": case["language"],
        "user_message": case["messages"][-1]["content"],
        "profile": case["profile"],
        "evidence": case["evidence"],
        "expected_final_requirements": case["expected_final"],
        "available_tools": tools_compact,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are being evaluated on insurance advisor tool calling for Indian banking customers. "
                "Return only valid JSON. Do not include markdown. "
                "Use the available tools in the safe order. Ask suitability questions before recommending. "
                "Use official evidence source IDs. Never invent premiums."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create the complete multi-turn tool-call plan for this case.\n"
                f"Return JSON exactly in this shape: {json.dumps(expected_shape, ensure_ascii=False)}\n"
                f"Case:\n{json.dumps(user_context, ensure_ascii=False, sort_keys=True)}"
            ),
        },
    ]


def _extract_json(text: str | None) -> Any:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)
    for end in range(len(text), start, -1):
        snippet = text[start:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return None


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_cases(cases: list[dict[str, Any]], limit: int, sample_mode: str) -> list[dict[str, Any]]:
    if limit >= len(cases):
        return cases
    if sample_mode == "head":
        return cases[:limit]
    if sample_mode == "evenly":
        if limit <= 1:
            return cases[:limit]
        step = (len(cases) - 1) / (limit - 1)
        indexes = [round(i * step) for i in range(limit)]
        return [cases[index] for index in indexes]
    raise ValueError(f"Unknown sample_mode: {sample_mode}")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
