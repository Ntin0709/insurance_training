"""LLM-paraphrase rule scenarios while preserving verified labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.rag_index import InsuranceRAGIndex
from insurance_agent.sdg_quality import assert_quality, quality_report
from insurance_agent.sdg_scale import split_rows_by_message
from insurance_agent.synthetic_data import (
    ScenarioCard,
    _renumber_scenarios,
    build_benchmark_cases,
    export_dpo_rows,
    export_rlvr_rows,
    export_sft_rows,
    generate_fast_oracle_traces,
)
from insurance_agent.vllm_testbed import VLLMConfig, call_vllm_messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-scenarios", default="data/insurance/sdg_rules_50k_balanced/synthetic_scenarios.jsonl")
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--output-dir", default="data/insurance/sdg_paraphrased_20k")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--allow-warnings", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "synthetic_scenarios.jsonl"
    source = _sample_source_scenarios(args.input_scenarios, args.count, args.seed)
    existing = _read_scenarios(checkpoint) if checkpoint.exists() else []
    config = VLLMConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )

    scenarios = paraphrase_scenarios(
        source=source,
        existing=existing,
        config=config,
        batch_size=args.batch_size,
        checkpoint_path=checkpoint,
        checkpoint_every=args.checkpoint_every,
    )
    scenarios = _renumber_scenarios(scenarios[: args.count])
    _write_jsonl(checkpoint, [scenario.to_dict() for scenario in scenarios])

    index = InsuranceRAGIndex.from_chunks_file(args.chunks)
    cases = build_benchmark_cases(scenarios, index)
    predictions = generate_fast_oracle_traces(cases)
    sft_rows = export_sft_rows(cases, predictions)
    dpo_rows = export_dpo_rows(cases, predictions)
    rlvr_rows = export_rlvr_rows(cases, predictions)
    splits = split_rows_by_message(cases, seed=args.seed)
    report = quality_report(scenarios, cases, predictions, sft_rows, dpo_rows, rlvr_rows, splits=splits)
    report["scenario_source"] = "llm_paraphrased_rules"
    report["splits"] = {name: len(rows) for name, rows in splits.items()}
    assert_quality(report, allow_warnings=args.allow_warnings)

    paths = {
        "benchmark": output / "synthetic_benchmark.jsonl",
        "oracle_predictions": output / "synthetic_oracle_predictions.jsonl",
        "sft": output / "synthetic_sft.jsonl",
        "dpo": output / "synthetic_dpo.jsonl",
        "rlvr": output / "synthetic_rlvr.jsonl",
        "report": output / "synthetic_quality_report.json",
    }
    _write_jsonl(paths["benchmark"], cases)
    _write_jsonl(paths["oracle_predictions"], predictions)
    _write_jsonl(paths["sft"], sft_rows)
    _write_jsonl(paths["dpo"], dpo_rows)
    _write_jsonl(paths["rlvr"], rlvr_rows)
    _write_splits(output, "benchmark", cases, splits)
    _write_splits(output, "sft", sft_rows, splits)
    _write_splits(output, "dpo", dpo_rows, splits)
    _write_splits(output, "rlvr", rlvr_rows, splits)
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "report": report}, indent=2, ensure_ascii=False, sort_keys=True))


def paraphrase_scenarios(
    source: list[ScenarioCard],
    existing: list[ScenarioCard],
    config: VLLMConfig,
    batch_size: int,
    checkpoint_path: Path,
    checkpoint_every: int,
) -> list[ScenarioCard]:
    scenarios = list(existing)
    start = len(scenarios)
    for offset in range(start, len(source), batch_size):
        batch = source[offset : offset + batch_size]
        raw = call_vllm_messages(_paraphrase_prompt(batch), config)
        parsed = _extract_paraphrases(raw)
        by_id = {str(row.get("scenario_id")): str(row.get("user_message", "")).strip() for row in parsed}
        accepted = 0
        for scenario in batch:
            message = by_id.get(scenario.scenario_id, "")
            if not _valid_paraphrase(message, scenario):
                message = scenario.user_message
            scenarios.append(
                ScenarioCard(
                    scenario_id=scenario.scenario_id,
                    scenario_type=f"paraphrased_{scenario.scenario_type}",
                    language=scenario.language,
                    user_message=message,
                    profile=scenario.profile,
                    expected_categories=scenario.expected_categories,
                    avoid_categories=scenario.avoid_categories,
                    missing_slots=scenario.missing_slots,
                    risk_tags=sorted(set([*scenario.risk_tags, "llm_paraphrased"])),
                    task=scenario.task,
                )
            )
            accepted += 1
        if len(scenarios) % checkpoint_every == 0 or offset + batch_size >= len(source):
            _write_jsonl(checkpoint_path, [scenario.to_dict() for scenario in scenarios])
        print(f"paraphrase_progress={len(scenarios)}/{len(source)} accepted={accepted}", flush=True)
    return scenarios


def _paraphrase_prompt(batch: list[ScenarioCard]) -> list[dict[str, str]]:
    rows = [
        {
            "scenario_id": scenario.scenario_id,
            "language": scenario.language,
            "task": scenario.task,
            "expected_categories": scenario.expected_categories,
            "missing_slots": scenario.missing_slots,
            "profile": scenario.profile,
            "original_user_message": scenario.user_message,
        }
        for scenario in batch
    ]
    return [
        {
            "role": "system",
            "content": (
                "Rewrite Indian bank insurance customer messages. Return JSON only. "
                "Preserve all facts, language, intent, refusal pressure, and insurance category labels. "
                "Do not add premiums or new facts."
            ),
        },
        {
            "role": "user",
            "content": (
                "Paraphrase each original_user_message into natural customer language with a different wording style. "
                "For reject_premature_recommendation cases, keep the user's refusal to answer questions. "
                "Return {\"scenarios\":[{\"scenario_id\":\"...\",\"user_message\":\"...\"}]}.\n"
                f"{json.dumps({'scenarios': rows}, ensure_ascii=False)}"
            ),
        },
    ]


def _extract_paraphrases(raw: str) -> list[dict[str, Any]]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    rows = parsed.get("scenarios", [])
    return rows if isinstance(rows, list) else []


def _valid_paraphrase(message: str, scenario: ScenarioCard) -> bool:
    if len(message) < 20 or message == scenario.user_message:
        return False
    if scenario.task == "reject_premature_recommendation":
        lowered = message.lower()
        return any(term in lowered for term in ("no questions", "do not want", "don't want", "without", "details", "answer"))
    return True


def _sample_source_scenarios(path: str | Path, count: int, seed: int) -> list[ScenarioCard]:
    scenarios = _read_scenarios(Path(path))
    rng = random.Random(seed)
    rng.shuffle(scenarios)
    return scenarios[:count]


def _read_scenarios(path: str | Path) -> list[ScenarioCard]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(ScenarioCard(**row))
    return rows


def _write_splits(output: Path, prefix: str, rows: list[dict[str, Any]], splits: dict[str, list[dict[str, Any]]]) -> None:
    by_id = {row.get("case_id") or row.get("id", "").split(":")[0]: row for row in rows}
    for split_name, split_cases in splits.items():
        _write_jsonl(output / f"{prefix}_{split_name}.jsonl", [by_id[case["case_id"]] for case in split_cases])


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
