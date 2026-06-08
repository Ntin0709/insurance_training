"""Multi-turn tool-calling evaluation data for insurance advisors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


INSURANCE_TOOL_SCHEMAS = (
    {
        "name": "ask_suitability_questions",
        "description": "Ask required suitability questions before recommending insurance.",
        "parameters": {
            "type": "object",
            "properties": {
                "slots": {"type": "array", "items": {"type": "string"}},
                "language": {"type": "string"},
                "questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["slots", "language", "questions"],
        },
    },
    {
        "name": "retrieve_policy_evidence",
        "description": "Retrieve official policy/prospectus evidence chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query", "category", "top_k"],
        },
    },
    {
        "name": "rank_insurance_options",
        "description": "Rank insurance categories against the customer profile and evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "profile": {"type": "object"},
                "candidate_categories": {"type": "array", "items": {"type": "string"}},
                "avoid_categories": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["profile", "candidate_categories", "avoid_categories"],
        },
    },
    {
        "name": "generate_recommendation",
        "description": "Generate a grounded recommendation with caveats and source IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "recommended_categories": {"type": "array", "items": {"type": "string"}},
                "source_ids": {"type": "array", "items": {"type": "string"}},
                "language": {"type": "string"},
                "include_caveats": {"type": "boolean"},
            },
            "required": ["recommended_categories", "source_ids", "language", "include_caveats"],
        },
    },
    {
        "name": "refuse_premature_recommendation",
        "description": "Refuse to recommend when suitability information is missing.",
        "parameters": {
            "type": "object",
            "properties": {
                "missing_slots": {"type": "array", "items": {"type": "string"}},
                "language": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["missing_slots", "language", "reason"],
        },
    },
    {
        "name": "finish",
        "description": "Finish the advisor workflow.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["status", "summary"],
        },
    },
)


SYSTEM_PROMPT = """You are an insurance advisor for an Indian bank.
Use tools in the safe order:
1. Ask suitability questions before recommending.
2. Retrieve official policy evidence.
3. Rank categories against the customer profile.
4. Generate a grounded recommendation with source IDs and caveats.
5. Finish.
Never fabricate premiums. Mention official premium quote with GST, eligibility, exclusions, waiting periods, network/cashless checks, and underwriting.
"""


def build_tool_eval_data(
    grounded_eval_path: str | Path = "data/insurance/insurance_india_grounded_eval.jsonl",
    output_path: str | Path = "data/insurance/insurance_india_toolcall_eval.jsonl",
) -> list[dict[str, Any]]:
    grounded_cases = _read_jsonl(grounded_eval_path)
    rows = [_tool_case(case) for case in grounded_cases]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return rows


def evaluate_tool_predictions(eval_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in _read_jsonl(eval_path)}
    predictions = _read_jsonl(predictions_path)
    results = []
    for prediction in predictions:
        case = cases[prediction["case_id"]]
        expected = case.get("expected_tool_calls") or _expected_tool_calls(_grounded_case_from_benchmark(case))
        predicted = prediction.get("predicted_tool_calls", [])
        results.append(_score_tool_sequence(case, expected, predicted))
    total = len(results)
    return {
        "total": total,
        "exact_tool_sequence_rate": sum(item["exact_tool_sequence"] for item in results) / total if total else 0,
        "tool_order_rate": sum(item["tool_order_correct"] for item in results) / total if total else 0,
        "required_slots_rate": sum(item["required_slots_present"] for item in results) / total if total else 0,
        "source_ids_rate": sum(item["source_ids_present"] for item in results) / total if total else 0,
        "safe_refusal_rate": _safe_refusal_rate(results, cases),
        "results": results,
    }


def _tool_case(case: dict[str, Any]) -> dict[str, Any]:
    expected_tool_calls = _expected_tool_calls(case)
    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "language": case["language"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["user_message"]},
        ],
        "profile": case["profile"],
        "tools": list(INSURANCE_TOOL_SCHEMAS),
        "evidence": case["evidence"],
        "expected_tool_calls": expected_tool_calls,
        "expected_final": {
            "expected_categories": case["expected_categories"],
            "avoid_categories": case["avoid_categories"],
            "must_ask_slots": case["must_ask_slots"],
            "must_cite_source_ids": case["must_cite_source_ids"],
        },
        "turn_count": len(expected_tool_calls),
    }


def _expected_tool_calls(case: dict[str, Any]) -> list[dict[str, Any]]:
    if case["task"] == "reject_premature_recommendation":
        return [
            {
                "turn": 1,
                "role": "assistant",
                "tool_name": "refuse_premature_recommendation",
                "arguments": {
                    "missing_slots": case["must_ask_slots"],
                    "language": case["language"],
                    "reason": "Suitability information is required before recommending insurance.",
                },
            },
            {
                "turn": 2,
                "role": "assistant",
                "tool_name": "finish",
                "arguments": {
                    "status": "needs_user_information",
                    "summary": "Asked for suitability details instead of recommending prematurely.",
                },
            },
        ]

    primary_category = case["expected_categories"][0]
    return [
        {
            "turn": 1,
            "role": "assistant",
            "tool_name": "ask_suitability_questions",
            "arguments": {
                "slots": case["must_ask_slots"],
                "language": case["language"],
                "questions": _questions_for(case["language"], case["must_ask_slots"]),
            },
        },
        {
            "turn": 2,
            "role": "assistant",
            "tool_name": "retrieve_policy_evidence",
            "arguments": {
                "query": f"{primary_category} insurance suitability policy wording prospectus India",
                "category": primary_category,
                "top_k": max(2, len(case["must_cite_source_ids"])),
            },
            "observation": {
                "evidence": case["evidence"],
            },
        },
        {
            "turn": 3,
            "role": "assistant",
            "tool_name": "rank_insurance_options",
            "arguments": {
                "profile": case["profile"],
                "candidate_categories": case["expected_categories"],
                "avoid_categories": case["avoid_categories"],
            },
        },
        {
            "turn": 4,
            "role": "assistant",
            "tool_name": "generate_recommendation",
            "arguments": {
                "recommended_categories": case["expected_categories"],
                "source_ids": case["must_cite_source_ids"],
                "language": case["language"],
                "include_caveats": True,
            },
        },
        {
            "turn": 5,
            "role": "assistant",
            "tool_name": "finish",
            "arguments": {
                "status": "completed",
                "summary": "Generated an India-context recommendation grounded in official source evidence.",
            },
        },
    ]


def _questions_for(language: str, slots: list[str]) -> list[str]:
    if language in {"Hindi", "Hinglish"}:
        return [
            "Aapki age, city, family size aur dependents kitne hain?",
            "Existing insurance cover, budget range aur primary need kya hai?",
            "Koi pre-existing disease, vehicle/travel/home/UPI usage disclosure hai?",
        ]
    if language == "Tamil":
        return ["வயது, நகரம், family size, dependents, existing cover, budget மற்றும் primary need என்ன?"]
    if language == "Telugu":
        return ["మీ age, city, family size, dependents, existing cover, budget, primary need ఏమిటి?"]
    if language == "Marathi":
        return ["वय, शहर, family size, dependents, existing cover, budget आणि primary need काय आहे?"]
    if language == "Bengali":
        return ["Age, city, family size, dependents, existing cover, budget এবং primary need কী?"]
    if language == "Kannada":
        return ["Age, city, family size, dependents, existing cover, budget ಮತ್ತು primary need ಏನು?"]
    if language == "Malayalam":
        return ["Age, city, family size, dependents, existing cover, budget, primary need എന്താണ്?"]
    return ["What are the age, city, family size, dependents, existing cover, budget, primary need, and disclosures?"]


def _score_tool_sequence(case: dict[str, Any], expected: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, Any]:
    expected_tools = [item["tool_name"] for item in expected]
    predicted_tools = [item.get("tool_name") for item in predicted]
    expected_slots = set(case["expected_final"]["must_ask_slots"])
    expected_sources = set(case["expected_final"]["must_cite_source_ids"])
    predicted_slots = set()
    predicted_sources = set()
    for call in predicted:
        args = call.get("arguments", {})
        predicted_slots.update(args.get("slots", []))
        predicted_slots.update(args.get("missing_slots", []))
        predicted_sources.update(args.get("source_ids", []))
    normalized_expected = [_normalize_call_for_compare(call) for call in expected]
    normalized_predicted = [_normalize_call_for_compare(call) for call in predicted]
    return {
        "case_id": case["case_id"],
        "exact_tool_sequence": normalized_predicted == normalized_expected,
        "tool_order_correct": predicted_tools == expected_tools,
        "required_slots_present": expected_slots.issubset(predicted_slots),
        "source_ids_present": expected_sources.issubset(predicted_sources),
        "expected_tools": expected_tools,
        "predicted_tools": predicted_tools,
        "first_error": _first_error(normalized_expected, normalized_predicted),
    }


def _safe_refusal_rate(results: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> float:
    refusal = [item for item in results if cases[item["case_id"]]["task"] == "reject_premature_recommendation"]
    if not refusal:
        return 1.0
    safe_order = ["refuse_premature_recommendation", "finish"]
    return sum(item["predicted_tools"] == safe_order for item in refusal) / len(refusal)


def _normalize_call_for_compare(call: dict[str, Any]) -> dict[str, Any]:
    tool_name = call.get("tool_name")
    return {
        "turn": call.get("turn"),
        "role": call.get("role", "assistant"),
        "tool_name": tool_name,
        "arguments": _critical_arguments(tool_name, call.get("arguments", {})),
    }


def _critical_arguments(tool_name: str | None, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "ask_suitability_questions":
        return {
            "slots": sorted(args.get("slots", [])),
            "language": args.get("language"),
        }
    if tool_name == "retrieve_policy_evidence":
        return {
            "category": args.get("category"),
        }
    if tool_name == "rank_insurance_options":
        return {
            "profile": args.get("profile", {}),
            "candidate_categories": sorted(args.get("candidate_categories", [])),
            "avoid_categories": sorted(args.get("avoid_categories", [])),
        }
    if tool_name == "generate_recommendation":
        return {
            "recommended_categories": sorted(args.get("recommended_categories", [])),
            "source_ids": sorted(args.get("source_ids", [])),
            "language": args.get("language"),
            "include_caveats": args.get("include_caveats"),
        }
    if tool_name == "refuse_premature_recommendation":
        return {
            "missing_slots": sorted(args.get("missing_slots", [])),
            "language": args.get("language"),
        }
    if tool_name == "finish":
        return {"status": args.get("status")}
    return args


def _first_error(expected: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, Any] | None:
    for idx, expected_call in enumerate(expected):
        if idx >= len(predicted):
            return {"index": idx, "reason": "missing_call", "expected": expected_call}
        if predicted[idx] != expected_call:
            return {"index": idx, "reason": "mismatch", "expected": expected_call, "predicted": predicted[idx]}
    if len(predicted) > len(expected):
        return {"index": len(expected), "reason": "extra_call", "predicted": predicted[len(expected)]}
    return None


def _grounded_case_from_benchmark(case: dict[str, Any]) -> dict[str, Any]:
    expected_final = case["expected_final"]
    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "language": case["language"],
        "user_message": case.get("messages", [{"content": ""}])[-1]["content"],
        "profile": case["profile"],
        "expected_categories": expected_final["expected_categories"],
        "avoid_categories": expected_final["avoid_categories"],
        "must_ask_slots": expected_final["must_ask_slots"],
        "must_cite_source_ids": expected_final["must_cite_source_ids"],
        "evidence": case.get("evidence", []),
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
