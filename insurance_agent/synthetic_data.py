"""Synthetic data generation for Indian insurance-agent training.

The generator is deterministic by default. It creates scenario cards from a
controlled taxonomy, grounds every recommendable case in local PDF chunks, and
exports training rows for SFT, DPO, and RLVR-style verifiable rewards.
"""

from __future__ import annotations

import json
import random
import copy
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from insurance_agent.eval_data import CATEGORY_QUERIES, INCOME_BANDS_INR, INDIAN_CITIES, LANGUAGES, OCCUPATIONS
from insurance_agent.production_agent import REQUIRED_SLOTS, ProductionInsuranceAgent
from insurance_agent.rag_index import InsuranceRAGIndex
from insurance_agent.vllm_testbed import VLLMConfig, call_vllm_messages


SYSTEM_PROMPT = (
    "You are an insurance advisor for an Indian bank. Ask suitability questions before recommending. "
    "Use official policy evidence, cite source_ids, never invent premiums, and mention GST quote, "
    "eligibility, exclusions, waiting periods, network/cashless checks, and underwriting."
)

TOOLS = [
    "ask_suitability_questions",
    "retrieve_policy_evidence",
    "rank_insurance_options",
    "generate_recommendation",
    "refuse_premature_recommendation",
    "finish",
]

SCENARIO_TYPES = (
    "new_family_health",
    "senior_parent_ped",
    "new_car_owner",
    "two_wheeler_owner",
    "upi_cyber_risk",
    "home_loan_property",
    "student_international_travel",
    "domestic_frequent_travel",
    "rural_low_budget",
    "self_employed_income_protection",
    "compare_two_needs",
    "premature_recommendation_pressure",
    "missing_budget",
    "claim_misconception",
    "price_only_request",
)


@dataclass(frozen=True)
class ScenarioCard:
    scenario_id: str
    scenario_type: str
    language: str
    user_message: str
    profile: dict[str, Any]
    expected_categories: list[str]
    avoid_categories: list[str]
    missing_slots: list[str]
    risk_tags: list[str]
    task: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_sdg_pipeline(
    chunks_path: str | Path,
    output_dir: str | Path,
    count: int = 500,
    seed: int = 42,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    fast_oracle: bool = True,
    scenario_source: str = "rules",
    llm_config: VLLMConfig | None = None,
    llm_batch_size: int = 10,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index = InsuranceRAGIndex.from_chunks_file(chunks_path)
    if scenario_source == "rules":
        scenarios = generate_scenarios(count=count, seed=seed)
    elif scenario_source == "llm":
        if llm_config is None:
            raise ValueError("llm_config is required when scenario_source='llm'.")
        scenarios = generate_llm_scenarios(count=count, config=llm_config, seed=seed, batch_size=llm_batch_size)
    elif scenario_source == "hybrid":
        if llm_config is None:
            raise ValueError("llm_config is required when scenario_source='hybrid'.")
        llm_count = int(count * 0.8)
        rule_count = count - llm_count
        scenarios = generate_llm_scenarios(count=llm_count, config=llm_config, seed=seed, batch_size=llm_batch_size)
        scenarios.extend(generate_scenarios(count=rule_count, seed=seed + 17))
        scenarios = _renumber_scenarios(scenarios)
    else:
        raise ValueError(f"Unknown scenario_source: {scenario_source}")
    benchmark_cases = build_benchmark_cases(scenarios, index)
    production_rows = generate_fast_oracle_traces(benchmark_cases) if fast_oracle else generate_oracle_traces(benchmark_cases, index)
    sft_rows = export_sft_rows(benchmark_cases, production_rows)
    dpo_rows = export_dpo_rows(benchmark_cases, production_rows)
    rlvr_rows = export_rlvr_rows(benchmark_cases, production_rows)
    splits = split_rows(benchmark_cases, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    quality_report = validate_sdg_outputs(scenarios, benchmark_cases, production_rows, sft_rows, dpo_rows, rlvr_rows)
    quality_report["splits"] = {name: len(rows) for name, rows in splits.items()}
    quality_report["recommended_use"] = _recommended_use(count)
    quality_report["scenario_source"] = scenario_source

    paths = {
        "scenarios": output / "synthetic_scenarios.jsonl",
        "benchmark": output / "synthetic_benchmark.jsonl",
        "oracle_predictions": output / "synthetic_oracle_predictions.jsonl",
        "sft": output / "synthetic_sft.jsonl",
        "dpo": output / "synthetic_dpo.jsonl",
        "rlvr": output / "synthetic_rlvr.jsonl",
        "report": output / "synthetic_report.json",
    }
    _write_jsonl(paths["scenarios"], [item.to_dict() for item in scenarios])
    _write_jsonl(paths["benchmark"], benchmark_cases)
    _write_jsonl(paths["oracle_predictions"], production_rows)
    _write_jsonl(paths["sft"], sft_rows)
    _write_jsonl(paths["dpo"], dpo_rows)
    _write_jsonl(paths["rlvr"], rlvr_rows)
    _write_splits(output, "benchmark", benchmark_cases, splits)
    _write_splits(output, "sft", sft_rows, splits)
    _write_splits(output, "dpo", dpo_rows, splits)
    _write_splits(output, "rlvr", rlvr_rows, splits)
    paths["report"].write_text(json.dumps(quality_report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "report": quality_report,
    }


def generate_scenarios(count: int, seed: int = 42) -> list[ScenarioCard]:
    rng = random.Random(seed)
    scenarios = []
    for idx in range(count):
        scenario_type = SCENARIO_TYPES[idx % len(SCENARIO_TYPES)]
        language = LANGUAGES[(idx + rng.randrange(len(LANGUAGES))) % len(LANGUAGES)]
        city, tier, region = INDIAN_CITIES[(idx * 3 + rng.randrange(len(INDIAN_CITIES))) % len(INDIAN_CITIES)]
        profile = _base_profile(idx, rng, language, city, tier, region)
        expected, avoid, missing, risks, task = _scenario_policy(scenario_type, idx, rng)
        profile["primary_need"] = _need_text(expected, scenario_type)
        message = _message_for(scenario_type, language, profile, expected)
        if missing:
            for slot in missing:
                profile.pop(slot, None)
        scenarios.append(
            ScenarioCard(
                scenario_id=f"sdg_{idx:06d}",
                scenario_type=scenario_type,
                language=language,
                user_message=message,
                profile=profile,
                expected_categories=expected,
                avoid_categories=avoid,
                missing_slots=missing,
                risk_tags=risks,
                task=task,
            )
        )
    return scenarios


def generate_llm_scenarios(
    count: int,
    config: VLLMConfig,
    seed: int = 42,
    batch_size: int = 10,
) -> list[ScenarioCard]:
    scenarios: list[ScenarioCard] = []
    attempt = 0
    max_attempts = max(8, (count // max(batch_size, 1)) * 4)
    while len(scenarios) < count and attempt < max_attempts:
        attempt += 1
        needed = min(batch_size, count - len(scenarios))
        raw = call_vllm_messages(_llm_scenario_messages(needed, seed, attempt, scenarios), config)
        parsed = _extract_json(raw)
        rows = parsed.get("scenarios", parsed if isinstance(parsed, list) else []) if isinstance(parsed, (dict, list)) else []
        for row in rows:
            card = _normalize_llm_scenario(row, f"sdg_{len(scenarios):06d}")
            if card and not _is_duplicate_scenario(card, scenarios):
                scenarios.append(card)
            if len(scenarios) >= count:
                break
    if len(scenarios) < count:
        raise RuntimeError(f"LLM generated only {len(scenarios)}/{count} valid scenarios after {attempt} attempts.")
    return scenarios


def _llm_scenario_messages(
    count: int,
    seed: int,
    attempt: int,
    existing: list[ScenarioCard],
) -> list[dict[str, str]]:
    categories = sorted(CATEGORY_QUERIES)
    recent = [
        {
            "scenario_type": item.scenario_type,
            "language": item.language,
            "user_message": item.user_message[:180],
            "expected_categories": item.expected_categories,
        }
        for item in existing[-12:]
    ]
    schema = {
        "scenarios": [
            {
                "scenario_type": "short_snake_case",
                "language": "English|Hindi|Hinglish|Tamil|Telugu|Marathi|Bengali|Kannada|Malayalam",
                "user_message": "realistic customer message, not templated",
                "profile": {
                    "language": "same as language",
                    "age": 35,
                    "city": "Indian city",
                    "city_tier": "metro|tier1|tier2|rural",
                    "region": "North|South|East|West|Central|North East",
                    "family_members": 4,
                    "dependents": 2,
                    "occupation": "realistic occupation",
                    "income_band_inr": "3-5L|5-8L|8-12L|12-18L|18-30L|30L+",
                    "budget_band": "low|medium|high",
                    "existing_cover": "none/employer/family floater/etc",
                    "primary_need": "customer's insurance need",
                },
                "expected_categories": ["health"],
                "avoid_categories": ["motor"],
                "missing_slots": [],
                "risk_tags": ["pre_existing_condition"],
                "task": "grounded_personalized_recommendation|compare_suitable_categories|reject_premature_recommendation",
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "You generate synthetic training scenarios for an Indian bank insurance advisor. "
                "Return JSON only. Create realistic, diverse customer situations. Do not copy templates. "
                "Do not mention exact premiums. Keep facts plausible for India."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate {count} unique scenarios. Seed={seed}, attempt={attempt}.\n"
                f"Allowed categories: {categories}\n"
                f"Allowed languages: {list(LANGUAGES)}\n"
                "Rules:\n"
                "- For reject_premature_recommendation, expected_categories must be [] and missing_slots should include all key suitability slots.\n"
                "- In each batch of 10, include 1-2 reject_premature_recommendation cases. Their user_message must explicitly say they want a recommendation/price without answering questions or sharing details.\n"
                "- For recommendation/comparison, expected_categories must use only allowed categories.\n"
                "- Include Hindi/Hinglish/regional language messages naturally when selected.\n"
                "- Include messy Indian realities: employer cover, PED, senior parents, UPI fraud, home loan, two-wheeler/car, rural budget, student travel, self-employed income risk.\n"
                "- Avoid repeating these recent examples:\n"
                f"{json.dumps(recent, ensure_ascii=False)}\n"
                f"Return exactly this JSON shape:\n{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def build_benchmark_cases(scenarios: list[ScenarioCard], index: InsuranceRAGIndex, top_k: int = 5) -> list[dict[str, Any]]:
    evidence_cache = _build_evidence_cache(index, pool_size=max(100, top_k * 20))
    cases = []
    for scenario in scenarios:
        required_slots = list(REQUIRED_SLOTS)
        expected_source_ids: list[str] = []
        evidence: list[dict[str, Any]] = []
        if scenario.task != "reject_premature_recommendation":
            primary = scenario.expected_categories[0]
            evidence = _sample_evidence(evidence_cache[primary], scenario.scenario_id, limit=min(2, top_k))
            expected_source_ids = sorted({item["source_id"] for item in evidence})

        case = {
            "case_id": scenario.scenario_id,
            "task": scenario.task,
            "scenario_type": scenario.scenario_type,
            "language": scenario.language,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": scenario.user_message},
            ],
            "profile": scenario.profile,
            "risk_tags": scenario.risk_tags,
            "evidence": evidence,
            "tools": [{"name": name} for name in TOOLS],
            "expected_final": {
                "expected_categories": scenario.expected_categories,
                "avoid_categories": scenario.avoid_categories,
                "must_ask_slots": required_slots,
                "must_cite_source_ids": expected_source_ids,
            },
        }
        cases.append(case)
    return cases


def _normalize_llm_scenario(row: Any, scenario_id: str) -> ScenarioCard | None:
    if not isinstance(row, dict):
        return None
    language = str(row.get("language", "English")).strip()
    if language not in LANGUAGES:
        language = "English"
    profile = row.get("profile")
    if not isinstance(profile, dict):
        return None
    profile = dict(profile)
    profile["language"] = language
    required_profile_defaults = {
        "age": 35,
        "city": "Mumbai",
        "city_tier": "metro",
        "region": "West",
        "family_members": 1,
        "dependents": 0,
        "occupation": "salaried_private",
        "income_band_inr": "8-12L",
        "budget_band": "medium",
        "existing_cover": "none",
        "primary_need": "insurance recommendation",
    }
    for key, value in required_profile_defaults.items():
        profile.setdefault(key, value)
    try:
        profile["age"] = int(profile["age"])
        profile["family_members"] = int(profile["family_members"])
        profile["dependents"] = int(profile["dependents"])
    except (TypeError, ValueError):
        return None

    message = str(row.get("user_message", "")).strip()
    task = str(row.get("task", "grounded_personalized_recommendation"))
    if task not in {"grounded_personalized_recommendation", "compare_suitable_categories", "reject_premature_recommendation"}:
        task = "grounded_personalized_recommendation"

    expected = _valid_categories(row.get("expected_categories", []))
    avoid = _valid_categories(row.get("avoid_categories", []))
    refusal_aligned = _looks_like_premature_or_refusal(message)
    if task == "reject_premature_recommendation" and refusal_aligned:
        expected = []
        avoid = sorted(CATEGORY_QUERIES)
        missing = list(REQUIRED_SLOTS)
    else:
        if task == "reject_premature_recommendation" and not refusal_aligned:
            task = "grounded_personalized_recommendation"
        if not expected:
            expected = [_infer_category(profile.get("primary_need", ""), message)]
        missing = [slot for slot in row.get("missing_slots", []) if slot in REQUIRED_SLOTS]
    if len(expected) > 1:
        task = "compare_suitable_categories"

    if len(message) < 20:
        message = _message_for(str(row.get("scenario_type", "llm_generated")), language, profile, expected)

    scenario_type = str(row.get("scenario_type", "llm_generated")).strip().lower().replace(" ", "_")[:80]
    if not scenario_type:
        scenario_type = "llm_generated"
    risk_tags = [str(item)[:60] for item in row.get("risk_tags", []) if isinstance(item, str)]
    return ScenarioCard(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        language=language,
        user_message=message,
        profile=profile,
        expected_categories=expected,
        avoid_categories=avoid,
        missing_slots=missing,
        risk_tags=risk_tags,
        task=task,
    )


def _valid_categories(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    categories = []
    for value in values:
        item = str(value).strip().lower()
        if item in CATEGORY_QUERIES and item not in categories:
            categories.append(item)
    return categories


def _looks_like_premature_or_refusal(message: str) -> bool:
    text = message.lower()
    refusal_terms = (
        "do not want to answer",
        "don't want to answer",
        "without questions",
        "no questions",
        "share details",
        "recommend immediately",
        "recommend one policy immediately",
        "price only",
        "premium only",
        "bas batao",
        "sawal nahi",
        "questions nahi",
        "details nahi",
    )
    return any(term in text for term in refusal_terms)


def _extract_json(text: str) -> Any:
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
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    return None


def _infer_category(primary_need: Any, user_message: Any) -> str:
    text = f"{primary_need} {user_message}".lower()
    scores = {
        category: sum(term.lower() in text for term in terms)
        for category, terms in CATEGORY_QUERIES.items()
    }
    best = max(scores.items(), key=lambda item: item[1])
    return best[0] if best[1] else "health"


def _is_duplicate_scenario(card: ScenarioCard, existing: list[ScenarioCard]) -> bool:
    normalized_message = " ".join(card.user_message.lower().split())
    return any(" ".join(item.user_message.lower().split()) == normalized_message for item in existing)


def _renumber_scenarios(scenarios: list[ScenarioCard]) -> list[ScenarioCard]:
    return [
        ScenarioCard(
            scenario_id=f"sdg_{idx:06d}",
            scenario_type=item.scenario_type,
            language=item.language,
            user_message=item.user_message,
            profile=item.profile,
            expected_categories=item.expected_categories,
            avoid_categories=item.avoid_categories,
            missing_slots=item.missing_slots,
            risk_tags=item.risk_tags,
            task=item.task,
        )
        for idx, item in enumerate(scenarios)
    ]


def _build_evidence_cache(index: InsuranceRAGIndex, pool_size: int) -> dict[str, list[dict[str, Any]]]:
    cache = {}
    for category, terms in CATEGORY_QUERIES.items():
        query = f"{category} {' '.join(terms)} policy wording prospectus coverage exclusions waiting period"
        results = index.search(query, category, top_k=pool_size)
        unique_by_source = []
        seen_sources = set()
        for item in results:
            if item.source_id in seen_sources and len(unique_by_source) < pool_size // 2:
                continue
            seen_sources.add(item.source_id)
            unique_by_source.append(
                {
                    "chunk_id": item.chunk_id,
                    "source_id": item.source_id,
                    "title": item.title,
                    "url": item.url,
                    "excerpt": item.text[:700],
                }
            )
            if len(unique_by_source) >= pool_size:
                break
        if not unique_by_source:
            raise ValueError(f"No evidence found for category: {category}")
        cache[category] = unique_by_source
    return cache


def _sample_evidence(pool: list[dict[str, Any]], scenario_id: str, limit: int) -> list[dict[str, Any]]:
    idx = int(scenario_id.rsplit("_", 1)[-1])
    selected = []
    for offset in range(max(limit * 3, 3)):
        item = pool[(idx + offset * 17) % len(pool)]
        if item["source_id"] not in {row["source_id"] for row in selected}:
            selected.append(item)
        if len(selected) >= limit:
            break
    if not selected:
        selected.append(pool[idx % len(pool)])
    return selected


def generate_oracle_traces(cases: list[dict[str, Any]], index: InsuranceRAGIndex) -> list[dict[str, Any]]:
    agent = ProductionInsuranceAgent(index)
    return [agent.run_case(case).to_prediction_row() for case in cases]


def generate_fast_oracle_traces(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate production-shaped traces directly from benchmark evidence.

    This is for large synthetic training runs. It preserves the same tool names,
    argument schema, observations, and final-response shape as the controlled
    production agent, while avoiding a BM25 search for every row.
    """

    rows = []
    for case in cases:
        expected = case["expected_final"]
        if case["task"] == "reject_premature_recommendation":
            calls = [
                {
                    "turn": 1,
                    "role": "assistant",
                    "tool_name": "refuse_premature_recommendation",
                    "arguments": {
                        "missing_slots": expected["must_ask_slots"],
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
            final_response = {
                "decision": "cannot_recommend_yet",
                "ask_slots": expected["must_ask_slots"],
                "reason": "I need suitability details before recommending an insurance product.",
            }
            rows.append(_prediction_row(case, calls, [_observation(calls[0], {"decision": "cannot_recommend_yet"})], final_response))
            continue

        categories = expected["expected_categories"]
        primary_category = categories[0]
        evidence = [_evidence_as_result(item) for item in case["evidence"]]
        source_ids = [item["source_id"] for item in evidence]
        calls = [
            {
                "turn": 1,
                "role": "assistant",
                "tool_name": "ask_suitability_questions",
                "arguments": {
                    "slots": expected["must_ask_slots"],
                    "language": case["language"],
                    "questions": _questions(case["language"]),
                },
            },
            {
                "turn": 2,
                "role": "assistant",
                "tool_name": "retrieve_policy_evidence",
                "arguments": {
                    "query": f"{primary_category} {case['profile'].get('primary_need', '')} policy wording prospectus coverage exclusions waiting period India",
                    "category": primary_category,
                    "top_k": max(5, len(source_ids)),
                },
            },
            {
                "turn": 3,
                "role": "assistant",
                "tool_name": "rank_insurance_options",
                "arguments": {
                    "profile": case["profile"],
                    "candidate_categories": categories,
                    "avoid_categories": expected["avoid_categories"],
                },
            },
            {
                "turn": 4,
                "role": "assistant",
                "tool_name": "generate_recommendation",
                "arguments": {
                    "recommended_categories": categories,
                    "source_ids": source_ids,
                    "language": case["language"],
                    "include_caveats": True,
                },
            },
            {
                "turn": 5,
                "role": "assistant",
                "tool_name": "finish",
                "arguments": {"status": "completed", "summary": "Generated a grounded insurance recommendation."},
            },
        ]
        final_response = {
            "recommendation": [
                {
                    "category": category,
                    "why_suitable": "Matches the stated Indian customer profile and should be validated against official policy wording.",
                    "source_ids": source_ids,
                }
                for category in categories
            ],
            "next_step": "Generate an official premium quote with GST and check eligibility, exclusions, waiting periods, network/cashless availability, and underwriting.",
            "caveats": [
                "Do not treat this as final underwriting approval.",
                "Final purchase depends on official policy wording, premium quote, disclosures, exclusions, waiting periods, and insurer underwriting.",
            ],
        }
        observations = [
            _observation(calls[0], {"collected_profile": case["profile"]}),
            _observation(calls[1], {"results": evidence}),
            _observation(calls[2], {"ranked_categories": categories, "avoid_categories": expected["avoid_categories"]}),
            _observation(calls[3], {"recommendation": final_response}),
            _observation(calls[4], {"summary": calls[4]["arguments"]["summary"]}),
        ]
        rows.append(_prediction_row(case, calls, observations, final_response))
    return rows


def export_sft_rows(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case = {row["case_id"]: row for row in predictions}
    rows = []
    for case in cases:
        prediction = by_case[case["case_id"]]
        rows.append(
            {
                "id": f"{case['case_id']}:tool_trace",
                "task": "insurance_agent_tool_trace_sft",
                "messages": [
                    *case["messages"],
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "tool_calls": prediction["predicted_tool_calls"],
                                "final_response": prediction["final_response"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                "metadata": _training_metadata(case),
            }
        )
    return rows


def export_dpo_rows(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case = {row["case_id"]: row for row in predictions}
    rows = []
    for case in cases:
        chosen = {
            "tool_calls": by_case[case["case_id"]]["predicted_tool_calls"],
            "final_response": by_case[case["case_id"]]["final_response"],
        }
        rejected = _hard_negative_response(case, chosen)
        rows.append(
            {
                "id": f"{case['case_id']}:preference",
                "prompt": case["messages"],
                "chosen": json.dumps(chosen, ensure_ascii=False, sort_keys=True),
                "rejected": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
                "metadata": _training_metadata(case),
            }
        )
    return rows


def export_rlvr_rows(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case = {row["case_id"]: row for row in predictions}
    rows = []
    for case in cases:
        rows.append(
            {
                "id": f"{case['case_id']}:rlvr",
                "prompt": case["messages"],
                "reference": {
                    "tool_calls": by_case[case["case_id"]]["predicted_tool_calls"],
                    "expected_final": case["expected_final"],
                },
                "reward_spec": {
                    "valid_json": 1.0,
                    "correct_workflow_order": 2.0,
                    "required_slots_present": 1.5,
                    "retrieval_before_recommendation": 1.0,
                    "source_ids_from_retrieval": 2.0,
                    "safe_refusal_when_premature": 2.0,
                    "no_fake_premium": 1.0,
                    "compliance_caveat": 1.0,
                },
                "metadata": _training_metadata(case),
            }
        )
    return rows


def validate_sdg_outputs(
    scenarios: list[ScenarioCard],
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    dpo_rows: list[dict[str, Any]],
    rlvr_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    case_ids = [case["case_id"] for case in cases]
    pred_ids = [row["case_id"] for row in predictions]
    normal_cases = [case for case in cases if case["task"] != "reject_premature_recommendation"]
    refusal_cases = [case for case in cases if case["task"] == "reject_premature_recommendation"]
    scenario_counts = Counter(case["scenario_type"] for case in cases)
    language_counts = Counter(case["language"] for case in cases)
    category_counts = Counter(category for case in cases for category in case["expected_final"]["expected_categories"])
    errors = []
    warnings = []
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate_case_ids")
    if set(case_ids) != set(pred_ids):
        errors.append("prediction_case_id_mismatch")
    if not all(case["expected_final"]["must_ask_slots"] for case in cases):
        errors.append("missing_required_slots")
    if not all(case["expected_final"]["must_cite_source_ids"] for case in normal_cases):
        errors.append("normal_case_without_source_ids")
    if not all(not case["expected_final"]["must_cite_source_ids"] for case in refusal_cases):
        errors.append("refusal_case_has_source_ids")
    if not all(_looks_like_premature_or_refusal(case["messages"][-1]["content"]) for case in refusal_cases):
        errors.append("refusal_case_not_aligned_with_user_message")
    if not (len(cases) == len(predictions) == len(sft_rows) == len(dpo_rows) == len(rlvr_rows)):
        errors.append("export_count_mismatch")
    if len(cases) >= 50 and len(refusal_cases) / len(cases) < 0.08:
        warnings.append("low_refusal_case_coverage")
    if len(cases) >= 50 and max(category_counts.values(), default=0) / max(sum(category_counts.values()), 1) > 0.55:
        warnings.append("category_distribution_imbalanced")

    return {
        "scenario_count": len(scenarios),
        "benchmark_case_count": len(cases),
        "oracle_prediction_count": len(predictions),
        "sft_count": len(sft_rows),
        "dpo_count": len(dpo_rows),
        "rlvr_count": len(rlvr_rows),
        "normal_case_count": len(normal_cases),
        "refusal_case_count": len(refusal_cases),
        "languages": sorted(language_counts),
        "language_counts": dict(sorted(language_counts.items())),
        "scenario_types": sorted(scenario_counts),
        "scenario_type_counts": dict(sorted(scenario_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "unique_user_messages": len({case["messages"][-1]["content"] for case in cases}),
        "duplicate_user_message_count": len(cases) - len({case["messages"][-1]["content"] for case in cases}),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def split_rows(
    cases: list[dict[str, Any]],
    seed: int,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _base_profile(idx: int, rng: random.Random, language: str, city: str, tier: str, region: str) -> dict[str, Any]:
    age = rng.choice([22, 27, 31, 36, 42, 48, 55, 62, 69])
    family_members = rng.choice([1, 2, 3, 4, 5, 6])
    dependents = max(0, min(family_members - 1, rng.choice([0, 1, 2, 3, 4])))
    return {
        "language": language,
        "age": age,
        "city": city,
        "city_tier": tier,
        "region": region,
        "family_members": family_members,
        "dependents": dependents,
        "occupation": OCCUPATIONS[(idx + rng.randrange(len(OCCUPATIONS))) % len(OCCUPATIONS)],
        "income_band_inr": INCOME_BANDS_INR[(idx + rng.randrange(len(INCOME_BANDS_INR))) % len(INCOME_BANDS_INR)],
        "budget_band": rng.choice(["low", "medium", "high"]),
        "existing_cover": rng.choice(["none", "employer health cover", "old low sum insured policy", "family floater exists"]),
    }


def _scenario_policy(
    scenario_type: str, idx: int, rng: random.Random
) -> tuple[list[str], list[str], list[str], list[str], str]:
    policies = {
        "new_family_health": (["health"], ["motor", "travel"], [], ["family", "hospitalization"], "grounded_personalized_recommendation"),
        "senior_parent_ped": (["health"], ["motor", "cyber"], [], ["senior_parent", "pre_existing_condition"], "grounded_personalized_recommendation"),
        "new_car_owner": (["motor", "personal_accident"], ["travel", "home"], [], ["new_vehicle"], "compare_suitable_categories"),
        "two_wheeler_owner": (["motor", "personal_accident"], ["home", "travel"], [], ["two_wheeler"], "compare_suitable_categories"),
        "upi_cyber_risk": (["cyber"], ["motor", "travel"], [], ["upi", "netbanking"], "grounded_personalized_recommendation"),
        "home_loan_property": (["home"], ["travel", "cyber"], [], ["home_loan"], "grounded_personalized_recommendation"),
        "student_international_travel": (["travel"], ["home", "motor"], [], ["student", "international_trip"], "grounded_personalized_recommendation"),
        "domestic_frequent_travel": (["travel"], ["home", "cyber"], [], ["domestic_travel"], "grounded_personalized_recommendation"),
        "rural_low_budget": (["personal_accident", "health"], ["travel", "cyber"], [], ["rural", "low_budget"], "compare_suitable_categories"),
        "self_employed_income_protection": (["personal_accident", "health"], ["travel", "home"], [], ["self_employed"], "compare_suitable_categories"),
        "compare_two_needs": (["health", "cyber"], ["motor"], [], ["comparison"], "compare_suitable_categories"),
        "premature_recommendation_pressure": ([], list(CATEGORY_QUERIES), list(REQUIRED_SLOTS), ["premature_recommendation"], "reject_premature_recommendation"),
        "missing_budget": (["health"], ["motor"], ["budget_band"], ["missing_budget"], "grounded_personalized_recommendation"),
        "claim_misconception": (["health"], ["travel"], [], ["claim_misconception"], "grounded_personalized_recommendation"),
        "price_only_request": ([], list(CATEGORY_QUERIES), list(REQUIRED_SLOTS), ["price_only"], "reject_premature_recommendation"),
    }
    expected, avoid, missing, risks, task = policies[scenario_type]
    if scenario_type == "compare_two_needs" and idx % 2:
        expected = ["home", "cyber"]
        avoid = ["motor", "travel"]
    if scenario_type == "missing_budget" and rng.random() < 0.5:
        missing = ["existing_cover", "budget_band"]
    return list(expected), list(avoid), list(missing), list(risks), task


def _need_text(expected: list[str], scenario_type: str) -> str:
    if not expected:
        return "quick insurance recommendation without suitability details"
    terms = {
        "health": "health cover and hospitalization protection",
        "motor": "vehicle protection and mandatory motor cover",
        "travel": "trip medical and travel protection",
        "cyber": "UPI, card, and netbanking fraud protection",
        "home": "home structure and contents protection",
        "personal_accident": "accidental death and disability protection",
    }
    if scenario_type == "senior_parent_ped":
        return "senior parent health cover with diabetes or hypertension disclosure"
    return " and ".join(terms[item] for item in expected)


def _message_for(scenario_type: str, language: str, profile: dict[str, Any], expected: list[str]) -> str:
    need = profile["primary_need"]
    city = profile.get("city", "my city")
    budget = profile.get("budget_band", "my budget")
    age = profile.get("age", "unknown age")
    occupation = str(profile.get("occupation", "customer")).replace("_", " ")
    family = profile.get("family_members", "unknown")
    dependents = profile.get("dependents", "unknown")
    cover = profile.get("existing_cover", "not sure")
    detail = (
        f"age {age}, {occupation}, family {family}, dependents {dependents}, "
        f"city {city}, existing cover {cover}, budget {budget}"
    )
    if scenario_type in {"premature_recommendation_pressure", "price_only_request"}:
        return f"Recommend one policy immediately for me in {city}. I do not want to answer questions or share details. Context: {occupation}, budget {budget}."
    if scenario_type == "claim_misconception":
        return f"I want a policy for {need} where all claims are guaranteed and no waiting period applies. My details: {detail}."
    if language == "Hindi":
        return f"Mujhe {need} ke liye insurance chahiye. Details: {detail}. Kya suitable hoga?"
    if language == "Hinglish":
        return f"Mujhe {need} chahiye. Details: {detail}. Options confusing hain."
    if language == "Tamil":
        return f"எனக்கு {need} க்கான insurance வேண்டும். Details: {detail}. சரியான option சொல்லுங்கள்."
    if language == "Telugu":
        return f"నాకు {need} కోసం insurance కావాలి. Details: {detail}. సరైన option చెప్పండి."
    if language == "Marathi":
        return f"मला {need} साठी insurance पाहिजे. Details: {detail}. योग्य option सांगा."
    if language == "Bengali":
        return f"আমার {need} এর জন্য insurance দরকার. Details: {detail}. কোন option ভালো হবে?"
    if language == "Kannada":
        return f"ನನಗೆ {need}ಗಾಗಿ insurance ಬೇಕು. Details: {detail}. ಸರಿಯಾದ option ಹೇಳಿ."
    if language == "Malayalam":
        return f"എനിക്ക് {need} വേണ്ടി insurance വേണം. Details: {detail}. ശരിയായ option പറയൂ."
    return f"I need insurance for {need}. My details: {detail}. Please recommend what suits my profile."


def _questions(language: str) -> list[str]:
    if language in {"Hindi", "Hinglish"}:
        return [
            "Aapki age, city, family size aur dependents kitne hain?",
            "Existing insurance cover, budget range aur primary need kya hai?",
            "Koi pre-existing disease, vehicle/travel/home/UPI usage disclosure hai?",
        ]
    return ["What are the age, city, family size, dependents, existing cover, budget, primary need, and key disclosures?"]


def _evidence_as_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": item["chunk_id"],
        "source_id": item["source_id"],
        "title": item["title"],
        "url": item["url"],
        "text": item["excerpt"],
        "score": 1.0,
    }


def _observation(call: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {"turn": call["turn"], "tool_name": call["tool_name"], "status": "ok", "payload": payload}


def _prediction_row(
    case: dict[str, Any],
    calls: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    final_response: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "predicted_tool_calls": calls,
        "observations": observations,
        "final_response": final_response,
        "audit": [],
        "agent_status": "completed",
        "raw_response": json.dumps(final_response, ensure_ascii=False, sort_keys=True),
    }


def _write_splits(output: Path, prefix: str, rows: list[dict[str, Any]], splits: dict[str, list[dict[str, Any]]]) -> None:
    by_id = {row.get("case_id") or row.get("id", "").split(":")[0]: row for row in rows}
    for split_name, split_cases in splits.items():
        split_rows_for_prefix = [by_id[case["case_id"]] for case in split_cases]
        _write_jsonl(output / f"{prefix}_{split_name}.jsonl", split_rows_for_prefix)


def _recommended_use(count: int) -> str:
    if count < 2_000:
        return "smoke_test_only"
    if count < 20_000:
        return "pilot_sft_or_eval_expansion"
    if count < 100_000:
        return "usable_lora_sft_starting_point"
    return "large_sft_dpo_candidate_after_quality_review"


def _training_metadata(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "language": case["language"],
        "scenario_type": case["scenario_type"],
        "task": case["task"],
        "risk_tags": case["risk_tags"],
    }


def _hard_negative_response(case: dict[str, Any], chosen: dict[str, Any]) -> dict[str, Any]:
    """Create a same-shape negative with one targeted defect.

    The negative intentionally keeps the chosen trace length and structure close
    to the positive label so DPO cannot solve the task with a trivial
    "longer-is-better" shortcut.
    """

    rejected = copy.deepcopy(chosen)
    modes = (
        "wrong_category",
        "missing_citation",
        "fabricated_premium",
        "premature_recommendation",
        "wrong_tool_order",
    )
    mode = modes[_stable_index(case["case_id"], len(modes))]
    if case["task"] == "reject_premature_recommendation":
        mode = "premature_recommendation"

    if mode == "wrong_category":
        _inject_wrong_category(case, rejected)
    elif mode == "missing_citation":
        _remove_citations(rejected)
    elif mode == "fabricated_premium":
        _inject_fake_premium(rejected)
    elif mode == "premature_recommendation":
        _inject_premature_recommendation(case, rejected)
    elif mode == "wrong_tool_order":
        _swap_retrieval_and_recommendation(rejected)

    rejected["negative_type"] = mode
    return rejected


def _inject_wrong_category(case: dict[str, Any], row: dict[str, Any]) -> None:
    expected = set(case["expected_final"].get("expected_categories", []))
    wrong = next((category for category in sorted(CATEGORY_QUERIES) if category not in expected), "health")
    for call in row.get("tool_calls", []):
        args = call.get("arguments", {})
        if call.get("tool_name") == "rank_insurance_options":
            args["candidate_categories"] = [wrong]
        if call.get("tool_name") == "generate_recommendation":
            args["recommended_categories"] = [wrong]
    final = row.get("final_response")
    if isinstance(final, dict) and isinstance(final.get("recommendation"), list):
        for item in final["recommendation"]:
            if isinstance(item, dict):
                item["category"] = wrong
                item["why_suitable"] = (
                    "This category is suggested despite not matching the stated primary need; "
                    "all policy wording, GST quote, exclusions, waiting periods, network/cashless checks, "
                    "and underwriting still need review."
                )


def _remove_citations(row: dict[str, Any]) -> None:
    for call in row.get("tool_calls", []):
        args = call.get("arguments", {})
        if call.get("tool_name") == "generate_recommendation":
            args["source_ids"] = []
    final = row.get("final_response")
    if isinstance(final, dict) and isinstance(final.get("recommendation"), list):
        for item in final["recommendation"]:
            if isinstance(item, dict):
                item["source_ids"] = []
                item["why_suitable"] = (
                    f"{item.get('why_suitable', 'May suit the stated profile')} "
                    "However, this answer fails to cite the official source_ids even though policy wording, "
                    "GST quote, exclusions, waiting periods, network/cashless checks, and underwriting remain necessary."
                )


def _inject_fake_premium(row: dict[str, Any]) -> None:
    final = row.get("final_response")
    fake_text = (
        "Estimated premium: Rs 4999 per year. This number is presented as if final, "
        "even though official premium quote with GST, eligibility, exclusions, waiting periods, "
        "network/cashless checks, and underwriting are still required."
    )
    if isinstance(final, dict):
        final["premium_statement"] = fake_text
        final["next_step"] = fake_text
    else:
        row["final_response"] = {"recommendation": [], "next_step": fake_text}


def _inject_premature_recommendation(case: dict[str, Any], row: dict[str, Any]) -> None:
    category = (case["expected_final"].get("avoid_categories") or ["health"])[0]
    row["tool_calls"] = [
        {
            "turn": 1,
            "role": "assistant",
            "tool_name": "generate_recommendation",
            "arguments": {
                "recommended_categories": [category],
                "source_ids": [],
                "language": case["language"],
                "include_caveats": True,
            },
        },
        {
            "turn": 2,
            "role": "assistant",
            "tool_name": "finish",
            "arguments": {
                "status": "completed",
                "summary": (
                    "Recommended before collecting suitability details; official premium quote with GST, eligibility, "
                    "exclusions, waiting periods, network/cashless checks, and underwriting still need review."
                ),
            },
        },
    ]
    row["final_response"] = {
        "recommendation": [
            {
                "category": category,
                "why_suitable": (
                    "Suggested immediately without first collecting age, city, family members, dependents, existing cover, "
                    "budget, and primary need. Policy wording, GST quote, exclusions, waiting periods, network/cashless "
                    "checks, and underwriting are still pending."
                ),
                "source_ids": [],
            }
        ],
        "next_step": "Proceed to quote even though suitability information is incomplete.",
        "caveats": [
            "This recommendation is premature because required suitability questions were skipped.",
            "Final purchase depends on policy wording, premium quote with GST, exclusions, waiting periods, and underwriting.",
        ],
    }


def _swap_retrieval_and_recommendation(row: dict[str, Any]) -> None:
    calls = row.get("tool_calls", [])
    retrieve_idx = next((idx for idx, call in enumerate(calls) if call.get("tool_name") == "retrieve_policy_evidence"), None)
    recommend_idx = next((idx for idx, call in enumerate(calls) if call.get("tool_name") == "generate_recommendation"), None)
    if retrieve_idx is not None and recommend_idx is not None and retrieve_idx < recommend_idx:
        calls[retrieve_idx], calls[recommend_idx] = calls[recommend_idx], calls[retrieve_idx]
        for idx, call in enumerate(calls, start=1):
            call["turn"] = idx
    else:
        _remove_citations(row)


def _stable_index(text: str, modulo: int) -> int:
    return sum(ord(char) for char in text) % modulo


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
