"""Scalable SDG pipeline with checkpointed LLM generation and quality gates."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from insurance_agent.eval_data import CATEGORY_QUERIES, LANGUAGES
from insurance_agent.rag_index import InsuranceRAGIndex
from insurance_agent.sdg_quality import assert_quality, quality_report
from insurance_agent.sdg_quality import normalize_text
from insurance_agent.synthetic_data import (
    ScenarioCard,
    _extract_json,
    _is_duplicate_scenario,
    _normalize_llm_scenario,
    _renumber_scenarios,
    build_benchmark_cases,
    export_dpo_rows,
    export_rlvr_rows,
    export_sft_rows,
    generate_fast_oracle_traces,
    generate_scenarios,
    split_rows,
)
from insurance_agent.vllm_testbed import VLLMConfig, call_vllm_messages


TASKS = (
    "grounded_personalized_recommendation",
    "compare_suitable_categories",
    "reject_premature_recommendation",
)
SCENARIO_VARIANTS = (
    "ordinary_customer",
    "messy_customer",
    "rural_low_literacy",
    "senior_parent_ped",
    "upi_fraud_victim",
    "vehicle_owner_confused",
    "home_loan_customer",
    "student_travel",
    "self_employed_irregular_income",
    "claims_or_waiting_period_confusion",
    "price_pressure",
    "code_mixed_regional",
)
CATEGORIES = tuple(sorted(CATEGORY_QUERIES))

BALANCED_RULE_SCENARIOS = (
    ("health", "grounded_personalized_recommendation", "family health cover with hospitalisation and PED disclosure"),
    ("motor", "grounded_personalized_recommendation", "vehicle insurance, own damage, third-party and add-on confusion"),
    ("travel", "grounded_personalized_recommendation", "domestic or international trip protection"),
    ("cyber", "grounded_personalized_recommendation", "UPI, card, netbanking or phishing fraud protection"),
    ("home", "grounded_personalized_recommendation", "home loan, building and contents protection"),
    ("personal_accident", "grounded_personalized_recommendation", "accidental death and disability income protection"),
    ("health", "compare_suitable_categories", "health cover compared with personal accident"),
    ("motor", "compare_suitable_categories", "motor cover compared with personal accident"),
    ("cyber", "compare_suitable_categories", "cyber protection compared with home or health"),
    ("health", "reject_premature_recommendation", "price pressure without suitability details"),
)


def generate_scaled_sdg_pipeline(
    chunks_path: str | Path,
    output_dir: str | Path,
    count: int,
    seed: int,
    scenario_source: str,
    llm_config: VLLMConfig | None = None,
    llm_batch_size: int = 10,
    checkpoint_every: int = 25,
    resume: bool = True,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    heldout_paths: list[str | Path] | None = None,
    allow_warnings: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenario_path = output / "synthetic_scenarios.jsonl"

    if scenario_source == "llm":
        if not llm_config:
            raise ValueError("llm_config is required for scenario_source='llm'.")
        scenarios = generate_checkpointed_llm_scenarios(
            count=count,
            config=llm_config,
            seed=seed,
            batch_size=llm_batch_size,
            checkpoint_path=scenario_path,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    elif scenario_source == "hybrid":
        if not llm_config:
            raise ValueError("llm_config is required for scenario_source='hybrid'.")
        llm_count = int(count * 0.7)
        rule_count = count - llm_count
        scenarios = generate_checkpointed_llm_scenarios(
            count=llm_count,
            config=llm_config,
            seed=seed,
            batch_size=llm_batch_size,
            checkpoint_path=scenario_path,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
        scenarios.extend(generate_scenarios(rule_count, seed=seed + 99))
        scenarios = _renumber_scenarios(scenarios)
        _write_jsonl(scenario_path, [scenario.to_dict() for scenario in scenarios])
    elif scenario_source == "rules":
        scenarios = generate_balanced_rule_scenarios(count=count, seed=seed)
        _write_jsonl(scenario_path, [scenario.to_dict() for scenario in scenarios])
    else:
        raise ValueError(f"Unknown scenario_source: {scenario_source}")

    index = InsuranceRAGIndex.from_chunks_file(chunks_path)
    cases = build_benchmark_cases(scenarios, index)
    predictions = generate_fast_oracle_traces(cases)
    sft_rows = export_sft_rows(cases, predictions)
    dpo_rows = export_dpo_rows(cases, predictions)
    rlvr_rows = export_rlvr_rows(cases, predictions)
    splits = split_rows_by_message(cases, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    report = quality_report(
        scenarios,
        cases,
        predictions,
        sft_rows,
        dpo_rows,
        rlvr_rows,
        splits=splits,
        heldout_paths=heldout_paths,
    )
    report["scenario_source"] = scenario_source
    report["splits"] = {name: len(rows) for name, rows in splits.items()}
    assert_quality(report, allow_warnings=allow_warnings)

    paths = {
        "scenarios": scenario_path,
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
    return {"paths": {key: str(value) for key, value in paths.items()}, "report": report}


def generate_balanced_rule_scenarios(count: int, seed: int) -> list[ScenarioCard]:
    rng = random.Random(seed)
    languages = list(LANGUAGES)
    city_pool = [
        ("Mumbai", "metro", "West"),
        ("Delhi", "metro", "North"),
        ("Bengaluru", "metro", "South"),
        ("Chennai", "metro", "South"),
        ("Hyderabad", "metro", "South"),
        ("Pune", "tier1", "West"),
        ("Lucknow", "tier2", "North"),
        ("Jaipur", "tier2", "North"),
        ("Ranchi", "tier2", "East"),
        ("Guwahati", "tier2", "North East"),
        ("Coimbatore", "tier2", "South"),
        ("Nashik", "tier2", "West"),
    ]
    occupations = [
        "salaried private employee",
        "government teacher",
        "self-employed shop owner",
        "gig delivery worker",
        "farmer",
        "software engineer",
        "small manufacturer",
        "retired parent",
        "student",
        "doctor",
    ]
    scenarios: list[ScenarioCard] = []
    for idx in range(count):
        category, task, need = BALANCED_RULE_SCENARIOS[idx % len(BALANCED_RULE_SCENARIOS)]
        language = languages[(idx + rng.randrange(len(languages))) % len(languages)]
        city, tier, region = city_pool[(idx * 5 + rng.randrange(len(city_pool))) % len(city_pool)]
        age = rng.choice([22, 27, 31, 36, 42, 48, 55, 62, 69])
        family_members = rng.choice([1, 2, 3, 4, 5, 6])
        dependents = max(0, min(family_members - 1, rng.choice([0, 1, 2, 3, 4])))
        budget = rng.choice(["low", "medium", "high"])
        existing_cover = rng.choice(["none", "employer health cover", "family floater exists", "old low sum insured policy"])
        occupation = occupations[(idx + rng.randrange(len(occupations))) % len(occupations)]
        profile = {
            "language": language,
            "age": age,
            "city": city,
            "city_tier": tier,
            "region": region,
            "family_members": family_members,
            "dependents": dependents,
            "occupation": occupation,
            "income_band_inr": rng.choice(["3-5L", "5-8L", "8-12L", "12-18L", "18-30L", "30L+"]),
            "budget_band": budget,
            "existing_cover": existing_cover,
            "primary_need": need,
        }
        expected = [category]
        avoid = [item for item in CATEGORIES if item != category][:2]
        missing: list[str] = []
        risk_tags = [need.replace(" ", "_")[:60], category]
        if task == "compare_suitable_categories":
            second = "personal_accident" if category != "personal_accident" else "health"
            expected = [category, second]
            risk_tags.append("comparison")
        if task == "reject_premature_recommendation":
            expected = []
            avoid = list(CATEGORIES)
            missing = ["age", "city", "family_members", "dependents", "existing_cover", "budget_band", "primary_need"]
            risk_tags.append("premature_recommendation")
        user_message = _balanced_message(idx, language, profile, category, task)
        scenarios.append(
            ScenarioCard(
                scenario_id=f"sdg_{idx:06d}",
                scenario_type=f"balanced_{category}_{task}",
                language=language,
                user_message=user_message,
                profile=profile,
                expected_categories=expected,
                avoid_categories=avoid,
                missing_slots=missing,
                risk_tags=sorted(set(risk_tags)),
                task=task,
            )
        )
    return scenarios


def _balanced_message(idx: int, language: str, profile: dict[str, Any], category: str, task: str) -> str:
    detail = (
        f"ref {idx:06d}, age {profile['age']}, {profile['occupation']}, "
        f"{profile['city']} {profile['city_tier']}, family {profile['family_members']}, "
        f"dependents {profile['dependents']}, existing cover {profile['existing_cover']}, budget {profile['budget_band']}"
    )
    if task == "reject_premature_recommendation":
        return f"I need one {category} policy and price now. I do not want to answer any questions or share more details. Context: {detail}."
    if task == "compare_suitable_categories":
        return f"Compare options for my situation: {detail}. I am confused about {profile['primary_need']} and want a practical priority."
    if language == "Hindi":
        return f"Mujhe {category} insurance chahiye. Details: {detail}. Mere liye suitable option kya hoga?"
    if language == "Hinglish":
        return f"Mujhe {category} insurance lena hai. Details: {detail}. Options confusing hain, please guide karo."
    return f"I need {category} insurance. My details are: {detail}. Recommend a suitable next step without inventing premium."


def generate_checkpointed_llm_scenarios(
    count: int,
    config: VLLMConfig,
    seed: int,
    batch_size: int,
    checkpoint_path: str | Path,
    checkpoint_every: int = 25,
    resume: bool = True,
) -> list[ScenarioCard]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = _read_scenarios(checkpoint_path) if resume and checkpoint_path.exists() else []
    scenarios = scenarios[:count]
    max_attempts = max(20, count * 3)
    attempt = 0
    rng = random.Random(seed)
    while len(scenarios) < count and attempt < max_attempts:
        attempt += 1
        needed = min(batch_size, count - len(scenarios))
        target = _next_targets(scenarios, rng)
        raw = call_vllm_messages(_scale_prompt(needed, seed, attempt, scenarios, target), config)
        parsed = _extract_json(raw)
        rows = parsed.get("scenarios", parsed if isinstance(parsed, list) else []) if isinstance(parsed, (dict, list)) else []
        accepted = 0
        for row in rows:
            row = _force_training_targets(row, target)
            card = _normalize_llm_scenario(row, f"sdg_{len(scenarios):06d}")
            if card and not _is_duplicate_scenario(card, scenarios):
                scenarios.append(card)
                accepted += 1
            if len(scenarios) >= count:
                break
        if accepted and (len(scenarios) % checkpoint_every == 0 or len(scenarios) >= count):
            _write_jsonl(checkpoint_path, [scenario.to_dict() for scenario in scenarios])
        print(
            f"sdg_progress={len(scenarios)}/{count} attempt={attempt} accepted={accepted} "
            f"target_category={target['category']} target_language={target['language']} target_task={target['task']}",
            flush=True,
        )
    if len(scenarios) < count:
        _write_jsonl(checkpoint_path, [scenario.to_dict() for scenario in scenarios])
        raise RuntimeError(f"LLM generated only {len(scenarios)}/{count} valid scenarios after {attempt} attempts.")
    scenarios = _renumber_scenarios(scenarios)
    _write_jsonl(checkpoint_path, [scenario.to_dict() for scenario in scenarios])
    return scenarios


def _scale_prompt(
    count: int,
    seed: int,
    attempt: int,
    existing: list[ScenarioCard],
    target: dict[str, str],
) -> list[dict[str, str]]:
    recent = [
        {
            "language": item.language,
            "scenario_type": item.scenario_type,
            "user_message": item.user_message[:200],
            "expected_categories": item.expected_categories,
        }
        for item in existing[-12:]
    ]
    schema = {
        "scenarios": [
            {
                "scenario_type": "specific_training_scenario_name",
                "language": target["language"],
                "user_message": "natural customer message",
                "profile": {
                    "language": target["language"],
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
                    "primary_need": "specific insurance need",
                },
                "expected_categories": [target["category"]],
                "avoid_categories": [],
                "missing_slots": [],
                "risk_tags": [target["variant"]],
                "task": target["task"],
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "You generate diverse SFT/DPO training scenarios for an Indian bank insurance advisor. "
                "Return JSON only. Make each user message realistic, varied, and not benchmark-like."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate {count} unique synthetic TRAINING scenarios. Seed={seed}, attempt={attempt}.\n"
                f"Target category: {target['category']}\n"
                f"Target language: {target['language']}\n"
                f"Target task: {target['task']}\n"
                f"Target variation: {target['variant']}\n"
                "Allowed categories: health, motor, travel, cyber, home, personal_accident.\n"
                "Use Indian customer details: occupation, city/tier, income band, dependents, existing cover, budget.\n"
                "Include natural Hindi/Hinglish/regional language when requested. Code-mix is allowed.\n"
                "Do not quote exact premiums. Do not copy any prior examples.\n"
                "For reject_premature_recommendation, the user must explicitly demand a recommendation/price without answering questions.\n"
                "For normal recommendation, expected_categories must include the target category.\n"
                "For comparison, include the target category plus one relevant second category.\n"
                f"Recent examples to avoid:\n{json.dumps(recent, ensure_ascii=False)}\n"
                f"Return exactly this JSON shape:\n{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def _next_targets(scenarios: list[ScenarioCard], rng: random.Random) -> dict[str, str]:
    category_counts = Counter(category for item in scenarios for category in item.expected_categories)
    language_counts = Counter(item.language for item in scenarios)
    task_counts = Counter(item.task for item in scenarios)
    variant_counts = Counter(item.risk_tags[0] if item.risk_tags else item.scenario_type for item in scenarios)
    category = min(CATEGORIES, key=lambda item: (category_counts.get(item, 0), CATEGORIES.index(item)))
    language = min(LANGUAGES, key=lambda item: (language_counts.get(item, 0), LANGUAGES.index(item)))
    task_targets = {
        "grounded_personalized_recommendation": 0.72,
        "compare_suitable_categories": 0.16,
        "reject_premature_recommendation": 0.12,
    }
    total = max(len(scenarios), 1)
    task = min(TASKS, key=lambda item: (task_counts.get(item, 0) / total - task_targets[item], TASKS.index(item)))
    variant = min(SCENARIO_VARIANTS, key=lambda item: (variant_counts.get(item, 0), SCENARIO_VARIANTS.index(item)))
    if rng.random() < 0.05:
        variant = rng.choice(SCENARIO_VARIANTS)
    return {"category": category, "language": language, "task": task, "variant": variant}


def _force_training_targets(row: Any, target: dict[str, str]) -> Any:
    if not isinstance(row, dict):
        return row
    row = dict(row)
    row["language"] = target["language"]
    row["task"] = target["task"]
    tags = row.get("risk_tags", [])
    if not isinstance(tags, list):
        tags = []
    tags.append(target["variant"])
    row["risk_tags"] = sorted({str(tag) for tag in tags if tag})
    if target["task"] == "reject_premature_recommendation":
        message = str(row.get("user_message", "")).strip()
        refusal_marker = " I do not want to answer any questions or share more details; give me one recommendation or price now."
        if not any(term in message.lower() for term in ("do not want to answer", "don't want to answer", "no questions", "questions nahi", "details nahi")):
            row["user_message"] = f"{message}{refusal_marker}".strip()
        row["expected_categories"] = []
        row["avoid_categories"] = list(CATEGORIES)
        row["missing_slots"] = ["age", "city", "family_members", "dependents", "existing_cover", "budget_band", "primary_need"]
        return row
    expected = row.get("expected_categories", [])
    if not isinstance(expected, list):
        expected = []
    if target["category"] not in expected:
        expected = [target["category"], *[str(item) for item in expected if str(item) in CATEGORIES]]
    if target["task"] == "compare_suitable_categories" and len(expected) == 1:
        second = next(category for category in CATEGORIES if category != target["category"])
        expected.append(second)
    row["expected_categories"] = expected[:2]
    return row


def _read_scenarios(path: Path) -> list[ScenarioCard]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                ScenarioCard(
                    scenario_id=row.get("scenario_id", f"sdg_{idx:06d}"),
                    scenario_type=row["scenario_type"],
                    language=row["language"],
                    user_message=row["user_message"],
                    profile=row["profile"],
                    expected_categories=row["expected_categories"],
                    avoid_categories=row.get("avoid_categories", []),
                    missing_slots=row.get("missing_slots", []),
                    risk_tags=row.get("risk_tags", []),
                    task=row["task"],
                )
            )
    return rows


def split_rows_by_message(
    cases: list[dict[str, Any]],
    seed: int,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        key = normalize_text(case["messages"][-1]["content"])
        groups.setdefault(key, []).append(case)
    group_items = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(group_items)
    train_target = len(cases) * train_ratio
    val_target = len(cases) * val_ratio
    splits = {"train": [], "val": [], "test": []}
    for group in group_items:
        if len(splits["train"]) + len(group) <= train_target:
            splits["train"].extend(group)
        elif len(splits["val"]) + len(group) <= val_target:
            splits["val"].extend(group)
        else:
            splits["test"].extend(group)
    return splits


def _write_splits(output: Path, prefix: str, rows: list[dict[str, Any]], splits: dict[str, list[dict[str, Any]]]) -> None:
    by_id = {row.get("case_id") or row.get("id", "").split(":")[0]: row for row in rows}
    for split_name, split_cases in splits.items():
        split_rows = [by_id[case["case_id"]] for case in split_cases]
        _write_jsonl(output / f"{prefix}_{split_name}.jsonl", split_rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
