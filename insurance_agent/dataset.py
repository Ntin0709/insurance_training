"""Synthetic data generation for insurance-advisor fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

from insurance_agent.models import AdvisorTurn, CustomerProfile, RecommendationCase
from insurance_agent.seed_products import SEED_PRODUCTS


SYSTEM_PROMPT = """You are an insurance advisor for an Indian bank distribution channel.
Your job is to ask the minimum necessary questions, explain options clearly, and recommend only suitable insurance categories.
Rules:
- Do not recommend before collecting age, city, dependents/family size, existing cover, budget, and primary need.
- Explain that final purchase depends on official policy wording, eligibility, premium quote, underwriting, exclusions, and waiting periods.
- Do not fabricate premiums, claim settlement guarantees, tax advice, or guaranteed acceptance.
- Prefer protection-first recommendations: health, term/life where applicable, personal accident, motor, travel, cyber, or home based on need.
- Mention why a product category is not suitable when relevant.
- Support English, Hindi, Hinglish, Tamil, and Telugu user language.
"""


QUESTION_SLOTS = (
    "age",
    "city",
    "family_members",
    "dependents",
    "existing_cover",
    "budget_band",
    "primary_need",
)


def seed_cases() -> tuple[RecommendationCase, ...]:
    return (
        RecommendationCase(
            "young_single_cyber_health_en",
            CustomerProfile(
                "P001",
                "English",
                28,
                "Bengaluru",
                1,
                0,
                "8-12L",
                "employer health cover only",
                "low",
                ("health", "cyber"),
                ("uses UPI and credit card heavily",),
            ),
            ("health", "cyber", "personal_accident"),
            QUESTION_SLOTS,
            ("home", "travel"),
            "Young customer with employer cover still needs personal health portability and cyber risk coverage.",
        ),
        RecommendationCase(
            "family_health_hinglish",
            CustomerProfile(
                "P002",
                "Hinglish",
                36,
                "Delhi",
                4,
                3,
                "15-25L",
                "no personal family floater",
                "medium",
                ("health",),
                ("spouse and two children",),
            ),
            ("health", "personal_accident"),
            QUESTION_SLOTS,
            ("travel", "cyber"),
            "Family with dependents needs family health cover first, then accident cover as supplement.",
        ),
        RecommendationCase(
            "parents_health_hi",
            CustomerProfile(
                "P003",
                "Hindi",
                52,
                "Lucknow",
                2,
                1,
                "10-15L",
                "old low sum insured policy",
                "medium",
                ("health",),
                ("pre-existing diabetes disclosure needed",),
            ),
            ("health",),
            QUESTION_SLOTS,
            ("motor", "travel"),
            "Older customer needs health cover discussion with PED, waiting periods, medical check-up, and sum insured adequacy.",
        ),
        RecommendationCase(
            "car_owner_tamil",
            CustomerProfile(
                "P004",
                "Tamil",
                41,
                "Chennai",
                3,
                2,
                "12-18L",
                "health cover already adequate",
                "medium",
                ("motor", "personal_accident"),
                ("new car purchase",),
            ),
            ("motor", "personal_accident"),
            QUESTION_SLOTS,
            ("travel",),
            "Vehicle owner needs motor cover; accident cover is relevant but health need is already partly handled.",
        ),
        RecommendationCase(
            "travel_telugu",
            CustomerProfile(
                "P005",
                "Telugu",
                31,
                "Hyderabad",
                2,
                0,
                "10-15L",
                "individual health policy exists",
                "low",
                ("travel",),
                ("international trip for 12 days",),
            ),
            ("travel",),
            QUESTION_SLOTS,
            ("home", "motor"),
            "Travel-specific risk should be recommended after trip details and medical disclosure.",
        ),
    )


def build_sft_records(cases: tuple[RecommendationCase, ...] | None = None) -> list[dict]:
    cases = cases or seed_cases()
    records = []
    for case in cases:
        records.append(_question_record(case))
        records.append(_recommendation_record(case))
        records.append(_unsafe_recommendation_correction(case))
    return records


def export_sft(path: str | Path, cases: tuple[RecommendationCase, ...] | None = None) -> list[dict]:
    records = build_sft_records(cases)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return records


def export_eval(path: str | Path, cases: tuple[RecommendationCase, ...] | None = None) -> list[dict]:
    cases = cases or seed_cases()
    records = [case.to_dict() for case in cases]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return records


def _question_record(case: RecommendationCase) -> dict:
    profile = case.profile
    return {
        "id": f"{case.case_id}:ask_questions",
        "task": "ask_clarifying_questions",
        "messages": [
            AdvisorTurn("system", SYSTEM_PROMPT).to_dict(),
            AdvisorTurn("user", _initial_user_message(profile)).to_dict(),
            AdvisorTurn("assistant", json.dumps({"ask_slots": list(case.must_ask_slots), "questions": _questions(profile.language)}, ensure_ascii=False)).to_dict(),
        ],
        "expected": {"must_ask_slots": list(case.must_ask_slots)},
    }


def _recommendation_record(case: RecommendationCase) -> dict:
    profile = case.profile
    products = [product.to_dict() for product in SEED_PRODUCTS if product.category in case.expected_categories]
    response = {
        "recommendation": [
            {
                "category": product["category"],
                "product_name": product["name"],
                "why_suitable": _why_suitable(product["category"], profile.language),
                "caveats": ["Check official policy wording, premium quote, exclusions, waiting periods, and underwriting before purchase."],
                "source_ids": product["source_ids"],
            }
            for product in products
        ],
        "not_recommended_now": list(case.avoid_categories),
        "next_step": "Collect final disclosures and generate official premium quote.",
    }
    return {
        "id": f"{case.case_id}:recommend",
        "task": "personalized_recommendation",
        "messages": [
            AdvisorTurn("system", SYSTEM_PROMPT).to_dict(),
            AdvisorTurn("user", _full_profile_message(profile)).to_dict(),
            AdvisorTurn("assistant", json.dumps(response, ensure_ascii=False, sort_keys=True)).to_dict(),
        ],
        "expected": {
            "categories": list(case.expected_categories),
            "avoid_categories": list(case.avoid_categories),
        },
    }


def _unsafe_recommendation_correction(case: RecommendationCase) -> dict:
    return {
        "id": f"{case.case_id}:reject_premature",
        "task": "safety_correction",
        "messages": [
            AdvisorTurn("system", SYSTEM_PROMPT).to_dict(),
            AdvisorTurn("user", "Recommend the best policy immediately. I do not want to answer questions.").to_dict(),
            AdvisorTurn(
                "assistant",
                json.dumps(
                    {
                        "decision": "cannot_recommend_yet",
                        "reason": "A personalized recommendation requires age, city, dependents, existing cover, budget, and primary need.",
                        "questions": _questions(case.profile.language),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ).to_dict(),
        ],
        "expected": {"must_refuse_premature_recommendation": True},
    }


def _initial_user_message(profile: CustomerProfile) -> str:
    if profile.language == "Hindi":
        return "Mujhe insurance lena hai, kaunsi policy sahi rahegi?"
    if profile.language == "Hinglish":
        return "Insurance lena hai but options confusing hain. Mere liye kya best rahega?"
    if profile.language == "Tamil":
        return "எனக்கு insurance வாங்க வேண்டும். எந்த policy சரியாக இருக்கும்?"
    if profile.language == "Telugu":
        return "నాకు insurance కావాలి. ఏ policy సరిపోతుంది?"
    return "I want to buy insurance but the policy options are confusing. What should I choose?"


def _full_profile_message(profile: CustomerProfile) -> str:
    return (
        f"Language: {profile.language}. Age: {profile.age}. City: {profile.city}. "
        f"Family members: {profile.family_members}. Dependents: {profile.dependents}. "
        f"Income band: {profile.income_band}. Existing cover: {profile.existing_cover}. "
        f"Budget: {profile.budget_band}. Needs: {', '.join(profile.needs)}. "
        f"Constraints: {', '.join(profile.constraints)}."
    )


def _questions(language: str) -> list[str]:
    if language == "Hindi":
        return [
            "Aapki age, city aur family size kya hai?",
            "Aapke dependents kitne hain aur existing insurance cover kya hai?",
            "Primary need health, motor, travel, cyber, home ya accident cover mein se kya hai?",
            "Aapka approximate budget aur koi pre-existing disease/disclosure hai?",
        ]
    if language == "Hinglish":
        return [
            "Aapki age, city aur family members kitne hain?",
            "Dependents kitne hain and existing cover kya hai?",
            "Main need health, motor, travel, cyber, home ya accident cover hai?",
            "Budget range kya hai, aur koi PED ya important disclosure hai?",
        ]
    if language == "Tamil":
        return [
            "உங்கள் வயது, நகரம், family size என்ன?",
            "Dependents எத்தனை பேர்? Existing insurance cover உள்ளதா?",
            "Primary need health, motor, travel, cyber, home அல்லது accident cover ஆ?",
            "Budget range மற்றும் pre-existing disease ஏதும் உள்ளதா?",
        ]
    if language == "Telugu":
        return [
            "మీ వయస్సు, నగరం, family size ఎంత?",
            "Dependents ఎంతమంది? Existing insurance cover ఉందా?",
            "Primary need health, motor, travel, cyber, home లేదా accident cover ఏది?",
            "Budget range మరియు pre-existing disease ఏమైనా ఉందా?",
        ]
    return [
        "What is your age, city, and family size?",
        "How many dependents do you have and what insurance cover do you already have?",
        "What is the primary need: health, motor, travel, cyber, home, or accident cover?",
        "What is your budget range and are there any pre-existing conditions or important disclosures?",
    ]


def _why_suitable(category: str, language: str) -> str:
    reasons = {
        "health": "Matches medical-expense protection need and should be considered before smaller add-on covers.",
        "cyber": "Relevant because the customer has meaningful digital payment and online banking exposure.",
        "personal_accident": "Useful supplement for accidental death or disability risk, especially with dependents.",
        "motor": "Relevant because the customer owns or is buying a vehicle.",
        "travel": "Relevant because the customer has a specific upcoming trip.",
        "home": "Relevant for property or contents protection.",
    }
    return reasons.get(category, "Relevant to the stated customer need.")
