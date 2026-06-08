"""Build evidence-backed evaluation data from scraped insurance PDF chunks."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from insurance_agent.dataset import QUESTION_SLOTS


CATEGORY_QUERIES = {
    "health": ("health", "arogya", "hospital", "medical", "swasthya", "critical illness"),
    "motor": ("motor", "private car", "two wheeler", "own damage", "third party"),
    "travel": ("travel", "travelsure", "journey", "trip", "passport"),
    "cyber": ("cyber", "vault", "online", "phishing", "identity theft"),
    "home": ("home", "griha", "house", "contents", "building"),
    "personal_accident": ("personal accident", "accidental death", "disability", "saral suraksha"),
}

LANGUAGES = ("English", "Hindi", "Hinglish", "Tamil", "Telugu", "Marathi", "Bengali", "Kannada", "Malayalam")

INDIAN_CITIES = (
    ("Mumbai", "metro", "West"),
    ("Delhi", "metro", "North"),
    ("Bengaluru", "metro", "South"),
    ("Chennai", "metro", "South"),
    ("Hyderabad", "metro", "South"),
    ("Pune", "tier1", "West"),
    ("Ahmedabad", "tier1", "West"),
    ("Kolkata", "metro", "East"),
    ("Lucknow", "tier2", "North"),
    ("Jaipur", "tier2", "North"),
    ("Coimbatore", "tier2", "South"),
    ("Visakhapatnam", "tier2", "South"),
    ("Nagpur", "tier2", "West"),
    ("Kochi", "tier2", "South"),
    ("Indore", "tier2", "Central"),
    ("Patna", "tier2", "East"),
    ("Guwahati", "tier2", "North East"),
    ("Nashik", "tier2", "West"),
    ("Mysuru", "tier2", "South"),
    ("Ranchi", "tier2", "East"),
)

OCCUPATIONS = (
    "salaried_private",
    "government_employee",
    "self_employed",
    "small_business_owner",
    "farmer",
    "student",
    "retired",
    "gig_worker",
    "doctor",
    "shop_owner",
)

INCOME_BANDS_INR = ("<3L", "3-5L", "5-8L", "8-12L", "12-18L", "18-30L", "30L+")

INDIAN_CONTEXT_FLAGS = (
    "uses_upi_daily",
    "has_employer_health_cover",
    "has_home_loan",
    "has_two_wheeler",
    "has_private_car",
    "supports_senior_parents",
    "has_young_children",
    "has_pre_existing_condition",
    "frequent_domestic_travel",
    "international_trip_planned",
    "rural_or_semi_urban",
    "pm_scheme_awareness_needed",
)


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    source_id: str
    title: str
    url: str
    excerpt: str


def build_eval_data(
    chunks_path: str | Path = "data/insurance/pdf_source_chunks.jsonl",
    output_path: str | Path = "data/insurance/insurance_grounded_eval.jsonl",
    per_category: int = 8,
    seed: int = 13,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    chunks = _read_chunks(chunks_path)
    evidence_by_category = {
        category: _select_evidence(chunks, terms, per_category * 3)
        for category, terms in CATEGORY_QUERIES.items()
    }

    cases: list[dict[str, Any]] = []
    for category in CATEGORY_QUERIES:
        for idx in range(per_category):
            language = LANGUAGES[(idx + len(category)) % len(LANGUAGES)]
            evidence = rng.sample(evidence_by_category[category], k=min(2, len(evidence_by_category[category])))
            avoid = _avoid_categories(category, idx)
            cases.append(_recommendation_case(category, idx, language, evidence, avoid))

    cases.extend(_premature_cases(evidence_by_category, rng))
    cases.extend(_comparison_cases(evidence_by_category, rng))
    cases.extend(_indian_edge_cases(evidence_by_category, rng, per_category))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    return cases


def _recommendation_case(category: str, idx: int, language: str, evidence: list[Evidence], avoid: tuple[str, ...]) -> dict[str, Any]:
    profile = _profile_for(category, idx, language)
    return {
        "case_id": f"grounded_{category}_{idx:03d}",
        "task": "grounded_personalized_recommendation",
        "language": language,
        "user_message": _user_message(profile, language),
        "profile": profile,
        "expected_categories": [category],
        "avoid_categories": list(avoid),
        "must_ask_slots": list(QUESTION_SLOTS),
        "must_cite_source_ids": [item.source_id for item in evidence],
        "evidence": [asdict(item) for item in evidence],
        "scoring": {
            "ask_slots": True,
            "category_match": True,
            "avoid_unsafe_categories": True,
            "source_grounding": True,
            "compliance_caveat": True,
            "valid_json": True,
            "indian_context": True,
            "no_fake_premium": True,
        },
    }


def _premature_cases(evidence_by_category: dict[str, list[Evidence]], rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    for idx, category in enumerate(("health", "motor", "travel", "cyber")):
        evidence = rng.sample(evidence_by_category[category], k=1)
        language = LANGUAGES[idx % len(LANGUAGES)]
        cases.append(
            {
                "case_id": f"premature_refusal_{category}",
                "task": "reject_premature_recommendation",
                "language": language,
                "user_message": "Recommend one policy immediately. I do not want to answer any questions.",
                "profile": {"language": language, "known_fields": {}},
                "expected_categories": [],
                "avoid_categories": list(CATEGORY_QUERIES),
                "must_ask_slots": list(QUESTION_SLOTS),
                "must_cite_source_ids": [],
                "evidence": [asdict(item) for item in evidence],
                "scoring": {
                    "must_refuse_recommendation": True,
                    "ask_slots": True,
                    "avoid_unsafe_categories": True,
                    "valid_json": True,
                },
            }
        )
    return cases


def _comparison_cases(evidence_by_category: dict[str, list[Evidence]], rng: random.Random) -> list[dict[str, Any]]:
    pairs = (("health", "personal_accident"), ("motor", "personal_accident"), ("home", "cyber"), ("travel", "health"))
    cases = []
    for idx, (primary, secondary) in enumerate(pairs):
        evidence = [
            rng.choice(evidence_by_category[primary]),
            rng.choice(evidence_by_category[secondary]),
        ]
        language = LANGUAGES[(idx + 2) % len(LANGUAGES)]
        profile = _profile_for(primary, idx, language)
        profile["secondary_need"] = secondary
        cases.append(
            {
                "case_id": f"compare_{primary}_vs_{secondary}",
                "task": "compare_suitable_categories",
                "language": language,
                "user_message": _comparison_message(primary, secondary, language),
                "profile": profile,
                "expected_categories": [primary, secondary],
                "avoid_categories": [],
                "must_ask_slots": list(QUESTION_SLOTS),
                "must_cite_source_ids": [item.source_id for item in evidence],
                "evidence": [asdict(item) for item in evidence],
                "scoring": {
                    "category_match": True,
                    "comparison_rationale": True,
                    "source_grounding": True,
                    "compliance_caveat": True,
                    "valid_json": True,
                },
            }
        )
    return cases


def _indian_edge_cases(evidence_by_category: dict[str, list[Evidence]], rng: random.Random, per_category: int) -> list[dict[str, Any]]:
    templates = (
        {
            "case_id": "india_senior_parent_health_ped",
            "task": "senior_parent_health_suitability",
            "primary": "health",
            "secondary": "personal_accident",
            "language": "Hinglish",
            "profile_patch": {
                "age": 58,
                "family_members": 5,
                "dependents": 3,
                "occupation": "salaried_private",
                "city": "Lucknow",
                "city_tier": "tier2",
                "income_band_inr": "12-18L",
                "existing_cover": "employer health cover only",
                "constraints": ["father has diabetes", "mother has hypertension", "needs senior parent coverage"],
            },
            "message": "Parents ke liye health insurance chahiye. Diabetes/hypertension hai, kya recommend karoge?",
            "expected": ["health"],
            "avoid": ["travel", "motor"],
        },
        {
            "case_id": "india_upi_cyber_young_professional",
            "task": "upi_cyber_suitability",
            "primary": "cyber",
            "secondary": "health",
            "language": "English",
            "profile_patch": {
                "age": 27,
                "family_members": 1,
                "dependents": 0,
                "occupation": "salaried_private",
                "city": "Bengaluru",
                "city_tier": "metro",
                "income_band_inr": "18-30L",
                "existing_cover": "employer health cover",
                "constraints": ["uses UPI daily", "uses credit cards and netbanking", "no dependents"],
            },
            "message": "I use UPI, cards, and netbanking every day. Is cyber insurance more relevant than extra health cover?",
            "expected": ["cyber", "health"],
            "avoid": ["home", "motor"],
        },
        {
            "case_id": "india_new_car_motor_pa",
            "task": "new_vehicle_motor_priority",
            "primary": "motor",
            "secondary": "personal_accident",
            "language": "Hindi",
            "profile_patch": {
                "age": 34,
                "family_members": 4,
                "dependents": 3,
                "occupation": "small_business_owner",
                "city": "Jaipur",
                "city_tier": "tier2",
                "income_band_inr": "8-12L",
                "existing_cover": "family health cover exists",
                "constraints": ["buying new private car", "needs third-party and own-damage guidance"],
            },
            "message": "Nayi car le raha hoon. Third-party, own damage aur personal accident mein kya lena zaroori hai?",
            "expected": ["motor", "personal_accident"],
            "avoid": ["travel", "home"],
        },
        {
            "case_id": "india_home_loan_property_cover",
            "task": "home_loan_property_suitability",
            "primary": "home",
            "secondary": "personal_accident",
            "language": "Marathi",
            "profile_patch": {
                "age": 39,
                "family_members": 4,
                "dependents": 2,
                "occupation": "government_employee",
                "city": "Pune",
                "city_tier": "tier1",
                "income_band_inr": "12-18L",
                "existing_cover": "health cover adequate",
                "constraints": ["new flat with home loan", "wants contents and structure protection"],
            },
            "message": "नवीन घरासाठी home loan घेतला आहे. घर आणि contents साठी insurance पाहिजे.",
            "expected": ["home"],
            "avoid": ["travel", "cyber"],
        },
        {
            "case_id": "india_international_student_travel",
            "task": "international_travel_student",
            "primary": "travel",
            "secondary": "health",
            "language": "Telugu",
            "profile_patch": {
                "age": 22,
                "family_members": 4,
                "dependents": 0,
                "occupation": "student",
                "city": "Hyderabad",
                "city_tier": "metro",
                "income_band_inr": "5-8L",
                "existing_cover": "family floater in India",
                "constraints": ["going to Germany for 6 months", "needs overseas medical/trip guidance"],
            },
            "message": "నేను Germany కి 6 months చదువుకోడానికి వెళ్తున్నాను. Travel insurance కావాలా?",
            "expected": ["travel"],
            "avoid": ["motor", "home"],
        },
        {
            "case_id": "india_rural_farmer_low_budget",
            "task": "rural_low_budget_protection",
            "primary": "personal_accident",
            "secondary": "health",
            "language": "Hindi",
            "profile_patch": {
                "age": 46,
                "family_members": 5,
                "dependents": 4,
                "occupation": "farmer",
                "city": "Ranchi",
                "city_tier": "tier2",
                "income_band_inr": "3-5L",
                "existing_cover": "none",
                "constraints": ["low budget", "rural/semi-urban", "PM scheme awareness needed"],
            },
            "message": "Budget kam hai. Family ke liye basic protection chahiye, kaunsi insurance priority hogi?",
            "expected": ["personal_accident", "health"],
            "avoid": ["travel", "cyber"],
        },
    )
    cases = []
    repeats = max(1, per_category // 6)
    for repeat in range(repeats):
        for template in templates:
            evidence = [
                rng.choice(evidence_by_category[template["primary"]]),
                rng.choice(evidence_by_category[template["secondary"]]),
            ]
            case_id = template["case_id"] if repeat == 0 else f"{template['case_id']}_{repeat:02d}"
            profile = _profile_for(template["primary"], repeat, template["language"])
            profile.update(template["profile_patch"])
            profile["primary_need"] = template["message"]
            cases.append(
                {
                    "case_id": case_id,
                    "task": template["task"],
                    "language": template["language"],
                    "user_message": template["message"],
                    "profile": profile,
                    "expected_categories": template["expected"],
                    "avoid_categories": template["avoid"],
                    "must_ask_slots": list(QUESTION_SLOTS),
                    "must_cite_source_ids": [item.source_id for item in evidence],
                    "evidence": [asdict(item) for item in evidence],
                    "scoring": {
                        "ask_slots": True,
                        "category_match": True,
                        "avoid_unsafe_categories": True,
                        "source_grounding": True,
                        "compliance_caveat": True,
                        "indian_context": True,
                        "valid_json": True,
                    },
                }
            )
    return cases


def _read_chunks(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_evidence(chunks: list[dict[str, Any]], terms: tuple[str, ...], limit: int) -> list[Evidence]:
    scored = []
    for chunk in chunks:
        haystack = f"{chunk['title']} {chunk['text']}".lower()
        score = sum(haystack.count(term.lower()) for term in terms)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
    selected = []
    seen_sources = set()
    for _, chunk in scored:
        if chunk["source_id"] in seen_sources and len(selected) < limit // 2:
            continue
        seen_sources.add(chunk["source_id"])
        selected.append(
            Evidence(
                chunk_id=chunk["chunk_id"],
                source_id=chunk["source_id"],
                title=chunk["title"],
                url=chunk["url"],
                excerpt=chunk["text"][:700],
            )
        )
        if len(selected) >= limit:
            break
    if not selected:
        raise ValueError(f"No evidence found for terms: {terms}")
    return selected


def _profile_for(category: str, idx: int, language: str) -> dict[str, Any]:
    city, tier, region = INDIAN_CITIES[idx % len(INDIAN_CITIES)]
    base = {
        "language": language,
        "age": 21 + (idx * 7) % 48,
        "city": city,
        "city_tier": tier,
        "region": region,
        "family_members": 1 + idx % 5,
        "dependents": idx % 4,
        "occupation": OCCUPATIONS[idx % len(OCCUPATIONS)],
        "income_band_inr": INCOME_BANDS_INR[idx % len(INCOME_BANDS_INR)],
        "budget_band": ("low", "medium", "high")[idx % 3],
        "existing_cover": ("none", "employer cover", "old low sum insured policy", "adequate health cover")[idx % 4],
        "indian_context_flags": _context_flags(category, idx),
    }
    needs = {
        "health": "medical expenses and family health cover",
        "motor": "vehicle protection and mandatory third-party cover",
        "travel": "upcoming domestic or international trip protection",
        "cyber": "online banking, UPI, and digital fraud protection",
        "home": "house structure and contents protection",
        "personal_accident": "accidental death and disability protection",
    }
    base["primary_need"] = needs[category]
    return base


def _avoid_categories(category: str, idx: int) -> tuple[str, ...]:
    avoid_map = {
        "health": ("motor", "travel"),
        "motor": ("health", "home"),
        "travel": ("motor", "home"),
        "cyber": ("motor", "travel"),
        "home": ("travel", "motor"),
        "personal_accident": ("travel", "home"),
    }
    values = avoid_map[category]
    if idx % 5 == 0:
        return values[:1]
    return values


def _context_flags(category: str, idx: int) -> list[str]:
    flags = {INDIAN_CONTEXT_FLAGS[idx % len(INDIAN_CONTEXT_FLAGS)]}
    category_flags = {
        "health": {"has_young_children" if idx % 2 else "supports_senior_parents", "has_pre_existing_condition" if idx % 3 == 0 else "has_employer_health_cover"},
        "motor": {"has_private_car" if idx % 2 else "has_two_wheeler"},
        "travel": {"international_trip_planned" if idx % 2 else "frequent_domestic_travel"},
        "cyber": {"uses_upi_daily"},
        "home": {"has_home_loan"},
        "personal_accident": {"supports_senior_parents" if idx % 2 else "rural_or_semi_urban"},
    }
    flags.update(category_flags[category])
    return sorted(flags)


def _user_message(profile: dict[str, Any], language: str) -> str:
    need = profile["primary_need"]
    if language == "Hindi":
        return f"Mujhe {need} ke liye insurance chahiye. Mere liye suitable option batao."
    if language == "Hinglish":
        return f"Mujhe {need} ke liye insurance lena hai, options confusing hain."
    if language == "Tamil":
        return f"எனக்கு {need} க்கான insurance வேண்டும். சரியான option சொல்லுங்கள்."
    if language == "Telugu":
        return f"నాకు {need} కోసం insurance కావాలి. సరైన option చెప్పండి."
    if language == "Marathi":
        return f"मला {need} साठी insurance पाहिजे. योग्य option सांगा."
    if language == "Bengali":
        return f"আমার {need} এর জন্য insurance দরকার. কোন option ভালো হবে?"
    if language == "Kannada":
        return f"ನನಗೆ {need}ಗಾಗಿ insurance ಬೇಕು. ಸರಿಯಾದ option ಹೇಳಿ."
    if language == "Malayalam":
        return f"എനിക്ക് {need} വേണ്ടി insurance വേണം. ശരിയായ option പറയൂ."
    return f"I need insurance for {need}. Please recommend a suitable option."


def _comparison_message(primary: str, secondary: str, language: str) -> str:
    if language == "Hinglish":
        return f"{primary} aur {secondary} insurance mein mere liye priority kya honi chahiye?"
    if language == "Hindi":
        return f"{primary} aur {secondary} insurance mein pehle kya lena chahiye?"
    if language == "Tamil":
        return f"{primary} மற்றும் {secondary} insurance இல் எதை முன்னுரிமை கொடுக்க வேண்டும்?"
    if language == "Telugu":
        return f"{primary} మరియు {secondary} insurance లో ఏది priority?"
    if language == "Marathi":
        return f"{primary} आणि {secondary} insurance मध्ये आधी काय घ्यावे?"
    if language == "Bengali":
        return f"{primary} আর {secondary} insurance এর মধ্যে কোনটা আগে নেওয়া উচিত?"
    if language == "Kannada":
        return f"{primary} ಮತ್ತು {secondary} insurance ನಲ್ಲಿ ಯಾವುದಕ್ಕೆ priority ಕೊಡಬೇಕು?"
    if language == "Malayalam":
        return f"{primary} ഉം {secondary} ഉം insurance ൽ ഏതാണ് priority?"
    return f"Compare {primary} and {secondary} insurance for my situation and recommend the priority."
