"""Structured records for insurance advisor training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    url: str
    title: str
    document_type: str
    text: str
    retrieved_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InsuranceProduct:
    product_id: str
    name: str
    category: str
    provider: str
    suitable_for: tuple[str, ...]
    key_features: tuple[str, ...]
    exclusions_or_waiting: tuple[str, ...] = ()
    sum_insured_options: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CustomerProfile:
    profile_id: str
    language: str
    age: int
    city: str
    family_members: int
    dependents: int
    income_band: str
    existing_cover: str
    budget_band: str
    needs: tuple[str, ...]
    constraints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecommendationCase:
    case_id: str
    profile: CustomerProfile
    expected_categories: tuple[str, ...]
    must_ask_slots: tuple[str, ...]
    avoid_categories: tuple[str, ...] = ()
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "profile": self.profile.to_dict(),
            "expected_categories": self.expected_categories,
            "must_ask_slots": self.must_ask_slots,
            "avoid_categories": self.avoid_categories,
            "rationale": self.rationale,
        }


@dataclass
class AdvisorTurn:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
