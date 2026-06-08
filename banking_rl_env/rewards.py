"""Reward accounting for the banking tool-use environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardBreakdown:
    components: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.components[name] = self.components.get(name, 0.0) + value

    @property
    def total(self) -> float:
        return round(sum(self.components.values()), 4)

    def as_dict(self) -> dict[str, float]:
        return dict(self.components)
