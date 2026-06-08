"""Gymnasium-compatible RL environment for insurance advisor tool use."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - only used when gymnasium is absent.
    gym = None
    spaces = None

from insurance_agent.tool_eval import INSURANCE_TOOL_SCHEMAS
from insurance_agent.training import completion_reward, read_jsonl


INSURANCE_TOOL_NAMES = tuple(tool["name"] for tool in INSURANCE_TOOL_SCHEMAS)
FINISH_TOOL = "finish"


class InsuranceToolEnv(gym.Env if gym else object):
    """RL task where actions are insurance advisor tool choices.

    The environment samples RLVR rows and rewards a policy for selecting the
    correct safe tool sequence: ask/refuse, retrieve, rank, recommend, finish.
    It is intentionally tool-sequence focused; language generation is scored by
    the separate RLVR reward functions in :mod:`insurance_agent.training`.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        dataset_path: str | Path = "data/insurance/sdg_rules_50k_balanced/rlvr_train.jsonl",
        max_steps: int = 6,
        seed: int | None = None,
    ) -> None:
        self.rows = rows if rows is not None else read_jsonl(dataset_path)
        if not self.rows:
            raise ValueError("InsuranceToolEnv needs at least one RLVR row")
        self.max_steps = max_steps
        self.random = random.Random(seed)
        self.action_names = INSURANCE_TOOL_NAMES
        self.case: dict[str, Any] = self.rows[0]
        self.expected_calls: list[dict[str, Any]] = []
        self.predicted_calls: list[dict[str, Any]] = []
        self.step_count = 0
        self.done = False
        self.last_event = "reset"

        if spaces:
            self.action_space = spaces.Discrete(len(self.action_names))
            self.observation_space = spaces.Dict(
                {
                    "expected_length": spaces.Discrete(max_steps + 1),
                    "step_index": spaces.Discrete(max_steps + 1),
                    "last_action": spaces.Discrete(len(self.action_names) + 1),
                    "is_refusal_case": spaces.Discrete(2),
                    "has_retrieved": spaces.Discrete(2),
                    "has_recommended": spaces.Discrete(2),
                    "done": spaces.Discrete(2),
                }
            )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if gym:
            super().reset(seed=seed)
        if seed is not None:
            self.random.seed(seed)
        self.case = self.random.choice(self.rows)
        if options and "case_id" in options:
            self.case = next(row for row in self.rows if row.get("metadata", {}).get("case_id") == options["case_id"])
        self.expected_calls = list(self.case["reference"]["tool_calls"])
        self.predicted_calls = []
        self.step_count = 0
        self.done = False
        self.last_event = "reset"
        return self._obs(), self._info()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Cannot call step() after episode is done. Call reset() first.")

        tool_name = self.action_names[int(action)]
        expected_tool = self.expected_calls[self.step_count]["tool_name"] if self.step_count < len(self.expected_calls) else None
        self.step_count += 1
        self.predicted_calls.append({"turn": self.step_count, "role": "assistant", "tool_name": tool_name, "arguments": {}})

        reward = -0.05
        terminated = False
        truncated = False

        if tool_name == expected_tool:
            reward += 1.0
            self.last_event = f"matched:{tool_name}"
        else:
            reward -= 0.7
            self.last_event = f"mismatch:{tool_name}:expected:{expected_tool}"

        if tool_name == "generate_recommendation" and self._is_refusal_case():
            reward -= 2.0
            self.last_event = "unsafe_recommendation_on_refusal_case"

        if tool_name == FINISH_TOOL:
            terminated = True
            reward += self._terminal_reward()
        elif self.step_count >= min(self.max_steps, len(self.expected_calls)):
            truncated = True
            reward -= 1.0

        self.done = terminated or truncated
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        expected = [call["tool_name"] for call in self.expected_calls]
        predicted = [call["tool_name"] for call in self.predicted_calls]
        return f"case={self.case_id} expected={expected} predicted={predicted} event={self.last_event}"

    @property
    def case_id(self) -> str:
        return str(self.case.get("metadata", {}).get("case_id", self.case.get("id", "")))

    def _terminal_reward(self) -> float:
        completion = {"tool_calls": self.predicted_calls, "final_response": {}}
        return completion_reward(
            __import__("json").dumps(completion, ensure_ascii=False),
            self.case["reference"],
            self.case.get("reward_spec", {}),
        )

    def _is_refusal_case(self) -> bool:
        return not self.case["reference"]["expected_final"].get("expected_categories", [])

    def _obs(self) -> dict[str, int]:
        names = [call["tool_name"] for call in self.predicted_calls]
        last_action = self.action_names.index(names[-1]) if names else len(self.action_names)
        return {
            "expected_length": min(len(self.expected_calls), self.max_steps),
            "step_index": min(self.step_count, self.max_steps),
            "last_action": last_action,
            "is_refusal_case": int(self._is_refusal_case()),
            "has_retrieved": int("retrieve_policy_evidence" in names),
            "has_recommended": int("generate_recommendation" in names),
            "done": int(self.done),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "last_event": self.last_event,
            "expected_tools": [call["tool_name"] for call in self.expected_calls],
            "predicted_tools": [call["tool_name"] for call in self.predicted_calls],
        }
