"""Gymnasium-compatible RL environment for banking tool selection."""

from __future__ import annotations

import random
from typing import Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only when gymnasium is absent.
    gym = None
    spaces = None

from banking_rl_env.scenarios import SCENARIOS, Scenario
from banking_rl_env.rewards import RewardBreakdown
from banking_rl_env.simulator import BankingToolSimulator
from banking_rl_env.tools import ACTIONS, TOOL_NAMES, TOOLS, Action

ERROR_CODES = ("NONE", "AUTH_FAILED", "SESSION_EXPIRED", "ACCOUNT_LOCKED", "INSUFFICIENT_BALANCE", "VALIDATION_ERROR", "DUPLICATE_REQUEST", "SYSTEM_UNAVAILABLE")


class BankingToolEnv(gym.Env if gym else object):
    """RL task where actions are banking tool choices and control actions.

    Episodes sample one user goal. The agent must call authentication and
    confirmation control actions when required, satisfy prerequisite tools, call
    the target tool, then finish.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, scenarios: tuple[Scenario, ...] = SCENARIOS, max_steps: int = 8, seed: int | None = None):
        self.scenarios = scenarios
        self.max_steps = max_steps
        self.random = random.Random(seed)
        self.action_names = ACTIONS
        self.scenario: Scenario = scenarios[0]
        self.simulator = BankingToolSimulator()
        self.step_count = 0
        self.target_called = False
        self.done = False
        self.last_event = "reset"
        self.last_error_code = "NONE"
        self.trace: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None

        if spaces:
            self.action_space = spaces.Discrete(len(self.action_names))
            self.observation_space = spaces.Dict(
                {
                    "scenario": spaces.Discrete(len(self.scenarios)),
                    "target_tool": spaces.Discrete(len(TOOL_NAMES)),
                    "authenticated": spaces.Discrete(2),
                    "confirmed": spaces.Discrete(2),
                    "called_tools": spaces.MultiBinary(len(TOOL_NAMES)),
                    "required_prerequisites": spaces.MultiBinary(len(TOOL_NAMES)),
                    "satisfied_prerequisites": spaces.MultiBinary(len(TOOL_NAMES)),
                    "last_error_code": spaces.Discrete(len(ERROR_CODES)),
                    "steps_remaining": spaces.Discrete(max_steps + 1),
                }
            )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if gym:
            super().reset(seed=seed)
        if seed is not None:
            self.random.seed(seed)
        self.scenario = self.random.choice(self.scenarios)
        if options and "scenario_id" in options:
            self.scenario = next(s for s in self.scenarios if s.scenario_id == options["scenario_id"])
        self.simulator = BankingToolSimulator()
        self.step_count = 0
        self.target_called = False
        self.done = False
        self.last_event = "reset"
        self.last_error_code = "NONE"
        self.trace = []
        self.last_result = None
        return self._obs(), self._info()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Cannot call step() after episode is done. Call reset() first.")

        self.step_count += 1
        action_name = self.action_names[int(action)]
        breakdown = RewardBreakdown()
        breakdown.add("step_cost", -0.05)
        terminated = False
        truncated = False
        result: dict[str, Any] | None = None

        target_spec = TOOLS[self.scenario.target_tool]

        if action_name == Action.AUTHENTICATE.value:
            first_auth = not self.simulator.authenticated
            result = self.simulator.authenticate()
            if not first_auth:
                breakdown.add("duplicate_auth_penalty", -0.2)
            elif target_spec.requires_auth:
                breakdown.add("auth_gate", 0.8)
            else:
                breakdown.add("auth_gate", -0.3)
            self.last_event = "authenticated"
            self.last_error_code = "NONE"

        elif action_name == Action.ASK_CONFIRMATION.value:
            first_confirmation = not self.simulator.confirmed
            if not target_spec.requires_confirmation:
                breakdown.add("unneeded_confirmation_penalty", -0.6)
                self.last_event = "unneeded_confirmation"
                result = {"confirmed": False}
            elif target_spec.requires_auth and not self.simulator.authenticated:
                breakdown.add("unsafe_confirmation_penalty", -1.0)
                self.last_event = "confirmation_before_auth"
                result = {"confirmed": False, "error_code": "AUTH_FAILED"}
                self.last_error_code = "AUTH_FAILED"
            elif not self._prerequisites_satisfied():
                breakdown.add("premature_confirmation_penalty", -0.9)
                self.last_event = "confirmation_before_prerequisites"
                result = {"confirmed": False, "error_code": "VALIDATION_ERROR"}
                self.last_error_code = "VALIDATION_ERROR"
            else:
                result = self.simulator.ask_confirmation()
                if first_confirmation:
                    breakdown.add("confirmation_gate", 0.9)
                else:
                    breakdown.add("duplicate_confirmation_penalty", -0.2)
                self.last_event = "confirmed"
                self.last_error_code = "NONE"

        elif action_name == Action.FINISH.value:
            if self.target_called:
                breakdown.add("finish_success", 3.0)
                terminated = True
                self.last_event = "success"
                self.last_error_code = "NONE"
            else:
                breakdown.add("early_finish_penalty", -2.0)
                terminated = True
                self.last_event = "finished_before_target"
                self.last_error_code = "NONE"

        else:
            result, tool_breakdown = self._tool_reward(action_name)
            for name, value in tool_breakdown.items():
                breakdown.add(name, value)

        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            breakdown.add("truncation_penalty", -1.0)
            self.last_event = "max_steps"

        self.done = terminated or truncated
        self.last_result = result
        reward = breakdown.total
        self.trace.append(
            {
                "step": self.step_count,
                "action": action_name,
                "reward": reward,
                "reward_breakdown": breakdown.as_dict(),
                "event": self.last_event,
                "result": result,
            }
        )
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        return (
            f"goal={self.scenario.scenario_id} target={self.scenario.target_tool} "
            f"auth={self.simulator.authenticated} confirm={self.simulator.confirmed} "
            f"called={sorted(self.simulator.called_tools)} event={self.last_event}"
        )

    def _tool_reward(self, tool_name: str) -> tuple[dict[str, Any], dict[str, float]]:
        spec = TOOLS[tool_name]
        target = TOOLS[self.scenario.target_tool]
        breakdown: dict[str, float] = {}
        already_called = tool_name in self.simulator.called_tools

        params = self._params_for(tool_name)
        result = self.simulator.call(tool_name, params)
        if "error_code" in result:
            self.last_event = result["error_code"]
            self.last_error_code = result["error_code"]
            if result["error_code"] == "AUTH_FAILED":
                breakdown["auth_error_penalty"] = -1.0
                return result, breakdown
            if "confirmation" in result["error_message"].lower():
                breakdown["confirmation_error_penalty"] = -1.0
                return result, breakdown
            if "prerequisite" in result["error_message"].lower():
                breakdown["prerequisite_error_penalty"] = -1.25
                return result, breakdown
            breakdown["validation_error_penalty"] = -0.8
            return result, breakdown

        self.last_error_code = "NONE"

        if tool_name == self.scenario.target_tool:
            self.last_event = f"target_called:{tool_name}"
            if self.target_called or already_called:
                breakdown["duplicate_target_penalty"] = -0.5
            else:
                self.target_called = True
                breakdown["target_tool_success"] = 8.0
            return result, breakdown

        if tool_name in target.prerequisites:
            self.last_event = f"prerequisite_called:{tool_name}"
            if already_called:
                breakdown["duplicate_prerequisite_penalty"] = -0.3
            else:
                breakdown["prerequisite_success"] = 1.5
            return result, breakdown

        if already_called:
            self.last_event = f"called:{tool_name}"
            breakdown["duplicate_wrong_tool_penalty"] = -0.3
            return result, breakdown

        self.last_event = f"irrelevant:{tool_name}"
        breakdown["irrelevant_tool_penalty"] = -0.6
        return result, breakdown

    def _params_for(self, tool_name: str) -> dict[str, Any]:
        if tool_name == self.scenario.target_tool:
            return dict(self.scenario.params)

        if tool_name == "calc_fd_maturity":
            return {
                "principal": self.scenario.params.get("principal_amount", self.scenario.params.get("principal", 50000)),
                "interest_rate": 6.75,
                "tenure": self.scenario.params.get("tenure", 365),
                "payout_type": self.scenario.params.get("payout_type", "CUMULATIVE"),
            }

        if tool_name == "get_account_balance":
            return {
                "customer_id": self.scenario.params.get("customer_id", "C001"),
                "account_type": "SAVINGS",
            }

        if tool_name == "calc_premature_penalty":
            return {
                "fd_number": self.scenario.params.get("fd_number", "FD001"),
                "closure_date": self.scenario.params.get("closure_date", "2026-06-04"),
                "reason": self.scenario.params.get("reason", "OTHER"),
            }

        defaults = {
            "get_branch_info": {"query": "Mumbai"},
            "get_gold_rate_today": {},
            "get_fd_details": {"customer_id": "C001"},
            "get_rd_details": {"customer_id": "C001"},
            "get_loan_account_details": {"customer_id": "C001"},
            "get_transaction_history": {"account_id": "SA001", "from_date": "2026-05-01", "to_date": "2026-06-04"},
            "get_kyc_status": {"customer_id": "C001"},
            "get_foreclosure_quote": {"customer_id": "C001"},
            "calc_rd_maturity": {"monthly_installment": 5000, "interest_rate": 6.5, "tenure_months": 24},
            "calc_emi": {"principal": 500000, "interest_rate": 10.5, "tenure_months": 60},
            "calc_gold_ltv": {"gold_weight_grams": 25, "purity": "22K"},
            "search_product_kb": {"query": "fixed deposit rates"},
            "search_rbi_circular": {"query": "gold loan ltv"},
            "get_fd_rate_card": {},
            "get_loan_product_details": {"product_type": "HOME"},
            "generate_uuid": {},
            "raise_service_request": {
                "customer_id": "C001",
                "request_type": "STATEMENT",
                "account_id": "SA001",
                "idempotency_key": "00000000-0000-4000-8000-000000000010",
            },
            "generate_interest_certificate": {
                "customer_id": "C001",
                "financial_year": "2025-26",
                "idempotency_key": "00000000-0000-4000-8000-000000000011",
            },
            "log_complaint": {
                "customer_id": "C001",
                "complaint_category": "SERVICE",
                "description": "Delayed response.",
                "idempotency_key": "00000000-0000-4000-8000-000000000012",
            },
            "book_new_fd": {
                "customer_id": "C001",
                "source_account_id": "SA001",
                "principal_amount": 50000,
                "tenure": 365,
                "idempotency_key": "00000000-0000-4000-8000-000000000013",
            },
            "request_premature_closure": {
                "fd_number": "FD001",
                "closure_date": "2026-06-04",
                "credit_account_id": "SA001",
                "reason": "OTHER",
                "idempotency_key": "00000000-0000-4000-8000-000000000014",
            },
        }
        return defaults[tool_name]

    def _obs(self) -> dict[str, Any]:
        called = [1 if name in self.simulator.called_tools else 0 for name in TOOL_NAMES]
        required_prereqs = [1 if name in self.scenario.prerequisites else 0 for name in TOOL_NAMES]
        satisfied_prereqs = [1 if name in self.simulator.called_tools and name in self.scenario.prerequisites else 0 for name in TOOL_NAMES]
        return {
            "scenario": self.scenarios.index(self.scenario),
            "target_tool": TOOL_NAMES.index(self.scenario.target_tool),
            "authenticated": int(self.simulator.authenticated),
            "confirmed": int(self.simulator.confirmed),
            "called_tools": called,
            "required_prerequisites": required_prereqs,
            "satisfied_prerequisites": satisfied_prereqs,
            "last_error_code": ERROR_CODES.index(self.last_error_code),
            "steps_remaining": max(self.max_steps - self.step_count, 0),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "user_goal": self.scenario.user_goal,
            "target_tool": self.scenario.target_tool,
            "intent": self.scenario.intent,
            "difficulty": self.scenario.difficulty,
            "oracle_actions": self.scenario.oracle_actions(),
            "last_event": self.last_event,
            "last_error_code": self.last_error_code,
            "action_names": self.action_names,
        }

    def _prerequisites_satisfied(self) -> bool:
        return all(name in self.simulator.called_tools for name in self.scenario.prerequisites)

    def _trace_has_event(self, event: str) -> bool:
        return any(item["event"] == event for item in self.trace)
