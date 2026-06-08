"""LLM dataset generation and tool-call evaluation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from banking_rl_env.env import BankingToolEnv
from banking_rl_env.policies import oracle_action
from banking_rl_env.scenarios import SCENARIOS, Scenario
from banking_rl_env.tools import ACTIONS, TOOL_NAMES, TOOLS, Action, ToolSpec


SYSTEM_PROMPT = """You are a banking assistant that must use tools safely.
Follow these rules:
- Authenticate before tools that read customer data or change bank state.
- Ask user confirmation before write tools and official document actions.
- Before booking a fixed deposit, calculate maturity and check account balance.
- Before premature FD closure, calculate the premature closure penalty.
- Use idempotency keys exactly as provided for write tools.
- Return exactly one JSON object per assistant turn.
"""

CONTROL_TOOL_SCHEMAS = (
    {
        "name": Action.AUTHENTICATE.value,
        "description": "Authenticate the customer session before private-data or write actions.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": Action.ASK_CONFIRMATION.value,
        "description": "Ask the user to confirm an action after prerequisites and impact have been shown.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": Action.FINISH.value,
        "description": "Finish the task after the requested information or action has been completed.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
    },
)


TYPE_HINTS = {
    "amount": "number",
    "principal": "number",
    "principal_amount": "number",
    "monthly_installment": "number",
    "interest_rate": "number",
    "tenure": "integer",
    "tenure_months": "integer",
    "gold_weight_grams": "number",
    "ltv_percentage": "number",
    "radius_km": "number",
    "limit": "integer",
    "offset": "integer",
    "include_overdue": "boolean",
    "include_co_applicants": "boolean",
    "include_collateral": "boolean",
    "senior_citizen": "boolean",
    "auto_renewal": "boolean",
    "account_ids": "array",
    "tenure_range": "object",
}


@dataclass(frozen=True)
class LLMExample:
    example_id: str
    scenario_id: str
    task_type: str
    messages: list[dict[str, Any]]
    expected_actions: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.example_id,
                "scenario_id": self.scenario_id,
                "task_type": self.task_type,
                "messages": self.messages,
                "expected_actions": self.expected_actions,
            },
            sort_keys=True,
        )


def tool_schema(spec: ToolSpec) -> dict[str, Any]:
    properties = {}
    for param in (*spec.required_params, *spec.optional_params):
        properties[param] = {"type": _json_type(param)}
    return {
        "name": spec.name,
        "description": (
            f"{spec.category} banking tool. "
            f"auth_required={spec.requires_auth}; confirmation_required={spec.requires_confirmation}; "
            f"prerequisites={list(spec.prerequisites)}"
        ),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(spec.required_params),
        },
    }


def all_tool_schemas(include_control_tools: bool = True) -> list[dict[str, Any]]:
    schemas = [tool_schema(TOOLS[name]) for name in TOOL_NAMES]
    if include_control_tools:
        return [*CONTROL_TOOL_SCHEMAS, *schemas]
    return schemas


def build_action_call(scenario: Scenario, action_name: str) -> dict[str, Any]:
    if action_name == Action.AUTHENTICATE.value:
        return {"tool": action_name, "arguments": {}}
    if action_name == Action.ASK_CONFIRMATION.value:
        return {"tool": action_name, "arguments": {}}
    if action_name == Action.FINISH.value:
        return {"tool": action_name, "arguments": {"summary": _finish_summary(scenario)}}
    if action_name == scenario.target_tool:
        return {"tool": action_name, "arguments": dict(scenario.params)}
    return {"tool": action_name, "arguments": _prerequisite_params(scenario, action_name)}


def generate_sft_examples(scenarios: tuple[Scenario, ...] = SCENARIOS) -> list[LLMExample]:
    examples: list[LLMExample] = []
    for scenario in scenarios:
        examples.append(_full_trajectory_example(scenario))
        examples.extend(_next_action_examples(scenario))
    return examples


def write_jsonl(path: str | Path, examples: list[LLMExample]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.to_json() + "\n")


def export_dataset(output_path: str | Path, scenarios: tuple[Scenario, ...] = SCENARIOS) -> list[LLMExample]:
    examples = generate_sft_examples(scenarios)
    write_jsonl(output_path, examples)
    return examples


def evaluate_action_sequence(scenario: Scenario, predicted_actions: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [build_action_call(scenario, action) for action in scenario.oracle_actions()]
    expected_tools = [item["tool"] for item in expected]
    predicted_tools = [item.get("tool") for item in predicted_actions]
    exact_sequence = predicted_actions == expected
    tool_sequence_accuracy = int(predicted_tools == expected_tools)
    first_error = None
    for idx, expected_item in enumerate(expected):
        if idx >= len(predicted_actions):
            first_error = {"index": idx, "reason": "missing_action", "expected": expected_item}
            break
        if predicted_actions[idx] != expected_item:
            first_error = {"index": idx, "reason": "mismatch", "expected": expected_item, "predicted": predicted_actions[idx]}
            break
    return {
        "scenario_id": scenario.scenario_id,
        "exact_sequence": exact_sequence,
        "tool_sequence_accuracy": tool_sequence_accuracy,
        "expected_tools": expected_tools,
        "predicted_tools": predicted_tools,
        "first_error": first_error,
    }


def _full_trajectory_example(scenario: Scenario) -> LLMExample:
    expected_actions = [build_action_call(scenario, action) for action in scenario.oracle_actions()]
    return LLMExample(
        example_id=f"{scenario.scenario_id}:trajectory",
        scenario_id=scenario.scenario_id,
        task_type="full_trajectory",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(scenario)},
            {"role": "assistant", "content": json.dumps({"actions": expected_actions}, sort_keys=True)},
        ],
        expected_actions=expected_actions,
    )


def _next_action_examples(scenario: Scenario) -> list[LLMExample]:
    env = BankingToolEnv(scenarios=(scenario,), max_steps=10, seed=0)
    env.reset(options={"scenario_id": scenario.scenario_id})
    examples: list[LLMExample] = []

    while True:
        action_idx = oracle_action(env)
        action_name = ACTIONS[action_idx]
        expected_action = build_action_call(scenario, action_name)
        examples.append(
            LLMExample(
                example_id=f"{scenario.scenario_id}:step:{env.step_count + 1}",
                scenario_id=scenario.scenario_id,
                task_type="next_action",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(scenario)},
                    {
                        "role": "assistant_context",
                        "content": json.dumps(
                            {
                                "completed_actions": [item["action"] for item in env.trace],
                                "authenticated": env.simulator.authenticated,
                                "confirmed": env.simulator.confirmed,
                                "called_tools": sorted(env.simulator.called_tools),
                            },
                            sort_keys=True,
                        ),
                    },
                    {"role": "assistant", "content": json.dumps(expected_action, sort_keys=True)},
                ],
                expected_actions=[expected_action],
            )
        )
        env.step(action_idx)
        if action_name == Action.FINISH.value:
            break

    return examples


def _user_prompt(scenario: Scenario) -> str:
    return (
        f"Goal: {scenario.user_goal}\n"
        f"Known parameters: {json.dumps(scenario.params, sort_keys=True)}\n"
        f"Intent: {scenario.intent}\n"
        f"Return the next safe tool call or complete tool-call sequence as JSON."
    )


def _prerequisite_params(scenario: Scenario, action_name: str) -> dict[str, Any]:
    if action_name == "calc_fd_maturity":
        return {
            "principal": scenario.params.get("principal_amount", scenario.params.get("principal", 50000)),
            "interest_rate": 6.75,
            "tenure": scenario.params.get("tenure", 365),
            "payout_type": scenario.params.get("payout_type", "CUMULATIVE"),
        }
    if action_name == "get_account_balance":
        return {"customer_id": scenario.params.get("customer_id", "C001"), "account_type": "SAVINGS"}
    if action_name == "calc_premature_penalty":
        return {
            "fd_number": scenario.params.get("fd_number", "FD001"),
            "closure_date": scenario.params.get("closure_date", "2026-06-04"),
            "reason": scenario.params.get("reason", "OTHER"),
        }
    return {}


def _finish_summary(scenario: Scenario) -> str:
    if TOOLS[scenario.target_tool].write_tool:
        return f"Completed {scenario.target_tool} safely after required gates."
    return f"Returned result for {scenario.target_tool}."


def _json_type(param: str) -> str:
    if param in TYPE_HINTS:
        return TYPE_HINTS[param]
    if param.startswith("include_") or param.startswith("auto_"):
        return "boolean"
    return "string"
