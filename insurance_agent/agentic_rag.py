"""Agentic RAG runner for insurance advisor tool-calling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insurance_agent.rag_index import InsuranceRAGIndex
from insurance_agent.vllm_testbed import VLLMConfig, call_vllm_messages, parse_tool_calls


AGENT_TOOLS = (
    "ask_suitability_questions",
    "retrieve_policy_evidence",
    "rank_insurance_options",
    "generate_recommendation",
    "refuse_premature_recommendation",
    "finish",
)


@dataclass
class AgentState:
    case: dict[str, Any]
    observations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_response: dict[str, Any] | None = None


class InsuranceAgenticRAG:
    def __init__(self, index: InsuranceRAGIndex, config: VLLMConfig, max_steps: int = 6) -> None:
        self.index = index
        self.config = config
        self.max_steps = max_steps

    def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        state = AgentState(case=case)
        for step in range(1, self.max_steps + 1):
            prompt_case = self._prompt_case(state)
            raw = call_vllm_messages(prompt_case["messages"], self.config)
            calls = parse_tool_calls(raw)
            if not calls:
                return self._prediction(case, state, raw, "parse_failed")

            call = calls[0]
            call["turn"] = len(state.tool_calls) + 1
            state.tool_calls.append(call)
            observation = self._execute_tool(call, state)
            state.observations.append(observation)
            if call["tool_name"] == "finish":
                return self._prediction(case, state, raw, "finished")

        return self._prediction(case, state, "", "max_steps")

    def _execute_tool(self, call: dict[str, Any], state: AgentState) -> dict[str, Any]:
        name = call["tool_name"]
        args = call.get("arguments", {})
        if name == "ask_suitability_questions":
            return {"tool_name": name, "status": "ok", "collected_profile": state.case["profile"], "asked_slots": args.get("slots", [])}
        if name == "retrieve_policy_evidence":
            results = self.index.search(args.get("query", ""), args.get("category"), int(args.get("top_k", 5)))
            return {"tool_name": name, "status": "ok", "results": [result.to_dict() for result in results]}
        if name == "rank_insurance_options":
            candidates = args.get("candidate_categories", [])
            avoid = args.get("avoid_categories", [])
            if not isinstance(candidates, list):
                candidates = []
            if not isinstance(avoid, list):
                avoid = []
            return {"tool_name": name, "status": "ok", "ranked_categories": candidates, "avoid_categories": avoid}
        if name == "generate_recommendation":
            source_ids = args.get("source_ids") or _source_ids_from_observations(state.observations)
            response = {
                "recommended_categories": args.get("recommended_categories", []),
                "source_ids": source_ids,
                "caveats": [
                    "Check official policy wording, premium quote with GST, eligibility, exclusions, waiting periods, network/cashless availability, and underwriting before purchase."
                ],
            }
            state.final_response = response
            return {"tool_name": name, "status": "ok", "recommendation": response}
        if name == "refuse_premature_recommendation":
            state.final_response = {"decision": "cannot_recommend_yet", "missing_slots": args.get("missing_slots", [])}
            return {"tool_name": name, "status": "ok", "refusal": state.final_response}
        if name == "finish":
            return {"tool_name": name, "status": "ok", "summary": args.get("summary", "")}
        return {"tool_name": name, "status": "error", "error": "unknown_tool"}

    def _prompt_case(self, state: AgentState) -> dict[str, Any]:
        case = state.case
        tools = [
            {
                "name": tool,
                "description": _tool_description(tool),
            }
            for tool in AGENT_TOOLS
        ]
        expected_shape = {
            "tool_calls": [
                {
                    "tool_name": "ask_suitability_questions",
                    "arguments": {},
                }
            ]
        }
        prompt = {
            "case_id": case["case_id"],
            "task": case["task"],
            "language": case["language"],
            "user_message": case["messages"][-1]["content"],
            "profile_if_known": case["profile"],
            "available_tools": tools,
            "previous_tool_calls": state.tool_calls,
            "tool_observations": state.observations,
            "instructions": [
                "Return exactly one next tool call as JSON.",
                "Do not include markdown.",
                "Do not fabricate source IDs; use IDs returned by retrieve_policy_evidence.",
                "Normal flow: ask_suitability_questions -> retrieve_policy_evidence -> rank_insurance_options -> generate_recommendation -> finish.",
                "If user asks for immediate recommendation without suitability details: refuse_premature_recommendation -> finish.",
            ],
            "output_shape": expected_shape,
        }
        return {
            "case_id": case["case_id"],
            "messages": [
                {"role": "system", "content": "You are an Indian bank insurance advisor running a tool-use workflow. Return only JSON."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True)},
            ],
            "tools": [],
            "evidence": [],
            "expected_final": case["expected_final"],
        }

    def _prediction(self, case: dict[str, Any], state: AgentState, raw: str, status: str) -> dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "predicted_tool_calls": state.tool_calls,
            "raw_response": raw,
            "agent_status": status,
            "observations": state.observations,
            "final_response": state.final_response,
        }


def run_agentic_rag_benchmark(
    eval_path: str | Path,
    chunks_path: str | Path,
    predictions_path: str | Path,
    config: VLLMConfig,
    limit: int = 0,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(eval_path)
    if limit:
        rows = rows[:limit]
    index = InsuranceRAGIndex.from_chunks_file(chunks_path)
    agent = InsuranceAgenticRAG(index, config)
    predictions = []
    for idx, case in enumerate(rows, start=1):
        prediction = agent.run_case(case)
        predictions.append(prediction)
        print(f"completed={idx}/{len(rows)} case_id={case['case_id']} status={prediction['agent_status']} calls={len(prediction['predicted_tool_calls'])}", flush=True)
    output = Path(predictions_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return predictions


def _tool_description(tool: str) -> str:
    return {
        "ask_suitability_questions": "Ask for age, city, family size, dependents, existing cover, budget, primary need, and disclosures.",
        "retrieve_policy_evidence": "Retrieve official SBI General policy/prospectus evidence from local RAG index.",
        "rank_insurance_options": "Rank candidate insurance categories for the profile.",
        "generate_recommendation": "Create a recommendation using retrieved source IDs and compliance caveats.",
        "refuse_premature_recommendation": "Refuse recommendation when suitability details are missing.",
        "finish": "End the workflow.",
    }[tool]


def _source_ids_from_observations(observations: list[dict[str, Any]]) -> list[str]:
    ids = []
    for observation in observations:
        for result in observation.get("results", []):
            ids.append(result["source_id"])
    return ids[:3]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
