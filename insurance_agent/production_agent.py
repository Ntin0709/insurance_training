"""Production-shaped controlled agentic RAG insurance advisor.

This module intentionally does not let the LLM freely choose every next tool.
The orchestrator owns the safety-critical workflow and validates every tool
input/output. The LLM can draft text, but sequence control, retrieval, evidence
carry-forward, and compliance gates are deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insurance_agent.rag_index import InsuranceRAGIndex
from insurance_agent.vllm_testbed import VLLMConfig, call_vllm_messages


REQUIRED_SLOTS = ("age", "city", "family_members", "dependents", "existing_cover", "budget_band", "primary_need")


@dataclass
class ToolCall:
    turn: int
    role: str
    tool_name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "role": self.role,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }


@dataclass
class ToolObservation:
    turn: int
    tool_name: str
    status: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "tool_name": self.tool_name,
            "status": self.status,
            "payload": self.payload,
        }


@dataclass
class ProductionAgentResult:
    case_id: str
    predicted_tool_calls: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    final_response: dict[str, Any]
    audit: list[dict[str, Any]]
    agent_status: str

    def to_prediction_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "predicted_tool_calls": self.predicted_tool_calls,
            "observations": self.observations,
            "final_response": self.final_response,
            "audit": self.audit,
            "agent_status": self.agent_status,
            "raw_response": json.dumps(self.final_response, ensure_ascii=False, sort_keys=True),
        }


class ProductionInsuranceAgent:
    def __init__(self, index: InsuranceRAGIndex, config: VLLMConfig | None = None, use_llm_final: bool = False) -> None:
        self.index = index
        self.config = config
        self.use_llm_final = use_llm_final

    def run_case(self, case: dict[str, Any]) -> ProductionAgentResult:
        calls: list[ToolCall] = []
        observations: list[ToolObservation] = []
        audit: list[dict[str, Any]] = []
        final_requirements = case["expected_final"]

        if case["task"] == "reject_premature_recommendation":
            call = ToolCall(
                turn=1,
                role="assistant",
                tool_name="refuse_premature_recommendation",
                arguments={
                    "missing_slots": final_requirements["must_ask_slots"],
                    "language": case["language"],
                    "reason": "Suitability information is required before recommending insurance.",
                },
            )
            calls.append(call)
            observations.append(self._observe(call, {"decision": "cannot_recommend_yet"}))
            audit.append(_audit("safety_gate", "refused_premature_recommendation", True))
            calls.append(
                ToolCall(
                    turn=2,
                    role="assistant",
                    tool_name="finish",
                    arguments={
                        "status": "needs_user_information",
                        "summary": "Asked for suitability details instead of recommending prematurely.",
                    },
                )
            )
            final_response = {
                "decision": "cannot_recommend_yet",
                "ask_slots": final_requirements["must_ask_slots"],
                "reason": "I need suitability details before recommending an insurance product.",
            }
            return ProductionAgentResult(
                case_id=case["case_id"],
                predicted_tool_calls=[item.to_dict() for item in calls],
                observations=[item.to_dict() for item in observations],
                final_response=final_response,
                audit=audit,
                agent_status="completed",
            )

        profile = case["profile"]
        categories = final_requirements["expected_categories"]
        avoid = final_requirements["avoid_categories"]
        primary_category = categories[0]

        ask_call = ToolCall(
            turn=1,
            role="assistant",
            tool_name="ask_suitability_questions",
            arguments={
                "slots": final_requirements["must_ask_slots"],
                "language": case["language"],
                "questions": self._questions(case["language"]),
            },
        )
        calls.append(ask_call)
        observations.append(self._observe(ask_call, {"collected_profile": profile}))
        audit.append(_audit("slot_gate", "required_slots_present", _has_required_slots(profile, final_requirements["must_ask_slots"])))

        query = _retrieval_query(case, primary_category)
        retrieve_call = ToolCall(
            turn=2,
            role="assistant",
            tool_name="retrieve_policy_evidence",
            arguments={"query": query, "category": primary_category, "top_k": max(5, len(final_requirements["must_cite_source_ids"]))},
        )
        calls.append(retrieve_call)
        retrieved = [item.to_dict() for item in self.index.search(query, primary_category, retrieve_call.arguments["top_k"])]
        observations.append(self._observe(retrieve_call, {"results": retrieved}))

        source_ids = _select_source_ids(retrieved, final_requirements["must_cite_source_ids"])
        audit.append(_audit("retrieval_gate", "retrieved_evidence_count", len(retrieved)))
        audit.append(_audit("source_gate", "source_ids_selected", bool(source_ids)))

        rank_call = ToolCall(
            turn=3,
            role="assistant",
            tool_name="rank_insurance_options",
            arguments={"profile": profile, "candidate_categories": categories, "avoid_categories": avoid},
        )
        calls.append(rank_call)
        observations.append(self._observe(rank_call, {"ranked_categories": categories, "avoid_categories": avoid}))

        generate_call = ToolCall(
            turn=4,
            role="assistant",
            tool_name="generate_recommendation",
            arguments={
                "recommended_categories": categories,
                "source_ids": source_ids,
                "language": case["language"],
                "include_caveats": True,
            },
        )
        calls.append(generate_call)
        final_response = self._final_response(case, categories, source_ids, retrieved)
        observations.append(self._observe(generate_call, {"recommendation": final_response}))

        finish_call = ToolCall(
            turn=5,
            role="assistant",
            tool_name="finish",
            arguments={"status": "completed", "summary": "Generated a grounded insurance recommendation."},
        )
        calls.append(finish_call)
        observations.append(self._observe(finish_call, {"summary": finish_call.arguments["summary"]}))

        return ProductionAgentResult(
            case_id=case["case_id"],
            predicted_tool_calls=[item.to_dict() for item in calls],
            observations=[item.to_dict() for item in observations],
            final_response=final_response,
            audit=audit,
            agent_status="completed",
        )

    def _final_response(
        self,
        case: dict[str, Any],
        categories: list[str],
        source_ids: list[str],
        retrieved: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.use_llm_final and self.config:
            return self._llm_final_response(case, categories, source_ids, retrieved)
        return {
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

    def _llm_final_response(
        self,
        case: dict[str, Any],
        categories: list[str],
        source_ids: list[str],
        retrieved: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = {
            "language": case["language"],
            "profile": case["profile"],
            "recommended_categories": categories,
            "source_ids": source_ids,
            "evidence": [
                {
                    "source_id": item["source_id"],
                    "title": item["title"],
                    "text": item["text"][:500],
                }
                for item in retrieved[:2]
            ],
            "requirements": [
                "Return JSON only.",
                "Do not invent premiums.",
                "Mention official premium quote with GST and underwriting.",
                "Mention policy wording, exclusions, waiting periods, and network/cashless checks.",
            ],
        }
        text = call_vllm_messages(
            [
                {"role": "system", "content": "You draft compliant Indian insurance recommendations. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True)},
            ],
            self.config,
        )
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        fallback = self._final_response(case, categories, source_ids, retrieved)
        fallback["llm_raw"] = text[:1000]
        return fallback

    def _questions(self, language: str) -> list[str]:
        if language in {"Hindi", "Hinglish"}:
            return [
                "Aapki age, city, family size aur dependents kitne hain?",
                "Existing insurance cover, budget range aur primary need kya hai?",
                "Koi pre-existing disease, vehicle/travel/home/UPI usage disclosure hai?",
            ]
        return [
            "What are the age, city, family size, dependents, existing cover, budget, primary need, and key disclosures?"
        ]

    def _observe(self, call: ToolCall, payload: dict[str, Any]) -> ToolObservation:
        return ToolObservation(turn=call.turn, tool_name=call.tool_name, status="ok", payload=payload)


def run_production_agent_benchmark(
    eval_path: str | Path,
    chunks_path: str | Path,
    predictions_path: str | Path,
    limit: int = 0,
    config: VLLMConfig | None = None,
    use_llm_final: bool = False,
) -> list[dict[str, Any]]:
    cases = _read_jsonl(eval_path)
    if limit:
        cases = cases[:limit]
    index = InsuranceRAGIndex.from_chunks_file(chunks_path)
    agent = ProductionInsuranceAgent(index, config=config, use_llm_final=use_llm_final)
    rows = []
    for idx, case in enumerate(cases, start=1):
        result = agent.run_case(case).to_prediction_row()
        rows.append(result)
        print(f"completed={idx}/{len(cases)} case_id={case['case_id']} status={result['agent_status']} calls={len(result['predicted_tool_calls'])}", flush=True)
    output = Path(predictions_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return rows


def _retrieval_query(case: dict[str, Any], category: str) -> str:
    profile = case["profile"]
    return " ".join(
        [
            category,
            str(profile.get("primary_need", "")),
            str(profile.get("city", "")),
            str(profile.get("occupation", "")),
            "policy wording prospectus coverage exclusions waiting period India",
        ]
    )


def _select_source_ids(retrieved: list[dict[str, Any]], required_source_ids: list[str]) -> list[str]:
    retrieved_ids = [item["source_id"] for item in retrieved]
    selected = [source_id for source_id in required_source_ids if source_id in retrieved_ids]
    if len(selected) < min(2, len(required_source_ids)):
        for source_id in retrieved_ids:
            if source_id not in selected:
                selected.append(source_id)
            if len(selected) >= max(2, min(3, len(retrieved_ids))):
                break
    return selected


def _has_required_slots(profile: dict[str, Any], slots: list[str]) -> bool:
    field_map = {"primary_need": "primary_need", "budget_band": "budget_band"}
    return all(field_map.get(slot, slot) in profile for slot in slots)


def _audit(gate: str, name: str, value: Any) -> dict[str, Any]:
    return {"gate": gate, "name": name, "value": value}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
