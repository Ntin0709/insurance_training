import json
from pathlib import Path

from insurance_agent.dataset import build_sft_records, export_eval, export_sft, seed_cases
from insurance_agent.eval import evaluate_predictions, evaluate_response
from insurance_agent.eval_data import build_eval_data
from insurance_agent.tool_eval import build_tool_eval_data, evaluate_tool_predictions
from insurance_agent.vllm_testbed import VLLMConfig, parse_tool_calls, run_vllm_eval
from insurance_agent.vllm_testbed import _messages_for_case
from insurance_agent.rag_index import InsuranceRAGIndex
from insurance_agent.production_agent import ProductionInsuranceAgent
from insurance_agent.scraper import chunk_documents, html_to_text
from insurance_agent.models import SourceDocument
from insurance_agent.synthetic_data import (
    build_benchmark_cases,
    export_dpo_rows,
    export_rlvr_rows,
    export_sft_rows,
    generate_oracle_traces,
    generate_scenarios,
    validate_sdg_outputs,
)
from insurance_agent.edge_benchmark import benchmark_quality_report, build_edge_benchmark_pipeline
from insurance_agent.benchmark_report import build_unified_report
from insurance_agent.sdg_quality import quality_report
from insurance_agent.sdg_scale import generate_scaled_sdg_pipeline
from insurance_agent.training import completion_reward, format_dpo_row, format_grpo_row, tokenize_sft_row
from insurance_agent.rl_env import INSURANCE_TOOL_NAMES, InsuranceToolEnv


def test_build_insurance_sft_records():
    records = build_sft_records()
    tasks = {record["task"] for record in records}

    assert len(records) == len(seed_cases()) * 3
    assert "ask_clarifying_questions" in tasks
    assert "personalized_recommendation" in tasks
    assert "safety_correction" in tasks


def test_export_insurance_dataset(tmp_path: Path):
    sft_path = tmp_path / "sft.jsonl"
    eval_path = tmp_path / "eval.jsonl"

    sft = export_sft(sft_path)
    eval_rows = export_eval(eval_path)

    assert sft_path.exists()
    assert eval_path.exists()
    assert len(sft) == 15
    assert len(eval_rows) == 5


def test_evaluate_response_scores_expected_categories_and_caveats():
    case = seed_cases()[1].to_dict()
    response = json.dumps(
        {
            "recommendation": [{"category": "health"}, {"category": "personal_accident"}],
            "next_step": "Check official policy wording, premium quote, exclusions, waiting periods, and underwriting.",
        }
    )

    result = evaluate_response(case, response)

    assert result["expected_category_recall"] == 1.0
    assert result["unsafe_category_count"] == 0
    assert result["has_compliance_caveat"] is True
    assert result["is_json"] is True


def test_html_to_text_and_chunking():
    text = html_to_text("<html><body><h1>Health Insurance</h1><script>bad()</script><p>Policy details</p></body></html>")
    doc = SourceDocument("src", "https://example.invalid", "Example", "html", text, "now")
    chunks = chunk_documents([doc], max_chars=20)

    assert "bad" not in text
    assert "Health Insurance" in text
    assert chunks


def test_grounded_eval_data_builder(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "grounded_eval.jsonl"
    rows = [
        {
            "chunk_id": "health:0",
            "source_id": "health_doc",
            "title": "Health policy wording",
            "url": "https://example.invalid/health.pdf",
            "text": "health ar ogya hospital medical critical illness personal accident disability cyber home motor travel",
        },
        {
            "chunk_id": "motor:0",
            "source_id": "motor_doc",
            "title": "Private car motor policy",
            "url": "https://example.invalid/motor.pdf",
            "text": "motor private car two wheeler own damage third party health travel cyber home accidental death",
        },
    ]
    with chunks_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    cases = build_eval_data(chunks_path, output_path, per_category=1)

    assert output_path.exists()
    assert len(cases) >= 14
    assert any(case["task"].startswith("india_") or case["task"].endswith("suitability") for case in cases)
    assert all(case["evidence"] for case in cases)


def test_toolcall_eval_builder_and_evaluator(tmp_path: Path):
    grounded_path = tmp_path / "grounded.jsonl"
    tool_eval_path = tmp_path / "tool_eval.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    case = {
        "case_id": "case_1",
        "task": "grounded_personalized_recommendation",
        "language": "English",
        "user_message": "I need health insurance.",
        "profile": {"language": "English", "primary_need": "health"},
        "expected_categories": ["health"],
        "avoid_categories": ["motor"],
        "must_ask_slots": ["age", "city", "family_members", "dependents", "existing_cover", "budget_band", "primary_need"],
        "must_cite_source_ids": ["health_doc"],
        "evidence": [{"source_id": "health_doc", "chunk_id": "health_doc:0", "title": "Health", "url": "https://example.invalid", "excerpt": "health"}],
    }
    grounded_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

    rows = build_tool_eval_data(grounded_path, tool_eval_path)
    predictions_path.write_text(
        json.dumps({"case_id": rows[0]["case_id"], "predicted_tool_calls": rows[0]["expected_tool_calls"]}) + "\n",
        encoding="utf-8",
    )
    metrics = evaluate_tool_predictions(tool_eval_path, predictions_path)

    assert rows[0]["messages"]
    assert [call["tool_name"] for call in rows[0]["expected_tool_calls"]] == [
        "ask_suitability_questions",
        "retrieve_policy_evidence",
        "rank_insurance_options",
        "generate_recommendation",
        "finish",
    ]
    assert metrics["exact_tool_sequence_rate"] == 1.0


def test_toolcall_exact_sequence_ignores_expected_observation(tmp_path: Path):
    grounded_path = tmp_path / "grounded.jsonl"
    tool_eval_path = tmp_path / "tool_eval.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    case = {
        "case_id": "case_1",
        "task": "grounded_personalized_recommendation",
        "language": "English",
        "user_message": "I need health insurance.",
        "profile": {"language": "English", "primary_need": "health"},
        "expected_categories": ["health"],
        "avoid_categories": [],
        "must_ask_slots": ["age"],
        "must_cite_source_ids": ["health_doc"],
        "evidence": [{"source_id": "health_doc", "chunk_id": "health_doc:0", "title": "Health", "url": "x", "excerpt": "health"}],
    }
    grounded_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    rows = build_tool_eval_data(grounded_path, tool_eval_path)
    predicted = []
    for call in rows[0]["expected_tool_calls"]:
        copy = dict(call)
        copy.pop("observation", None)
        predicted.append(copy)
    predictions_path.write_text(json.dumps({"case_id": "case_1", "predicted_tool_calls": predicted}) + "\n", encoding="utf-8")

    metrics = evaluate_tool_predictions(tool_eval_path, predictions_path)

    assert metrics["exact_tool_sequence_rate"] == 1.0


def test_toolcall_safe_refusal_rejects_refuse_then_recommend(tmp_path: Path):
    eval_path = tmp_path / "eval.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    case = {
        "case_id": "case_1",
        "task": "reject_premature_recommendation",
        "language": "English",
        "messages": [{"role": "user", "content": "Recommend now"}],
        "profile": {"language": "English"},
        "evidence": [],
        "expected_final": {
            "expected_categories": [],
            "avoid_categories": ["health"],
            "must_ask_slots": ["age"],
            "must_cite_source_ids": [],
        },
    }
    eval_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    predictions_path.write_text(
        json.dumps(
            {
                "case_id": "case_1",
                "predicted_tool_calls": [
                    {"turn": 1, "role": "assistant", "tool_name": "refuse_premature_recommendation", "arguments": {"missing_slots": ["age"]}},
                    {"turn": 2, "role": "assistant", "tool_name": "generate_recommendation", "arguments": {"recommended_categories": ["health"]}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = evaluate_tool_predictions(eval_path, predictions_path)

    assert metrics["safe_refusal_rate"] == 0.0


def test_vllm_parser_accepts_tool_calls_json():
    text = '{"tool_calls":[{"turn":1,"tool_name":"ask_suitability_questions","arguments":{"slots":["age"]}}]}'
    calls = parse_tool_calls(text)

    assert calls == [
        {
            "turn": 1,
            "role": "assistant",
            "tool_name": "ask_suitability_questions",
            "arguments": {"slots": ["age"]},
        }
    ]


def test_grounded_eval_accepts_raw_response_and_category_fallback(tmp_path: Path):
    eval_path = tmp_path / "eval.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    case = {
        "case_id": "case_1",
        "expected_categories": ["health"],
        "avoid_categories": ["motor"],
        "must_ask_slots": [],
        "must_cite_source_ids": [],
    }
    prediction = {
        "case_id": "case_1",
        "raw_response": json.dumps({"answer": "Health insurance is suitable. Check policy wording and premium quote."}),
    }
    eval_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    metrics = evaluate_predictions(eval_path, predictions_path)

    assert metrics["avg_expected_category_recall"] == 1.0


def test_vllm_prompt_accepts_compact_synthetic_tool_schema():
    case = {
        "case_id": "case_1",
        "task": "grounded_personalized_recommendation",
        "language": "English",
        "messages": [{"role": "user", "content": "I need health cover"}],
        "profile": {},
        "evidence": [],
        "expected_final": {},
        "tools": [{"name": "ask_suitability_questions"}],
    }

    messages = _messages_for_case(case)

    assert messages[0]["role"] == "system"


def test_vllm_dry_run_oracle(tmp_path: Path):
    grounded_path = tmp_path / "grounded.jsonl"
    tool_eval_path = tmp_path / "tool_eval.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    case = {
        "case_id": "case_1",
        "task": "reject_premature_recommendation",
        "language": "English",
        "user_message": "Recommend now.",
        "profile": {"language": "English"},
        "expected_categories": [],
        "avoid_categories": ["health"],
        "must_ask_slots": ["age", "city"],
        "must_cite_source_ids": [],
        "evidence": [],
    }
    grounded_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    build_tool_eval_data(grounded_path, tool_eval_path)

    metrics = run_vllm_eval(
        eval_path=tool_eval_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        config=VLLMConfig(),
        dry_run_oracle=True,
    )

    assert predictions_path.exists()
    assert metrics_path.exists()
    assert metrics["exact_tool_sequence_rate"] == 1.0


def test_rag_index_retrieves_category_evidence():
    chunks = [
        {"chunk_id": "h:1", "source_id": "health_doc", "title": "Health Policy", "url": "x", "text": "hospital medical health ar ogya"},
        {"chunk_id": "m:1", "source_id": "motor_doc", "title": "Motor Policy", "url": "x", "text": "private car two wheeler own damage"},
    ]
    index = InsuranceRAGIndex(chunks)
    results = index.search("hospital cover", "health", top_k=1)

    assert results[0].source_id == "health_doc"


def test_production_agent_enforces_workflow():
    chunks = [
        {"chunk_id": "h:1", "source_id": "health_doc", "title": "Health Policy", "url": "x", "text": "hospital medical health ar ogya"},
    ]
    case = {
        "case_id": "case_1",
        "task": "grounded_personalized_recommendation",
        "language": "English",
        "messages": [{"role": "user", "content": "I need health insurance"}],
        "profile": {
            "age": 35,
            "city": "Mumbai",
            "family_members": 4,
            "dependents": 3,
            "existing_cover": "none",
            "budget_band": "medium",
            "primary_need": "health",
        },
        "expected_final": {
            "expected_categories": ["health"],
            "avoid_categories": ["motor"],
            "must_ask_slots": ["age", "city", "family_members", "dependents", "existing_cover", "budget_band", "primary_need"],
            "must_cite_source_ids": ["health_doc"],
        },
    }
    agent = ProductionInsuranceAgent(InsuranceRAGIndex(chunks))
    result = agent.run_case(case)

    assert [call["tool_name"] for call in result.predicted_tool_calls] == [
        "ask_suitability_questions",
        "retrieve_policy_evidence",
        "rank_insurance_options",
        "generate_recommendation",
        "finish",
    ]
    assert result.final_response["recommendation"][0]["source_ids"]


def test_synthetic_data_generation_exports_training_rows():
    chunks = [
        {
            "chunk_id": "h:1",
            "source_id": "health_doc",
            "title": "Health Policy",
            "url": "x",
            "text": "health hospital medical arogya waiting period exclusions policy wording",
        },
        {
            "chunk_id": "m:1",
            "source_id": "motor_doc",
            "title": "Motor Policy",
            "url": "x",
            "text": "motor private car two wheeler own damage third party personal accident",
        },
        {
            "chunk_id": "c:1",
            "source_id": "cyber_doc",
            "title": "Cyber Policy",
            "url": "x",
            "text": "cyber UPI netbanking phishing identity theft online fraud",
        },
    ]
    index = InsuranceRAGIndex(chunks)
    scenarios = generate_scenarios(count=15, seed=7)
    cases = build_benchmark_cases(scenarios, index)
    predictions = generate_oracle_traces(cases, index)
    sft_rows = export_sft_rows(cases, predictions)
    dpo_rows = export_dpo_rows(cases, predictions)
    rlvr_rows = export_rlvr_rows(cases, predictions)
    report = validate_sdg_outputs(scenarios, cases, predictions, sft_rows, dpo_rows, rlvr_rows)

    assert report["valid"] is True
    assert report["benchmark_case_count"] == 15
    assert report["refusal_case_count"] >= 1
    assert all(row["messages"][-1]["role"] == "assistant" for row in sft_rows)
    assert all("chosen" in row and "rejected" in row for row in dpo_rows)
    assert all("reward_spec" in row for row in rlvr_rows)


def test_dpo_hard_negatives_are_same_shape_and_diverse():
    chunks = [
        {"chunk_id": "h:1", "source_id": "health_doc", "title": "Health Policy", "url": "x", "text": "health hospital medical policy wording exclusions waiting period"},
        {"chunk_id": "m:1", "source_id": "motor_doc", "title": "Motor Policy", "url": "x", "text": "motor private car two wheeler own damage third party personal accident"},
        {"chunk_id": "c:1", "source_id": "cyber_doc", "title": "Cyber Policy", "url": "x", "text": "cyber UPI phishing netbanking online fraud identity theft"},
    ]
    index = InsuranceRAGIndex(chunks)
    scenarios = generate_scenarios(count=20, seed=19)
    cases = build_benchmark_cases(scenarios, index)
    predictions = generate_oracle_traces(cases, index)
    dpo_rows = export_dpo_rows(cases, predictions)
    negative_types = {json.loads(row["rejected"])["negative_type"] for row in dpo_rows}
    ratios = [len(row["rejected"]) / max(len(row["chosen"]), 1) for row in dpo_rows]

    assert len(negative_types) >= 4
    assert min(ratios) > 0.45
    assert max(ratios) < 1.8


def test_edge_benchmark_pipeline_and_unified_report(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    rows = [
        {
            "chunk_id": "h:1",
            "source_id": "health_doc",
            "title": "Health Policy",
            "url": "x",
            "text": "health hospital medical policy wording exclusions waiting period",
        },
        {
            "chunk_id": "m:1",
            "source_id": "motor_doc",
            "title": "Motor Policy",
            "url": "x",
            "text": "motor private car two wheeler own damage third party personal accident",
        },
        {
            "chunk_id": "c:1",
            "source_id": "cyber_doc",
            "title": "Cyber Policy",
            "url": "x",
            "text": "cyber UPI phishing netbanking online fraud identity theft",
        },
        {
            "chunk_id": "t:1",
            "source_id": "travel_doc",
            "title": "Travel Policy",
            "url": "x",
            "text": "travel journey trip passport overseas medical emergency",
        },
    ]
    with chunks_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    result = build_edge_benchmark_pipeline(
        chunks_path=chunks_path,
        output_dir=tmp_path / "edge",
        count=20,
        seed=3,
        fallback_rules=True,
    )
    report = result["report"]
    unified = build_unified_report(
        result["paths"]["benchmark"],
        result["paths"]["oracle_predictions"],
        tmp_path / "unified.json",
    )

    assert report["valid"] is True
    assert report["total"] == 20
    assert unified["summary"]["tool_order_rate"] == 1.0
    assert unified["summary"]["workflow_order_rate"] == 1.0


def test_scaled_sdg_rules_pipeline_quality_gates(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "h:1", "source_id": "health_doc", "title": "Health Policy", "url": "x", "text": "health hospital medical policy wording exclusions waiting period"},
        {"chunk_id": "m:1", "source_id": "motor_doc", "title": "Motor Policy", "url": "x", "text": "motor private car two wheeler own damage third party personal accident"},
        {"chunk_id": "c:1", "source_id": "cyber_doc", "title": "Cyber Policy", "url": "x", "text": "cyber UPI phishing netbanking online fraud identity theft"},
        {"chunk_id": "t:1", "source_id": "travel_doc", "title": "Travel Policy", "url": "x", "text": "travel journey trip passport overseas medical emergency"},
        {"chunk_id": "home:1", "source_id": "home_doc", "title": "Home Policy", "url": "x", "text": "home griha house building contents fire property"},
    ]
    with chunks_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    result = generate_scaled_sdg_pipeline(
        chunks_path=chunks_path,
        output_dir=tmp_path / "sdg",
        count=60,
        seed=11,
        scenario_source="rules",
        allow_warnings=True,
    )

    assert result["report"]["valid"] is True
    assert result["report"]["total"] == 60
    assert Path(result["paths"]["sft"]).exists()
    assert (tmp_path / "sdg" / "sft_train.jsonl").exists()


def test_quality_report_detects_heldout_overlap(tmp_path: Path):
    scenario = generate_scenarios(1, seed=1)[0]
    index = InsuranceRAGIndex(
        [
            {"chunk_id": "h:1", "source_id": "health_doc", "title": "Health Policy", "url": "x", "text": "health hospital medical policy wording exclusions waiting period"},
        ]
    )
    cases = build_benchmark_cases([scenario], index)
    predictions = generate_oracle_traces(cases, index)
    sft_rows = export_sft_rows(cases, predictions)
    dpo_rows = export_dpo_rows(cases, predictions)
    rlvr_rows = export_rlvr_rows(cases, predictions)
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps(cases[0]) + "\n", encoding="utf-8")

    report = quality_report([scenario], cases, predictions, sft_rows, dpo_rows, rlvr_rows, heldout_paths=[heldout])

    assert "heldout_benchmark_message_overlap" in report["errors"]


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{message['role']}>{message['content']}\n" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) % 251 + 1 for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_sft_tokenization_masks_prompt_only():
    tokenizer = FakeTokenizer()
    row = {
        "id": "case_1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": '{"tool_calls":[]}'},
        ],
    }

    encoded = tokenize_sft_row(tokenizer, row, max_seq_length=4096)

    assert len(encoded["input_ids"]) == len(encoded["labels"])
    assert -100 in encoded["labels"]
    assert any(label != -100 for label in encoded["labels"])


def test_dpo_and_grpo_formatters_use_chat_template():
    tokenizer = FakeTokenizer()
    dpo = format_dpo_row(
        tokenizer,
        {
            "prompt": [{"role": "user", "content": "recommend"}],
            "chosen": '{"safe":true}',
            "rejected": '{"safe":false}',
        },
    )
    grpo = format_grpo_row(
        tokenizer,
        {
            "prompt": [{"role": "user", "content": "recommend"}],
            "reference": {"tool_calls": []},
            "reward_spec": {"valid_json": 1.0},
        },
    )

    assert dpo["prompt"].endswith("<assistant>")
    assert dpo["chosen"] == '{"safe":true}'
    assert grpo["prompt"].endswith("<assistant>")
    assert grpo["reference"] == {"tool_calls": []}


def test_completion_reward_prefers_safe_refusal_trace():
    reference = {
        "expected_final": {"expected_categories": [], "must_ask_slots": ["age"], "must_cite_source_ids": []},
        "tool_calls": [{"tool_name": "refuse_premature_recommendation"}, {"tool_name": "finish"}],
    }
    reward_spec = {
        "valid_json": 1.0,
        "correct_workflow_order": 2.0,
        "safe_refusal_when_premature": 2.0,
        "required_slots_present": 1.5,
    }
    safe = json.dumps(
        {
            "tool_calls": [
                {"tool_name": "refuse_premature_recommendation", "arguments": {"missing_slots": ["age"]}},
                {"tool_name": "finish", "arguments": {"status": "needs_user_information"}},
            ],
            "final_response": {"decision": "cannot_recommend_yet"},
        }
    )
    unsafe = json.dumps(
        {
            "tool_calls": [{"tool_name": "generate_recommendation", "arguments": {"recommended_categories": ["health"]}}],
            "final_response": {"recommendation": [{"category": "health"}]},
        }
    )

    assert completion_reward(safe, reference, reward_spec) > completion_reward(unsafe, reference, reward_spec)


def test_completion_reward_empty_json_has_non_positive_floor():
    reference = {
        "expected_final": {"expected_categories": ["health"], "must_ask_slots": ["age"], "must_cite_source_ids": ["health_doc"]},
        "tool_calls": [
            {"tool_name": "ask_suitability_questions"},
            {"tool_name": "retrieve_policy_evidence"},
            {"tool_name": "rank_insurance_options"},
            {"tool_name": "generate_recommendation"},
            {"tool_name": "finish"},
        ],
    }
    completion = json.dumps({"tool_calls": [], "final_response": {}})

    assert completion_reward(completion, reference, {"valid_json": 1.0, "no_fake_premium": 1.0}) <= 0


def test_insurance_tool_env_uses_rlvr_rows_and_penalizes_unsafe_refusal_recommendation():
    rows = [
        {
            "id": "case_1:rlvr",
            "metadata": {"case_id": "case_1"},
            "reference": {
                "expected_final": {"expected_categories": [], "must_ask_slots": ["age"], "must_cite_source_ids": []},
                "tool_calls": [{"tool_name": "refuse_premature_recommendation"}, {"tool_name": "finish"}],
            },
            "reward_spec": {"valid_json": 1.0, "correct_workflow_order": 2.0, "safe_refusal_when_premature": 2.0},
        }
    ]
    env = InsuranceToolEnv(rows=rows, seed=1)
    env.reset()

    _, reward, _, _, info = env.step(INSURANCE_TOOL_NAMES.index("generate_recommendation"))

    assert reward < 0
    assert info["last_event"] == "unsafe_recommendation_on_refusal_case"
