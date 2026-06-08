from banking_rl_env import ACTIONS, BankingToolEnv, BankingToolSimulator
from banking_rl_env.llm import all_tool_schemas, build_action_call, evaluate_action_sequence, generate_sft_examples
from banking_rl_env.policies import oracle_action, run_policy


def act(name: str) -> int:
    return ACTIONS.index(name)


def test_public_tool_can_complete_without_auth_or_confirmation():
    env = BankingToolEnv(seed=1)
    env.reset(options={"scenario_id": "public_branch_lookup"})

    _, reward, terminated, truncated, info = env.step(act("get_branch_info"))
    assert reward > 0
    assert not terminated
    assert not truncated
    assert info["last_event"] == "target_called:get_branch_info"

    _, reward, terminated, truncated, info = env.step(act("FINISH"))
    assert reward > 0
    assert terminated
    assert not truncated
    assert info["last_event"] == "success"


def test_write_tool_requires_auth_confirmation_and_prerequisites():
    env = BankingToolEnv(seed=1)
    env.reset(options={"scenario_id": "book_fd"})

    _, reward, *_ = env.step(act("book_new_fd"))
    assert reward < 0

    env.step(act("AUTHENTICATE"))
    env.step(act("calc_fd_maturity"))
    env.step(act("get_account_balance"))
    env.step(act("ASK_CONFIRMATION"))
    _, reward, _, _, info = env.step(act("book_new_fd"))

    assert reward > 0
    assert info["last_event"] == "target_called:book_new_fd"


def test_repeated_gate_and_target_calls_do_not_farm_reward():
    env = BankingToolEnv(seed=1, max_steps=10)
    env.reset(options={"scenario_id": "book_fd"})

    _, first_auth_reward, *_ = env.step(act("AUTHENTICATE"))
    _, second_auth_reward, *_ = env.step(act("AUTHENTICATE"))
    env.step(act("calc_fd_maturity"))
    _, repeated_prereq_reward, *_ = env.step(act("calc_fd_maturity"))
    env.step(act("get_account_balance"))
    _, first_confirm_reward, *_ = env.step(act("ASK_CONFIRMATION"))
    _, second_confirm_reward, *_ = env.step(act("ASK_CONFIRMATION"))
    _, target_reward, *_ = env.step(act("book_new_fd"))
    _, repeated_target_reward, *_ = env.step(act("book_new_fd"))

    assert first_auth_reward > 0
    assert second_auth_reward < 0
    assert repeated_prereq_reward < 0
    assert first_confirm_reward > 0
    assert second_confirm_reward < 0
    assert target_reward > 0
    assert repeated_target_reward < 0


def test_first_irrelevant_tool_gets_irrelevant_penalty():
    env = BankingToolEnv(seed=1)
    env.reset(options={"scenario_id": "public_branch_lookup"})

    _, reward, _, _, info = env.step(act("get_gold_rate_today"))

    assert reward < 0
    assert info["last_event"] == "irrelevant:get_gold_rate_today"


def test_simulator_reuses_idempotency_response():
    sim = BankingToolSimulator()
    sim.authenticate()
    sim.ask_confirmation()
    params = {
        "customer_id": "C001",
        "complaint_category": "SERVICE",
        "description": "Delayed response",
        "idempotency_key": "00000000-0000-4000-8000-000000000099",
    }

    first = sim.call("log_complaint", params)
    second = sim.call("log_complaint", params)

    assert first["complaint_id"] == second["complaint_id"]
    assert second["duplicate_request"] is True


def test_oracle_policy_solves_all_scenarios():
    env = BankingToolEnv(seed=1, max_steps=10)
    for scenario in env.scenarios:
        env.reset(options={"scenario_id": scenario.scenario_id})
        while True:
            _, _, terminated, truncated, info = env.step(oracle_action(env))
            if terminated or truncated:
                break

        assert info["last_event"] == "success", scenario.scenario_id
        assert env.trace[-1]["event"] == "success"


def test_oracle_metrics_are_perfect():
    env = BankingToolEnv(seed=1, max_steps=10)
    metrics = run_policy(env, oracle_action, episodes=20)

    assert metrics["success_rate"] == 1.0
    assert metrics["avg_reward"] > 0


def test_transaction_history_rejects_more_than_three_years():
    sim = BankingToolSimulator()
    sim.authenticate()

    result = sim.call(
        "get_transaction_history",
        {
            "account_id": "SA001",
            "from_date": "2020-01-01",
            "to_date": "2026-06-04",
        },
    )

    assert result["error_code"] == "VALIDATION_ERROR"
    assert "3 years" in result["error_message"]


def test_llm_dataset_contains_trajectory_and_next_action_examples():
    env = BankingToolEnv(seed=1)
    examples = generate_sft_examples(env.scenarios)

    assert len(examples) > len(env.scenarios)
    assert any(example.task_type == "full_trajectory" for example in examples)
    assert any(example.task_type == "next_action" for example in examples)
    assert all(example.expected_actions for example in examples)


def test_tool_schemas_include_control_and_banking_tools():
    schemas = all_tool_schemas()
    names = {schema["name"] for schema in schemas}

    assert "AUTHENTICATE" in names
    assert "ASK_CONFIRMATION" in names
    assert "book_new_fd" in names
    assert "request_premature_closure" in names


def test_llm_oracle_sequence_evaluates_exactly():
    env = BankingToolEnv(seed=1)
    scenario = next(item for item in env.scenarios if item.scenario_id == "book_fd")
    predicted = [build_action_call(scenario, action) for action in scenario.oracle_actions()]

    result = evaluate_action_sequence(scenario, predicted)
    assert result["exact_sequence"] is True
