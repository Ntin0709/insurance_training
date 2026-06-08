# Banking RL Environment

This repo contains a reinforcement learning environment for the banking tools specified in `banking_tools_reference (2).md`.

The first environment focuses on tool-use policy learning:

- choosing the right banking tool for a user goal
- authenticating before private-data or write tools
- asking confirmation only after the financial impact is known
- calling prerequisite tools before high-impact write actions
- handling validation, insufficient funds, duplicate requests, and idempotency
- optimizing full episode success rather than a single tool classification

## Files

- `banking_rl_env/tools.py` contains all 24 tool specs, categories, required params, auth tiers, confirmation tiers, and prerequisites.
- `banking_rl_env/simulator.py` is a deterministic banking backend simulator.
- `banking_rl_env/scenarios.py` defines train/eval scenarios.
- `banking_rl_env/env.py` exposes `BankingToolEnv`, a Gymnasium-compatible environment.
- `banking_rl_env/policies.py` provides an oracle policy and evaluation helper.
- `banking_rl_env/cli.py` provides a simple command line evaluator.
- `tests/test_env.py` covers basic sequencing and idempotency behavior.

## Quick Start

```python
from banking_rl_env import ACTIONS, BankingToolEnv

env = BankingToolEnv(seed=7)
obs, info = env.reset(options={"scenario_id": "book_fd"})

for action_name in [
    "AUTHENTICATE",
    "calc_fd_maturity",
    "get_account_balance",
    "ASK_CONFIRMATION",
    "book_new_fd",
    "FINISH",
]:
    obs, reward, terminated, truncated, info = env.step(ACTIONS.index(action_name))
    print(action_name, reward, info["last_event"])
    if terminated or truncated:
        break
```

## Action Space

Actions are discrete:

1. `AUTHENTICATE`
2. `ASK_CONFIRMATION`
3. one action for each banking tool
4. `FINISH`

The environment currently supplies scenario-specific valid parameters internally. That keeps this benchmark focused on sequencing, tool selection, and safety gates. Parameter-generation can be added as a second action head later without changing the simulator contract.

## Observation

The observation includes:

- sampled scenario id
- target tool id
- authentication and confirmation flags
- called-tool bitset
- required-prerequisite bitset
- satisfied-prerequisite bitset
- last error code
- remaining steps

## Reward Sketch

Each step records a `reward_breakdown` in `env.trace`, so training failures are inspectable. The reward includes:

- large positive reward for the correct target tool after requirements
- positive reward for required prerequisites
- positive reward for useful authentication and confirmation
- penalties for missing auth, premature confirmation, missing prerequisites, invalid parameters, wrong tools, duplicate gates, early finish, and max-step truncation
- terminal reward for successful `FINISH`

## Evaluate Oracle

```bash
python3 -m banking_rl_env.cli --episodes 100 --dump-trace
```

## Train Baseline

```bash
python3 examples/train_q_learning.py --episodes 3000
```

This trains a dependency-free tabular Q-learning policy over the structured observations.

## Train A Small LLM For Tool Use

The RL environment now exports supervised tool-call trajectories for small LLM fine-tuning.

Generate JSONL training data and tool schemas:

```bash
python3 scripts/export_llm_dataset.py \
  --output data/banking_tool_sft.jsonl \
  --schemas-output data/tool_schemas.json
```

The JSONL contains two kinds of examples:

- `full_trajectory`: one assistant answer containing the complete safe action sequence.
- `next_action`: step-by-step examples where the model predicts the next safe tool call from the current state.

Install optional training dependencies:

```bash
python3 -m pip install "transformers>=4.41" "datasets>=2.19" "trl>=0.9" "peft>=0.11" accelerate bitsandbytes
```

Fine-tune a small model with LoRA:

```bash
python3 scripts/train_small_llm_lora.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/banking_tool_sft.jsonl \
  --output-dir outputs/banking-tool-lora
```

Evaluate model-produced tool sequences by writing predictions as JSONL:

```json
{"scenario_id":"book_fd","predicted_actions":[{"tool":"AUTHENTICATE","arguments":{}},{"tool":"calc_fd_maturity","arguments":{"principal":50000,"interest_rate":6.75,"tenure":365,"payout_type":"CUMULATIVE"}}]}
```

Then run:

```bash
python3 scripts/evaluate_llm_predictions.py --predictions data/predictions.jsonl
```

## Run Tests

```bash
python -m pytest
```

`gymnasium` is optional. If installed, `BankingToolEnv` exposes Gymnasium spaces; without it, the same `reset`, `step`, and `render` methods still work for simple experiments.
