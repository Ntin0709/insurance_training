# vLLM Insurance Tool-Calling Testbed

This testbed evaluates a locally hosted vLLM model on the Indian insurance advisor multi-turn tool-calling benchmark.

## Start vLLM

Example:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

For a fine-tuned adapter, use your vLLM-compatible adapter loading setup.

## Dry Run

Validates the harness without calling a model:

```bash
python3 scripts/run_vllm_insurance_eval.py \
  --eval data/insurance/insurance_india_toolcall_eval.jsonl \
  --predictions data/insurance/vllm_dryrun_predictions.jsonl \
  --metrics data/insurance/vllm_dryrun_metrics.json \
  --limit 5 \
  --dry-run-oracle
```

## Run Against vLLM

```bash
python3 scripts/run_vllm_insurance_eval.py \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --eval data/insurance/insurance_india_toolcall_eval.jsonl \
  --predictions data/insurance/vllm_toolcall_predictions.jsonl \
  --metrics data/insurance/vllm_toolcall_metrics.json \
  --limit 20
```

Remove `--limit` for the full 428-case evaluation.

## Model Output Contract

The model should return only JSON:

```json
{
  "tool_calls": [
    {
      "turn": 1,
      "role": "assistant",
      "tool_name": "ask_suitability_questions",
      "arguments": {
        "slots": ["age", "city"],
        "language": "English",
        "questions": ["What is your age and city?"]
      }
    }
  ]
}
```

## Metrics

- `exact_tool_sequence_rate`: full sequence and arguments match oracle.
- `tool_order_rate`: tool names are in the right order.
- `required_slots_rate`: required suitability slots are asked.
- `source_ids_rate`: required evidence source IDs are cited.
- `safe_refusal_rate`: premature recommendation cases are refused safely.

Use `tool_order_rate`, `required_slots_rate`, `source_ids_rate`, and `safe_refusal_rate` to decide whether SFT is needed. `exact_tool_sequence_rate` is intentionally strict.
