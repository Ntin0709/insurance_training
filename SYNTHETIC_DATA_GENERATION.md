# Insurance Synthetic Data Generation

This repo has an SDG pipeline for Indian insurance-agent training.
The recommended production mode uses an LLM to generate diverse customer
scenarios, then deterministic code validates, grounds, deduplicates, splits,
and exports training/eval rows.

## Run

```bash
PYTHONPATH=. python3 scripts/generate_insurance_sdg.py \
  --chunks data/insurance/pdf_source_chunks.jsonl \
  --output-dir data/insurance/synthetic \
  --count 500 \
  --seed 2026
```

## LLM Scenario Generation

Use this when generating real training data. The LLM creates customer scenario
diversity; the pipeline still enforces schemas, allowed insurance categories,
RAG grounding, safe-refusal labels, and reward specs.

```bash
PYTHONPATH=. python3 scripts/generate_insurance_sdg.py \
  --scenario-source llm \
  --base-url http://14.195.12.18:9532/v1 \
  --model /model \
  --api-key EMPTY \
  --chunks data/insurance/pdf_source_chunks.jsonl \
  --output-dir data/insurance/synthetic_llm_25k \
  --count 25000 \
  --llm-batch-size 10 \
  --seed 2026
```

Use `--scenario-source hybrid` for 80% LLM-generated scenarios and 20% rule
coverage anchors.

## Outputs

```text
data/insurance/synthetic/synthetic_scenarios.jsonl
data/insurance/synthetic/synthetic_benchmark.jsonl
data/insurance/synthetic/synthetic_oracle_predictions.jsonl
data/insurance/synthetic/synthetic_sft.jsonl
data/insurance/synthetic/synthetic_dpo.jsonl
data/insurance/synthetic/synthetic_rlvr.jsonl
data/insurance/synthetic/synthetic_report.json
data/insurance/synthetic/synthetic_production_report.json
```

## What It Generates

- Scenario cards across Indian customer contexts.
- Benchmark-compatible cases with `expected_final`.
- Oracle production-agent traces.
- SFT rows for tool-trace imitation.
- DPO rows with safe chosen responses and unsafe rejected responses.
- RLVR rows with deterministic reward specifications.

Current taxonomy:

```text
new_family_health
senior_parent_ped
new_car_owner
two_wheeler_owner
upi_cyber_risk
home_loan_property
student_international_travel
domestic_frequent_travel
rural_low_budget
self_employed_income_protection
compare_two_needs
premature_recommendation_pressure
missing_budget
claim_misconception
price_only_request
```

Languages:

```text
Bengali, English, Hindi, Hinglish, Kannada, Malayalam, Marathi, Tamil, Telugu
```

## Validate

```bash
PYTHONPATH=. python3 scripts/evaluate_production_agent.py \
  --eval data/insurance/synthetic/synthetic_benchmark.jsonl \
  --predictions data/insurance/synthetic/synthetic_oracle_predictions.jsonl \
  --output data/insurance/synthetic/synthetic_production_report.json
```

Expected oracle metrics:

```text
workflow_order_rate: 1.0
required_slots_rate: 1.0
retrieval_executed_rate: 1.0
citation_present_rate: 1.0
citation_from_retrieval_rate: 1.0
safe_refusal_rate: 1.0
completed_rate: 1.0
```

## Training Use

Use `synthetic_sft.jsonl` first for LoRA SFT.
Use `synthetic_dpo.jsonl` after SFT to prefer safe grounded answers.
Use `synthetic_rlvr.jsonl` only after SFT/DPO, with deterministic graders for
tool order, required slots, citation grounding, safe refusal, no fake premium,
and compliance caveats.
