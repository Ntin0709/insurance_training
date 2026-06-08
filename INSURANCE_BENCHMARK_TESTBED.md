# Insurance Model Benchmark Testbed

This benchmark layer is separate from training SDG. Its purpose is to break
models and expose whether the system needs fine-tuning, prompt changes, or
orchestration changes.

## Edge Benchmark Generation

LLM-generated hard benchmark:

```bash
PYTHONPATH=. python3 scripts/build_insurance_edge_benchmark.py \
  --base-url http://14.195.12.18:9532/v1 \
  --model /model \
  --api-key EMPTY \
  --count 45 \
  --llm-batch-size 3 \
  --seed 2031 \
  --output-dir data/insurance/edge_benchmark_llm_45
```

Offline fallback benchmark:

```bash
PYTHONPATH=. python3 scripts/build_insurance_edge_benchmark.py \
  --fallback-rules \
  --count 150 \
  --seed 2026 \
  --output-dir data/insurance/edge_benchmark_rules
```

## Covered Edge Buckets

```text
premature_recommendation_pressure
prompt_injection_ignore_safety
fake_premium_or_discount_claim
claim_guarantee_or_no_waiting_period
missing_or_conflicting_profile
pre_existing_disease_senior_parent
low_budget_rural_customer
upi_cyber_fraud_scam
vehicle_legal_vs_optional_cover
home_loan_property_risk
student_or_senior_travel
comparison_multiple_needs
out_of_scope_life_or_investment
claims_support_not_recommendation
regional_language_code_mix
```

## Unified Report

```bash
PYTHONPATH=. python3 scripts/report_unified_insurance_benchmark.py \
  --eval data/insurance/edge_benchmark_llm_45/edge_benchmark.jsonl \
  --predictions data/insurance/edge_benchmark_llm_45/edge_oracle_predictions.jsonl \
  --output data/insurance/edge_benchmark_llm_45/unified_report.json
```

The unified report combines:

- tool-call sequence metrics
- production workflow metrics
- citation/retrieval metrics
- final-answer safety metrics
- slice metrics by language, task, risk tag, and category

## Optional LLM Judge

Use only as an additional qualitative layer, not as the only score.

```bash
PYTHONPATH=. python3 scripts/judge_insurance_final_answers.py \
  --eval data/insurance/edge_benchmark_llm_45/edge_benchmark.jsonl \
  --predictions data/insurance/edge_benchmark_llm_45/edge_oracle_predictions.jsonl \
  --base-url http://14.195.12.18:9532/v1 \
  --model /model \
  --api-key EMPTY \
  --limit 10 \
  --output data/insurance/edge_benchmark_llm_45/llm_judge_report.json
```

## Current Generated Artifacts

```text
data/insurance/edge_benchmark_rules/
data/insurance/edge_benchmark_llm_30/
data/insurance/edge_benchmark_llm_45/
```

The 45-case LLM benchmark is the first useful real-AI edge smoke. For a serious
held-out benchmark, generate 300-1,000 LLM cases, then freeze it and never train
on it.
