# Insurance SDG At Scale

The scalable SDG path is separate from the frozen benchmark. Do not train on:

```text
data/insurance/edge_benchmark_llm_300/
```

## High-Volume Bootstrapping Data

Generated:

```text
data/insurance/sdg_rules_50k_balanced/
```

Quality:

```text
total: 50000
sft_count: 50000
dpo_count: 50000
rlvr_count: 50000
unique_user_messages: 50000
heldout_overlap_count: 0
split_leakage: {}
valid: true
warnings: []
```

This is useful as a bootstrapping/control dataset. It is deterministic and
balanced, but LLM-generated data should be mixed in before serious SFT.

## LLM Variation Data

Checkpointed hosted-LLM generation started here:

```text
data/insurance/sdg_llm_1k/
```

Current materialized subset:

```text
total: 320
unique_user_messages: 320
heldout_overlap_count: 0
valid: true
warning: low_refusal_coverage
```

The refusal-target generator has been fixed after this checkpoint. Resume with:

```bash
PYTHONPATH=. python3 scripts/generate_insurance_sdg_scale.py \
  --scenario-source llm \
  --count 1000 \
  --seed 2051 \
  --output-dir data/insurance/sdg_llm_1k \
  --heldout data/insurance/edge_benchmark_llm_300/edge_benchmark.jsonl \
  --allow-warnings \
  --base-url http://14.195.12.18:9532/v1 \
  --model /model \
  --api-key EMPTY \
  --llm-batch-size 10 \
  --checkpoint-every 20 \
  --timeout 180 \
  --max-tokens 4096
```

For a proper first training mix:

```text
45k deterministic balanced rows
5k-20k LLM varied rows
frozen 300-case benchmark held out
```

## Quality Gates

The deterministic gate checks:

- duplicate case IDs
- duplicate normalized user messages
- train/val/test message leakage
- held-out benchmark message overlap
- source grounding on non-refusal cases
- refusal cases with no citations
- category balance
- language balance
- refusal coverage
- exact premium leakage
- SFT/DPO/RLVR export count consistency
