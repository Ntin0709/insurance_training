# Gemma 3 Insurance Advisor Training

This repo now supports the intended three-stage path for a small Gemma 3 insurance advisor:

1. SFT for JSON tool traces and safe advisor workflow.
2. DPO for preference alignment against unsafe or ungrounded answers.
3. GRPO/RLVR for deterministic reward optimization on tool order, refusal safety, grounding, and no fake premium behavior.

## Data

Primary deterministic corpus:

```bash
data/insurance/sdg_rules_50k_balanced/sft_train.jsonl
data/insurance/sdg_rules_50k_balanced/dpo_train.jsonl
data/insurance/sdg_rules_50k_balanced/rlvr_train.jsonl
```

LLM-generated corpus is being written under:

```bash
data/insurance/sdg_llm_20k/
```

Use the deterministic corpus first for baseline training. Mix the LLM corpus only after its `synthetic_quality_report.json` is valid.
For phrasing diversity, create a paraphrased corpus from verified rule labels:

```bash
python3 scripts/paraphrase_insurance_scenarios.py \
  --input-scenarios data/insurance/sdg_rules_50k_balanced/synthetic_scenarios.jsonl \
  --output-dir data/insurance/sdg_paraphrased_20k \
  --count 20000 \
  --base-url http://14.195.12.18:9532/v1 \
  --model /model \
  --api-key EMPTY \
  --allow-warnings
```

Rebuild existing DPO rows with hard same-shape negatives:

```bash
python3 scripts/rebuild_insurance_dpo_hard_negatives.py \
  --output-dir data/insurance/sdg_rules_50k_balanced
```

## Dependencies

```bash
python3 -m pip install -e ".[llm,rl]"
```

For GRPO, use a TRL version with `GRPOTrainer`:

```bash
python3 -m pip install "trl>=0.16"
```

## SFT

```bash
accelerate launch scripts/train_insurance_lora.py \
  --model google/gemma-3-4b-it \
  --train-file data/insurance/sdg_rules_50k_balanced/sft_train.jsonl \
  --validation-file data/insurance/sdg_rules_50k_balanced/sft_val.jsonl \
  --output-dir outputs/gemma3-insurance-sft-lora \
  --max-seq-length 4096 \
  --batch-size 1 \
  --grad-accum 16
```

On the 8x H100 host, use the checked-in Accelerate config:

```bash
accelerate launch --config_file configs/accelerate_8x_h100.yaml scripts/train_insurance_lora.py \
  --model google/gemma-3-4b-it \
  --train-file data/insurance/sdg_rules_50k_balanced/sft_train.jsonl \
  --validation-file data/insurance/sdg_rules_50k_balanced/sft_val.jsonl
```

The SFT trainer uses the model chat template and masks prompt tokens so loss is applied only to the assistant tool-trace label.

## DPO

Start from the SFT adapter merged or loaded as the base model path:

```bash
accelerate launch scripts/train_gemma3_dpo.py \
  --model google/gemma-3-4b-it \
  --init-adapter outputs/gemma3-insurance-sft-lora \
  --train-file data/insurance/sdg_rules_50k_balanced/dpo_train.jsonl \
  --output-dir outputs/gemma3-insurance-dpo-lora \
  --batch-size 1 \
  --grad-accum 16
```

## GRPO / RLVR

Start from the DPO output:

```bash
accelerate launch scripts/train_gemma3_grpo.py \
  --model google/gemma-3-4b-it \
  --init-adapter outputs/gemma3-insurance-dpo-lora \
  --train-file data/insurance/sdg_rules_50k_balanced/rlvr_train.jsonl \
  --validation-file data/insurance/sdg_rules_50k_balanced/rlvr_val.jsonl \
  --output-dir outputs/gemma3-insurance-grpo-lora \
  --num-generations 8 \
  --batch-size 1 \
  --grad-accum 16
```

The GRPO reward is deterministic and scores valid JSON, workflow order, safe refusal, required suitability slots, retrieval-before-recommendation, source IDs, no fake premium, and compliance caveats.

## Insurance RL Environment

For tool-selection RL experiments independent of text generation:

```python
from insurance_agent.rl_env import InsuranceToolEnv

env = InsuranceToolEnv(dataset_path="data/insurance/sdg_rules_50k_balanced/rlvr_train.jsonl")
obs, info = env.reset()
```

This environment uses the insurance RLVR rows, not the older banking tool-selection environment.

## Post-Train Evaluation

Score a trained adapter locally on the held-out edge benchmark:

```bash
python3 scripts/evaluate_local_peft_insurance.py \
  --model google/gemma-3-4b-it \
  --adapter outputs/gemma3-insurance-grpo-lora \
  --eval data/insurance/edge_benchmark_llm_300/edge_benchmark.jsonl \
  --predictions outputs/gemma3-insurance-grpo-lora/edge_predictions.jsonl \
  --report outputs/gemma3-insurance-grpo-lora/edge_report.json
```
