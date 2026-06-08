# Agentic RAG Evaluation Report

## What Changed

The earlier benchmark prompted the model with evidence already present in the eval case. That is useful for tool-plan formatting, but it is not a production RAG test.

The new pipeline is production-shaped:

1. Build a local retrieval index over scraped SBI General PDFs.
2. Ask the model for one next tool call.
3. Execute the tool locally.
4. Feed tool observations back to the model.
5. Continue until `finish`.

## RAG Index

Manifest: `data/insurance/rag_index_manifest.json`

```text
chunk_count: 16,335
source_count: 790
vocab_size: 27,720
avg_doc_length_tokens: 292.22
```

## Direct Tool-Plan Benchmark

This benchmark includes evidence in the prompt.

40-case result against hosted `/model`:

```text
tool_order_rate: 0.925
required_slots_rate: 0.975
source_ids_rate: 0.975
safe_refusal_rate: 1.0
exact_tool_sequence_rate: 0.0
```

Interpretation: the model can imitate a full plan when all evidence and final requirements are visible.

## True Agentic RAG Smoke Test

This benchmark does not put evidence in the prompt. The model must call retrieval and use tool observations.

3-case result:

```text
tool_order_rate: 0.0
required_slots_rate: 0.0
source_ids_rate: 0.0
safe_refusal_rate: 1.0
```

Observed model behavior:

```text
retrieve_policy_evidence {}
rank_insurance_options {}
generate_recommendation {}
finish {}
```

Failure pattern:

- skipped `ask_suitability_questions`
- called tools with empty arguments
- did not formulate retrieval query/category
- did not pass retrieved source IDs into recommendation

## Training Decision

This is now enough evidence to justify training or at least targeted tool-use adaptation.

The open-source model can produce a plausible plan when spoon-fed evidence, but it does not reliably operate as an agentic RAG policy when it must:

- decide the next tool call step by step
- fill tool arguments
- consume tool observations
- carry source IDs forward
- preserve safety gates before recommendation

Recommended next step:

1. Generate SFT traces from the agentic RAG oracle.
2. Fine-tune on next-tool-call examples with tool observations.
3. Re-run the true agentic RAG benchmark.
4. Only consider RLVR after SFT reaches high format and tool-order reliability.

Target thresholds before production:

```text
tool_order_rate >= 0.98
required_slots_rate >= 0.98
source_ids_rate >= 0.95
safe_refusal_rate >= 0.99
empty_argument_tool_call_rate <= 0.01
parse_failure_rate <= 0.01
```
