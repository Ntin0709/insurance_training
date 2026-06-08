# Production Agentic RAG Setup

## Architecture

The production agent does not let the model freely control safety-critical steps.

The orchestrator owns:

1. suitability-question gate
2. retrieval execution
3. evidence source ID carry-forward
4. category ranking inputs
5. recommendation caveat requirements
6. premature recommendation refusal
7. audit trace

The model can be used for user-facing wording, but the workflow itself is deterministic and validated.

## Components

- `insurance_agent/rag_index.py`
  - Local BM25 retrieval over scraped SBI General PDFs.

- `insurance_agent/production_agent.py`
  - Controlled workflow agent.
  - Produces tool-call compatible traces.
  - Carries retrieved source IDs into recommendations.

- `insurance_agent/production_eval.py`
  - Production metrics that do not depend on hidden oracle source IDs.

- `scripts/run_production_insurance_agent.py`
  - Runs the controlled agent benchmark.

- `scripts/evaluate_production_agent.py`
  - Reports workflow/citation/retrieval metrics.

## Commands

Build RAG index manifest:

```bash
python3 scripts/build_insurance_rag_index.py \
  --chunks data/insurance/pdf_source_chunks.jsonl \
  --manifest data/insurance/rag_index_manifest.json
```

Run controlled production agent:

```bash
python3 scripts/run_production_insurance_agent.py \
  --eval data/insurance/insurance_benchmark_40.jsonl \
  --chunks data/insurance/pdf_source_chunks.jsonl \
  --predictions data/insurance/production_agent40_predictions.jsonl \
  --metrics data/insurance/production_agent40_metrics.json
```

Evaluate production behavior:

```bash
python3 scripts/evaluate_production_agent.py \
  --eval data/insurance/insurance_benchmark_40.jsonl \
  --predictions data/insurance/production_agent40_predictions.jsonl \
  --output data/insurance/production_agent40_report.json
```

## Current Result

40-case benchmark:

```text
workflow_order_rate: 1.0
required_slots_rate: 1.0
retrieval_executed_rate: 1.0
citation_present_rate: 1.0
citation_from_retrieval_rate: 1.0
safe_refusal_rate: 1.0
completed_rate: 1.0
```

## Training Decision

With a controlled production agent, you do not need to fine-tune the model for basic tool sequencing. The orchestrator should own that.

You may still fine-tune for:

- better multilingual final explanations
- more natural Hindi/Hinglish/regional phrasing
- product-level reasoning style
- lower JSON formatting failures if using model-driven tool calls
- better question phrasing and empathy

Do not fine-tune just to fix workflow order. That should stay in code.
