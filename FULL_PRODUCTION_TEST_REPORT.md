# Full Production Test Report

## Scope

This report covers the production-shaped insurance advisor agent for Indian banking customers using official SBI General PDF data.

## Data Foundation

Scraped/discovered official PDFs:

```text
discovered_pdf_urls: 831
extracted_pdf_docs: 800
failed_pdf_urls: 31
pdf_chunks: 16,335
source_count_in_rag_index: 790
```

RAG index:

```text
index_type: local BM25
chunk_count: 16,335
vocab_size: 27,720
avg_doc_length_tokens: 292.22
```

## Agent Architecture

The production agent uses controlled orchestration:

1. Ask suitability questions.
2. Retrieve official policy evidence.
3. Rank candidate categories.
4. Generate recommendation with source IDs and caveats.
5. Finish.

Premature recommendation requests:

1. Refuse premature recommendation.
2. Finish.

Safety-critical sequence control is deterministic code, not LLM-driven.

## 120-Case Benchmark Result

Benchmark file:

`data/insurance/insurance_benchmark_120.jsonl`

Predictions:

`data/insurance/production_agent120_predictions.jsonl`

Production report:

`data/insurance/production_agent120_report.json`

Metrics:

```text
workflow_order_rate: 1.0
required_slots_rate: 1.0
retrieval_executed_rate: 1.0
citation_present_rate: 1.0
citation_from_retrieval_rate: 1.0
safe_refusal_rate: 1.0
completed_rate: 1.0
```

Final-answer structured safety:

```text
category_recall: 1.0
source_present_rate: 1.0
caveat_rate: 1.0
fake_premium_rate: 0.0
safe_refusal_final_rate: 1.0
indian_context_rate: 0.9667
```

## Hosted Model Final-Answer Test

Attempted `--use-llm-final` with the hosted `/model` endpoint.

Result:

```text
status: blocked by latency
sample_size_attempted: 2
max_tokens: 512
timeout: 120s
result: first response did not complete within practical test window
```

This does not prove the model cannot write good final answers. It proves the current hosted endpoint/settings are not yet suitable for responsive final-answer generation in this setup.

## Training/Fine-Tuning Decision

Do not fine-tune for workflow sequencing. The production orchestrator solves that.

Training may still be needed for:

- final answer language quality
- Hindi/Hinglish/regional fluency
- concise customer-facing explanations
- product-level reasoning style
- robust JSON formatting if using LLM-driven final responses

Before deciding on SFT, first fix inference latency and run the final-answer benchmark with `--use-llm-final`.

## Not Yet Proven

The following are still required before real customer use:

- human insurance/compliance review of final answer templates
- retrieval relevance review for top products
- adversarial safety tests
- long messy multi-turn user sessions
- actual quote-tool integration
- PII/session security
- monitoring and audit log retention
- fallback behavior for retrieval misses
- latency and load testing

## Current Recommendation

The controlled agent is production-shaped and passes workflow benchmarks.

It is not yet customer-production-ready until final-answer quality, latency, compliance review, and quote-tool integration are complete.
