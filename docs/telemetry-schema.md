# Telemetry and observability schema

Each traced pipeline execution is appended as one JSON object per line (JSONL). The trace records the evolving transcript decisions, retrieval events, grounded output, answer version, timing, provider token usage, and estimated inference cost. Traces are local artifacts; session state itself remains in memory.

## Required fields

These fields are required by `validate-traces` and are emitted for complete and failed executions.

| Field | Type | Meaning |
|---|---|---|
| `trace_id` | string | Unique execution identifier. |
| `recorded_at_utc` | string | ISO 8601 recording time with UTC offset. |
| `status` | `complete` or `error` | Whether the traced pipeline call completed. |
| `domain` | string | Retrieval corpus domain, normally `cloud` or `govt`. |
| `retrieval_events` | array of objects | Searches actually issued; each has `timestamp_s`, `query`, and `trigger`. Timestamps are relative to the stream start. |
| `decisions` | array of objects | Per-chunk controller decisions; each has `timestamp_s`, transcript `text`, `decision`, and `reason`. Decisions are `Wait`, `Retrieve`, or `No-Retrieval`. |
| `citations` | array of strings | Corpus passage IDs cited in the answer. |
| `answer` | string | Answer text returned for the execution. |
| `version` | integer | Answer version number. |
| `previous_version` | integer or null | Prior version replaced or refined by this execution. |
| `applied_delta` | string or null | Late-arriving constraint applied to the prior answer, when applicable. |
| `stage_latency_ms` | object of numbers | Stage durations in milliseconds; current keys include `retrieval`, `generation`, and `total`. |
| `token_usage` | object of nonnegative integers | `prompt_tokens`, `completion_tokens`, and `total_tokens` attributed to provider calls in this execution. |
| `estimated_cost_usd` | nonnegative number or null | Estimated provider cost. Zero means no provider tokens; null means cost could not be estimated. |
| `cost_estimate_basis` | string | Pricing source or reason the estimate is zero or unavailable. |

## Optional fields

Dashboard SSE events additionally include `id` (reconnect cursor) and
`observed_elapsed_s` (actual wall time since run creation). `retrieval_start`
means preprocessing began; `search_dispatch` is emitted immediately before the
retriever call and records `queries`, `trigger`, and simulated `timestamp_s`.
Compare dispatch wall time with the final chunk's wall time for actual live G2
timing; simulated timestamps alone are not proof of early real-time dispatch.
`validate_trace_file(..., expected_ids={...})` validates exact execution IDs in
an append-only log, ignoring unrelated older runs and rejecting missing IDs.

`sub_queries` is an array of retrieval query strings when available. `uncertainty` is a string or null. `claim_citations`, when available, maps extractive answer sentences to the passage IDs that contain them. These fields may be absent from evaluator-generated traces that do not expose the corresponding pipeline detail.

Failed executions still receive a trace. Fields unavailable before failure use empty arrays/objects or null values, and `cost_estimate_basis` explains why no estimate was recorded. A `complete` trace must have a non-null cost estimate; provider-free executions use zero.

## Validation and measured coverage

Validate required fields, trace ID uniqueness, and expected execution coverage with:

```bash
.venv/bin/prism-rag curated-data validate-traces \
  --input data/curated_dataset/evaluation/g6-full-streaming-gpu-20260929.jsonl \
  --expected 200
```

The 200-execution GPU streaming trace artifact validated with 100% coverage. That run was deterministic and provider-free, so its token counts and estimated costs are zero. A separate Cerebras-backed provider trace records nonzero tokens and an estimated cost. Cost values are estimates, not billing records.
