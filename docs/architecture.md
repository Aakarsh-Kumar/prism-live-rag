# Architecture Reference

## Pipeline overview

```
Incoming Stream: [Chunk 0.0s] → [Chunk 0.8s] → [Chunk 1.6s] → [Utterance End 2.1s]
        │
        ▼
① Retrieval Controller — Intent Stability Check → Decision: Wait | Retrieve | No-Retrieval
        │ (Retrieve Triggered)
        ▼
② Multi-Intent Decomposer — Extract Sub-Queries → Parallel Intent Routing
        │
        ▼
③ Corpus Retrieval & Fusion — Search Corpus → Dense/Sparse Hybrid Scoring → Rerank & Dedup
        │
        ▼
④ Session-Aware Synthesis — Incremental Answer Update → Grounding & Citation Check → Uncertainty Flag
        │
        ▼
Output: Streamed Answer + Grounded Citations + Observability Telemetry
```

## ① Retrieval Controller

**Job:** decide, per incoming transcript chunk, whether to retrieve now, wait for more
speech, or suppress retrieval entirely.

**Two implementations required** (Samsung's brief explicitly wants a rule-based vs.
model-based ablation):

- **Rule-based (build first):** heuristic on entity/intent completeness. A practical
  starting threshold from production voice-RAG guides: fire retrieval once a stable
  ASR partial transcript clears ~70–80% confidence, running retrieval in parallel with
  time-to-first-token rather than waiting for full generation.
- **Model-based (build second, for the ablation):** Adaptive-RAG-style classifier —
  trained on auto-collected labels (which strategy actually worked best per query,
  not hand-annotated) to route to no-retrieval / single-step / iterative retrieval.

**Conceptual framing to reuse in your architecture brief:** treat this as three
named sub-problems (from the "Stream RAG" paper) —
*Trigger* (when to issue a new query), *Threads* (how many parallel query streams),
*Reflector* (whether intermediate results are sufficient to answer yet). Structuring
your controller code around these three names makes the design legible and citable.

**Known pitfall:** eager/premature retrieval on noise — triggering search on every
incremental token causes thrashing, high compute cost, and noisy context windows. The
controller must wait for semantic intent stability, not just any partial transcript.

## ② Multi-Intent Decomposer

**Job:** split one utterance into 2–4 independent, search-ready sub-queries and route
them for parallel retrieval.

- Cap sub-queries at 2–4. Over-fragmenting a simple question pollutes the reranker and
  wastes token budget (documented pitfall).
- Use `concat(last-turn ∥ standalone-rewrite)` as your base query formulation before
  decomposition — validated gain in ablations at zero added latency.
- Don't decompose dependent multi-hop questions into parallel sub-queries — they need
  sequential resolution, not parallel search.

## ③ Corpus Retrieval & Fusion

See `retrieval.md` for full config, hyperparameters, and validated/invalidated
techniques.

## ④ Session-Aware Synthesis

See `generation-grounding.md` for full config, prompts, and validated techniques.

## ⑤ Observability & Telemetry (cross-cutting)

Log from day one, not as a final step:
- Timestamps for every retrieval decision (wait/retrieve/suppress) and why
- Retrieval trigger events: query issued, timestamp, trigger type (provisional /
  multi-intent / refinement)
- Sub-queries generated per utterance
- Citations returned, with source chunk IDs
- Answer version lineage (for session refinement / delta updates)
- Token cost and latency per stage

## Structured output schema

Samsung's brief specifies this exact JSON shape — treat it as a hard requirement, not
a suggestion:

```json
{
  "retrieval_events": [
    { "timestamp_s": 0.8, "query": "...", "trigger": "provisional" },
    { "timestamp_s": 1.6, "query": "...", "trigger": "multi_intent" }
  ],
  "sub_queries": [
    "...",
    "..."
  ],
  "answer": "...",
  "citations": [
    "Doc_12 §2",
    "Doc_31 §4"
  ],
  "uncertainty": "Description of what could not be verified from the retrieved corpus, or empty/null if fully grounded."
}
```

`retrieval_events[].query` records the raw transcript chunk that triggered retrieval.
`sub_queries[]` records the actual retrieval query after optional LLM rewriting.
Deterministic corpus-vocabulary expansion is applied inside retrieval before both
dense and sparse legs, so downstream metrics should use raw event queries for
controller timing and sub-queries for retrieval debugging.

## Session refinement — patch, don't restart

When a late-arriving constraint changes the answer:
- Do NOT clear session state or re-run full-corpus retrieval.
- Recognize the modification to the existing topic.
- Dispatch targeted queries only for the delta (the new constraint), not the whole
  question again.
- Mutate only the affected claims in the stored answer; preserve everything else
  unchanged.
- Increment an answer version counter; the structured output should be able to show
  version lineage if the evaluator asks for it.

This mirrors FLARE's "detect uncertainty → retrieve targeted → regenerate only that
part" loop, repurposed to trigger on an explicit late constraint rather than
token-level confidence.
