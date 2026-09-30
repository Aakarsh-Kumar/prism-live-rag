# System architecture brief

## Purpose and runtime boundary

The engine accepts timestamped, evolving ASR text hypotheses. Each partial arrives at an interval and may be revised; the final hypothesis supersedes earlier wording. Retrieval may start on a stable partial, while the answer is synthesized after the final hypothesis. The implementation does **not** stream answer tokens. Python 3.11–3.13 runs the pipeline; LanceDB holds the dense index, and BM25 uses the same cleaned passage corpus in memory.

## Four-stage data path

1. **Controller.** `RuleBasedRetrievalController` is the default. It waits while a partial is too short or unstable, suppresses presentation-only requests, and fires one provisional retrieval when intent stabilizes. A final hypothesis can trigger a fresh search. Decisions retain the timestamp, hypothesis, action, and reason. The semantic controllers exist as alternatives but are not the default.
2. **Decomposition.** When enabled, `MultiIntentDecomposer` produces distinct subqueries for independent intents. The rule-based version is the offline fallback; Cerebras `gpt-oss-120b` is the provider option. A compound query routes its subqueries through multi-query retrieval rather than asking a generative model to rerank documents.
3. **Retrieval.** The cleaned Cloud and Govt passages feed a BGE-small dense index and BM25 sparse search. Both receive the same deterministic query expansion. Weighted reciprocal-rank fusion combines their candidates, and an optional cross-encoder reranks the shortlist. Citation IDs are original corpus passage IDs. The loaded index has 101,763 cleaned passages; raw source files remain intact and protected qrel passages survive deduplication.
4. **Grounding and session state.** Deterministic synthesis selects evidence sentences from retrieved passages. Provider synthesis asks Cerebras to identify exact spans, verifies their text and IDs against retrieved passages, then returns those spans as an extractive answer. It does not currently make a separate free-form final-generation call. Insufficient evidence yields an explicit abstention. A session retains answer, citations, domain, and version. A late constraint searches the delta, patches affected answer text, and increments the version; a presentation-only follow-up reuses the answer without search.

## Provenance and trace schema

The judge dashboard uses stdlib HTTP and server-sent events, with browser assets
included in the Python package. A CPU Docker image bundles the corpus, existing
index, BGE-small and cross-encoder caches. No microphone adapter, model downloads
or indexing are part of runtime startup. GPU acceleration is optional locally.
The browser can continue a completed run with an explicit refinement or
presentation-only follow-up; each branch copies its parent session and keeps
version lineage rather than sharing mutable state between independent runs.
Provider evidence selection is followed by a semantic relevance filter, not
generative document reranking. Exact source text and allowlisted passage IDs
remain mandatory. CPU cold-start and query times require final-image measurement.

Every completed execution can append one JSONL trace with an execution ID and timestamp, per-chunk decisions, retrieval triggers, actual subqueries, answer, citations, answer version lineage, stage latency, provider token usage, and estimated USD inference cost. Cost is zero when no provider tokens are used. Cerebras estimates use the published `gpt-oss-120b` input/output prices recorded in the trace basis; these are estimates, not invoices. Failed executions receive an error trace with unavailable fields marked explicitly.

The raw corpus and qrels come from MTRAG-UN Cloud and Govt. Local cleaning happens while loading, so source files and IDs remain auditable. The curated streaming suite stores query labels and interval-delivered ASR partials separately. Gold passage IDs join to the corpus before a case is accepted. The G3 compound candidates were accepted by the owner; their exact intent equivalence metric remains a lexical proxy.

## Trade-offs and failure controls

- Provisional retrieval buys time but can search an incomplete intent. Final-hypothesis retrieval supersedes stale partial evidence; the controller avoids repeated retrieval on each word.
- The sparse leg improves recall for exact entities while dense search handles paraphrases. Fusion and cross-encoder reranking add compute, so each was measured as an ablation on the same query set.
- A delta patch preserves unrelated prior text when it can identify affected sentences. If a restrictive new constraint cannot be mapped safely to old text, the system omits uncertain old claims. If delta evidence is absent, the prior answer is preserved with an uncertainty flag.
- Citation IDs are selected by code from retrieved passages. The provider cannot invent a new ID in the output path. Citation support is still evaluated independently because a valid ID does not guarantee that its text entails the claim.
- The current demonstration uses simulated ASR text chunks, not microphone audio. The benchmark report distinguishes retrieval timing, retrieval relevance, grounding, and trace coverage rather than treating one score as overall quality.
