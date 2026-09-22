# Approach Summary

## Problem Framing

We are building a Streaming Live RAG engine for Samsung Theme 04: a system that can
listen to a live, full-duplex conversation, decide when the user's intent is stable
enough to retrieve, and start grounding an answer before the user has fully finished
speaking. The core challenge is not just retrieval quality; it is balancing early
retrieval against the risk of acting on unstable partial transcripts.

## Technical Stack

- **Runtime:** Python 3.11-3.13, with Docker pinned to Python 3.11.
- **Vector database:** LanceDB for local persistent vector indexes.
- **Entrypoints:** `prism-rag validate-data`, `prism-rag index`, and
  `prism-rag run-demo --domain cloud`.
- **Container path:** `docker compose up --build` runs the demo service.
- **Primary corpus:** IBM MTRAG-UN Cloud domain, using passage-level technical
  documentation.
- **Secondary corpus:** MTRAG-UN Govt domain for optional distribution-shift testing.
- **Evaluation focus:** Samsung gates G1-G6, especially early retrieval, factual
  grounding, session refinement, and observability.

## Dataset Strategy

The local `data/` directory has been cleaned to the working set needed for the
pipeline. The large passage-level corpus files are kept locally but ignored by Git:

- `data/corpora/passage_level/cloud.jsonl`
- `data/corpora/passage_level/govt.jsonl`
- `data/mtragun-human/retrieval_tasks/qrels/cloud.tsv`
- `data/mtragun-human/retrieval_tasks/qrels/govt.tsv`
- `data/mtragun-human/generation_tasks/reference.jsonl`

MTRAG-UN does not provide a separate retrieval query file. We join qrels
`query-id` to `reference.jsonl.task_id` and use the final user turn as query text.
This join has been verified for Cloud and Govt, including passage-ID coverage against
the extracted passage corpora.

## Pipeline Approach

The pipeline has four main stages. Stages marked **[planned]** are specified here and
in `architecture.md` but are not yet implemented in code as of the 2026-09-22 audit.

1. **[implemented] Retrieval Controller:** watches incremental transcript chunks and
   decides `Wait`, `Retrieve`, or `No-Retrieval` based on intent stability. The
   rule-based controller uses LocalAgreement-n: it fires only once the last `n`
   partials agree on a stable word prefix (`agreement_n=2`, `min_stable_words=3`), and
   suppresses presentation-only turns (e.g. "repeat that in two bullets"). Every
   decision is logged with a timestamp and a reason.
2. **[planned] Multi-Intent Decomposer:** splits compound utterances into 2-4
   independent search queries when needed. Not yet built — there is no
   `decomposer.py`, and `RagResponse.sub_queries` is currently populated by the single
   query-rewrite step, not by decomposition.
3. **[implemented] Corpus Retrieval & Fusion:** uses hybrid dense+sparse retrieval,
   LanceDB-backed dense search, weighted/nested RRF fusion, and optional cross-encoder
   reranking when latency allows.
4. **[implemented] Session-Aware Synthesis:** extracts evidence spans before
   generation, enforces citation grounding, and applies a deterministic
   insufficient-evidence fallback. **[planned]** The session-refinement clause (patch
   late constraints instead of restarting) is not yet implemented: `RagResponse` has no
   `version` or `applied_delta`, and the pipeline has no session object.

The current implementation has two execution modes. The default deterministic mode
uses hash embeddings for initial local dense indexing, in-memory sparse scoring,
weighted RRF fusion, a rule-based streaming controller, and extraction-based
synthesis. Provider mode is enabled with `prism-rag run-demo --mode provider
--rewrite-query`: DeepSeek rewrites unstable utterances and extracts evidence spans,
while Groq generates the final grounded answer from those spans. Provider failures
or malformed JSON fall back to deterministic extraction, so local demos still run
without spending API calls. Provider mode also refreshes retrieval on the final ASR
transcript: the provisional event is preserved for early-retrieval measurement, but
the final answer uses evidence from the more complete query when available.
The current Groq default is `openai/gpt-oss-120b`, with `GROQ_MODEL` available as an
override because Groq model availability changes by date and account tier.
Provider calls use `PROVIDER_TIMEOUT_S` (default 12s) and are intended for demos, not
CI loops, until caching and rate-limit guards are added.

Local note: LanceDB indexing should be run under the supported Python range or via
Docker. Host Python 3.14 is currently not treated as a supported LanceDB runtime.

## Key Design Choices

We allow cross-encoder reranking, but ban generative LLM-as-judge reranking because
the reviewed SemEval systems found it slower, costlier, and lower quality than RRF
on the same candidate pool.

Query expansion is an explicit retrieval component, not hidden reranking. The first
staging rule maps region wording such as "South America" to IBM Cloud corpus tokens
such as `sao paulo` and `br-sao`, then feeds the expanded query to both dense and
sparse retrieval legs before weighted RRF.

Evaluation is exposed as a first-class CLI path. `prism-rag eval --mode retrieval`
runs offline recall@k, MRR, and early-retrieval measurements. `--mode provider`
executes bounded Groq+DeepSeek smoke tests and reports citation validity,
qrel-citation hits, abstentions, and latency so the real deliverable path is measured
instead of only mocked.

## Streaming Simulation & Evaluation

The live path is driven by a rigorous simulator rather than a naive word-prefix
split. `src/prism_live_rag/stream.py` reproduces the conditions the controller must
survive in real speech: irregular word-delivery timestamps, mid-stream ASR revisions
that correct an earlier partial (e.g. `rteain` → `retain`), LocalAgreement-n stability
labels, and a per-stream settling time. It generates three categories per domain —
`early_retrieval`, `multi_intent`, and `no_retrieval` — and the generated streams are
committed at `data/simulated_streams/{cloud,govt}.jsonl` so evaluation and the demo are
reproducible without regenerating.

- `prism-rag generate-streams --domain cloud` rebuilds the streams deterministically.
- `prism-rag validate-streams --domain cloud` checks their invariants.
- `prism-rag eval --domain cloud --mode streaming` reports early-retrieval rate,
  false-trigger rate, and settling-time distribution.
- `prism-rag play-stream --domain cloud --stream-id <id>` renders the full trace:
  each partial, the controller decision and reason, the revision, and the final
  grounded answer with citations.

Because several MTRAG-UN queries are only 1–4 words, they cannot have a meaningful
"before the user finishes" phase; the streaming metric therefore reports the rate over
**eligible** streams (final word count ≥ 5) and excludes them from the denominator
rather than inflating the score. Measured on the committed streams: early-retrieval
0.97 cloud / 0.95 govt at a 0.0 false-trigger rate on both domains.

**Metric definition, stated precisely.** The reported early-retrieval rate counts any
stream where the controller fired a *provisional* retrieval before the final chunk. It
does **not** yet measure the stricter condition "fired at or before
`stability_chunk_index`" (the ground-truth intent-stability point). The looser
definition matches Gate G2's wording ("retrieval commences before final transcript
completion"); the stricter rate is a planned secondary metric. Do not quote the
0.97/0.95 numbers as a before-stability result.

`multi_intent` streams are generated and validated but are not yet scored by any
metric — the G3 decomposer and its evaluation are still to be built. The legacy
`simulate_chunks` word-prefix helper is still used by `run-demo` and by
`eval --mode retrieval`; only `play-stream` and `eval --mode streaming` consume the
real streams.

The live path is latency-first. The controller, decomposer, and final streamed answer
should use the fastest acceptable provider to minimize time-to-first-token. Slower,
stronger models can be reserved for non-streaming quality work such as query
rewriting and evidence span extraction.

Grounding is treated as a structural constraint, not a prompt preference. The system
must only cite retrieved chunk IDs, avoid parametric factual claims, and return a
deterministic insufficient-evidence response when supporting passages are missing.
