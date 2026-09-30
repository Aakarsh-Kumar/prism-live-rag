# Build Order

## Current handoff — 30 September 2026

The original sequence below is retained as a dated development plan, not current
status. Decomposition, session state/refinement, cross-encoder reranking, JSONL
telemetry, CPU Docker packaging and the SSE dashboard are now implemented.
Rule/semantic controller and dense/hybrid/rerank ablations have saved evidence.
Default offline answer recall remains the main quality gap: 59/100 answerable
queries were uncertain or uncited in the latest 200-query CPU audit. No fresh
200-query semantic audit establishes current-release G4 acceptance.

Use [benchmark report](benchmark-evaluation-report.md),
[evaluation gates](evaluation-gates.md), [judge quickstart](judge-quickstart.md)
and [security policy](../SECURITY.md) for current verification and deployment limits.
Next priorities are answerable-query recovery without weakening grounding,
independent semantic validation, broader refinement checks and the human-recorded
video. Do not rebuild or shrink the full corpus/index as a smoke check.
The clone-and-run bootstrap now downloads pinned release assets automatically;
public availability still requires the maintainer publication step documented in
`release-publication.md`. Judges only clone, configure Cerebras and run Compose.

## Historical build sequence — 22 September 2026

## Current progress

- Runtime selected: **Python**.
- Vector database selected: **LanceDB**.
- Python package, CLI, Dockerfile, `docker-compose`, and baseline tests are
  scaffolded.
- Repo hygiene is configured for sequential commits: local secrets, venvs, caches,
  generated LanceDB indexes, nested dataset Git history, and large corpus JSONL files
  are ignored.
- Working corpus prepared under `data/`: Cloud passage corpus and qrels as the
  primary domain; Govt passage corpus and qrels as optional secondary domain.
- MTRAG-UN query text is sourced from
  `data/mtragun-human/generation_tasks/reference.jsonl` by joining
  `task_id == qrels.query-id`; Cloud and Govt coverage was verified at 100%.
- Provider-backed synthesis is available behind `prism-rag run-demo --mode
  provider`: DeepSeek handles optional query rewriting and evidence span extraction,
  Groq handles final grounded answer generation, and deterministic extraction remains
  the fallback.
- Final-transcript refresh is wired into provider mode by default: the system keeps
  the early provisional retrieval event for G2, then refreshes retrieval on the final
  ASR result before answer synthesis.
- Query expansion is factored as a retrieval component and applied before both dense
  and sparse retrieval, so weighted RRF fuses comparable candidate pools.
- Evaluation harness v1 is available via `prism-rag eval`: retrieval mode measures
  recall@k/MRR/early-retrieval rate without API calls, and provider mode runs a
  bounded Groq+DeepSeek smoke eval with citation validity, qrel citation hit rate,
  abstention rate, and latency.
- Streaming foundation (Day 1–2, committed `47881d1` + `fff2296`): `stream.py` is a
  rigorous simulator with irregular word delivery, mid-stream ASR revisions,
  LocalAgreement-n stability labels, per-stream settling time, and `early_retrieval` /
  `multi_intent` / `no_retrieval` generators. Generated streams are committed under
  `data/simulated_streams/` (106 cloud / 125 govt).
- The rule-based controller uses intent stability (LocalAgreement-n) with
  presentation-only suppression and a logged reason per decision; the pipeline records
  `decisions[]`. A semantic controller pair was added alongside it:
  `SemanticRetrievalController` (in `controller.py`) and the layered
  `ProductionSemanticController` (in `controller_v2.py`, circuit-breaker-wrapped
  embedding calls, validation, stability metrics). Both are accepted by
  `pipeline.py` and `evaluation.py`, but the rule-based one remains the default
  controller for demos and eval.
- Neural dense embeddings are now a first-class backend: `--embedding-backend
  auto|hash|fastembed` with `--embedding-device auto|cpu|cuda` and a tunable
  `--embedding-batch-size` / `--embedding-fixed-length` (or `EMBEDDING_*` env
  vars). GPU builds use `onnxruntime-gpu[cuda,cudnn]` via `scripts/setup_embeddings.sh`;
  encoding uses length-bucketed batching, a fixed sequence length on GPU for
  stable CUDA memory, and an adaptive batch-halving retry on out-of-memory. The
  index build reports progress and the encoder/device in its output JSON.
- `prism-rag eval --mode streaming` reports early-retrieval rate, false-trigger rate,
  and settling time; `prism-rag play-stream` renders a full stream trace. Measured:
  early-retrieval 0.97 cloud / 0.95 govt, false-trigger 0.0 both (eligible ≥5 words).
- `scripts/run_demo.sh`, `scripts/g1_smoke.sh`, `scripts/fetch-corpus.py`, and a pinned
  `requirements.lock` provide the one-command G1 path.

## Known gaps (audited 2026-09-22)

- The G2 streaming metric scores *any provisional retrieval before the final chunk*,
  not the stricter "fired at or before `stability_chunk_index`" from the plan. The
  looser definition is the gate-aligned one; the stricter rate is not yet computed.
- `simulate_chunks` (legacy word-prefix) is still used by `run-demo` and by
  `evaluation._first_retrieval_is_early`; only `play-stream` consumes real streams.
- `multi_intent` streams are generated and validated but never evaluated — no G3
  decomposer and no G3 metric exist.
- `sub_queries` is the single rewritten query, not decomposer output.
- No `refinement`-category streams exist, and `RagResponse` has no `version` /
  `applied_delta`, so G5 has neither data nor pipeline support.
- Telemetry is partial: `decisions[]` carries timestamps and reasons, but there is no
  per-stage latency, no token usage (the provider client discards it), and no trace log.
- `SemanticRetrievalController` and `ProductionSemanticController` are written but
  not yet benchmarked against the rule-based controller (`eval --mode streaming`
  still defaults to the rule-based one); the rule-vs-model/rule-vs-semantic
  controller ablation is still open.

## Next build sequence

1. Python repo scaffolding, `docker-compose`, one-command run script — complete for
   the deterministic baseline; keep hardening for [Gate G1](evaluation-gates.md).
   **Status: done** (`.env` optional, `requirements.lock`, `run_demo.sh`,
   `g1_smoke.sh`).
2. Transcript chunking simulator (own stub data; swap in real transcripts later) —
   base format on [dataset-and-augmentation.md](dataset-and-augmentation.md).
   **Status: done** (`stream.py`, Day 1b).
3. Data loader for the cleaned MTRAG-UN working set: Cloud first, Govt optional.
   **Status: done** (`data.py`).
4. LanceDB ingestion for passage-level Cloud/Govt corpus files — initial CLI exists;
   run it under Python 3.11-3.13 or Docker, then pass `--use-dense` in demos.
   **Status: done** (encoder-based, GPU/CPU, batch & fixed-length tunable).
5. Hybrid retrieval + weighted RRF, fixed-window chunking — initial deterministic
   dense+sparse baseline exists; see
   [retrieval.md](retrieval.md) for exact config to start from.
   **Status: done** (`retrieval.py`, BM25 + expansion + weighted RRF).
6. Rule-based Retrieval Controller (heuristic Wait/Retrieve/Suppress). **Provider
   choice follows the latency-first live-path routing rule in
   [llm-providers.md](llm-providers.md).**
   **Status: done** (intent-stability, Day 2; semantic variants in
   `controller.py`/`controller_v2.py` available but not the default).
7. Multi-Intent Decomposer (LLM-prompted, capped at 2–4 sub-queries). Use the same
   latency-first live-path routing rule.
   **Status: NOT STARTED** — no `decomposer.py`; `sub_queries` is the single rewrite.
8. Evidence span extraction + grounded generation + deterministic fallback — initial
   provider-backed path exists; continue hardening prompts, telemetry, and streaming
   token delivery. See
   [generation-grounding.md](generation-grounding.md) and [prompts.md](prompts.md).
   **Use stronger/slower models for evidence extraction when needed, and fast
   streaming models for final answer delivery.**
   **Status: partial** — extraction/grounding done; telemetry and token delivery open.
9. Model-based Retrieval Controller (classifier) — for the required rule-based vs.
   model-based ablation.
   **Status: NOT STARTED.**
10. Late-refinement patch logic (delta update, not restart) — final-transcript
    refresh exists for single-turn simulated streaming; next expand to multi-turn
    late-constraint patches.
    **Status: NOT STARTED** — `refine_on_final` refreshes retrieval, but there is no
    session object, delta patch, or `version` counter.
11. Evaluation harness — v1 exists for G2/G4 smoke metrics; next expand to G3/G5,
    dense-vs-hybrid ablations, and saved report artifacts.
    **Status: partial** — streaming mode added; G3/G5 metrics and ablation artifacts
    open.
12. Telemetry/observability — structured response fields are present, but full logs
    and token-cost tracing are still open; complete this alongside the model-based
    controller/eval harness, not as a final pass.
    **Status: partial** — `decisions[]` only; no per-stage latency, token usage, or
    trace log.
