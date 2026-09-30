# Build plan for coding agent — Plan B: "the stream is the sell" (2026-09-22)

## Current handoff — 30 September 2026

The ledger and plan below are historical; their unchecked boxes and “not built”
claims are not current status. The implementation now includes decomposition,
hybrid/rerank retrieval, query-selective Cerebras calls, evidence checks, versioned
session refinement, exact-ID JSONL trace validation, the SSE dashboard and bundled
CPU Docker assets. See the benchmark report and judge quickstart for saved evidence.

The latest full suite passed 275 tests with two skips; subsequent dashboard-security
and packaging verification passed 36 focused tests. CPU replay completed 200 unique
queries and 200 valid traces, with 118/130 actual pre-final searches. Offline answer
quality remains open: 59/100 answerable queries were uncertain or uncited. Do not
declare all gates passed from historical faithfulness means. The final image's
two-query offline check predates the subsequent HTTP hardening; rebuilding packages
is necessary to ship later source changes in binary form.

Provider requests use Cerebras GPT-OSS-120B at five requests/minute. Preserve saved
checkpoints and never rerun completed provider rows just to fill a progress counter.
The human records/uploads the demo video and decides when to push. Commits are now
explicitly authorized; no history rewrite or push is authorized.

## Progress ledger (reviewer, keep current)

- ✅ Day 1a — G1 reproducibility: `.env` optional in compose, `scripts/fetch-corpus.py`,
  `scripts/run_demo.sh`, `scripts/g1_smoke.sh`, `requirements.lock` (direct deps pinned),
  exec bits set. Verified: corpus URLs resolve (200/zip), lockfile installs, tests green.
- ✅ Day 1b — `src/prism_live_rag/stream.py`: TERTiUS-faithful simulator with
  mid-stream revisions, LocalAgreement-n stability labels, settling-time, and
  `early_retrieval` / `multi_intent` / `no_retrieval` generators. `generate-streams`
  and `validate-streams` CLIs. Streams committed at `data/simulated_streams/` (360K).
- ✅ Day 2 — controller reworked to intent-stability (LocalAgreement-n), presentation-only
  suppression, `last_reason` logging, `reset()`; pipeline records `decisions[]`;
  `eval --mode streaming` added. **Measured: early-retrieval 0.97 cloud / 0.95 govt,
  false-trigger 0.0 both** (eligible ≥5 words). `play-stream` demo command added.
- ✅ Day 2+ — encoder-made-pluggable: `--embedding-backend auto|hash|fastembed`,
  `--embedding-device auto|cpu|cuda`, `--embedding-batch-size`,
  `--embedding-fixed-length` (commits `16a361a`, `b579035`, `1eb108e`, `aa66f09`);
  then GPU support on top (uncommitted): `onnxruntime.preload_dlls()`, CUDA provider
  arena config, length-bucketed batching, adaptive batch-halving on OOM, and
  `scripts/setup_embeddings.sh`. A full 122,049-passage GPU build completes (fixed
 -length padding keeps CUDA memory stable on the 6 GB card).
- ✅ Day 2 controller variants (uncommitted): `SemanticRetrievalController`
  (controller.py) and `ProductionSemanticController` (controller_v2.py) wired into
  pipeline/eval types; rule-based remains the default. **Not yet benchmarked.**
- ✅ Committed and pushed: `47881d1` (stream simulator) and `fff2296` (controller +
  streaming eval + stream CLI) on `origin/master`.
- ⬜ Day 3 — decomposer (G3), session refinement (G5), telemetry (G6).
- ⬜ Day 4 — model controller + demo script + ablations + claim check.
- ⬜ Day 5 — report + polish + video + push.

**Resume point for the writer:** start at Day 3 below. The streaming foundation is done
and tested; do not rework `stream.py` / `controller.py` unless the reviewer flags a bug.

---

## Reality check (audited 2026-09-22, code-level)

Days 1–2 are genuinely done and tested (33 tests pass). Days 3–5 are untouched. The
audit also found five places where the ledger above was ahead of the code. Record them
here so the report and README do not overclaim:

1. **The G2 number measures a looser thing than Day 2b specified.**
   `evaluate_streaming` scores an "early hit" as *any* provisional retrieval event
   (`evaluation.py:162`), i.e. "fired before the final chunk". Day 2b asked for "first
   `Retrieve` fired at or before `stability_chunk_index`". The 0.97/0.95 figures are the
   looser, gate-aligned definition ("before final transcript completion", per
   `evaluation-gates.md` G2). **Decision:** document the looser definition as the
   primary G2 metric, and add the stricter before-stability rate as a *secondary* number
   in the report so the distinction is visible, not hidden.
2. **`simulate_chunks` is still live in two paths.** `cmd_run_demo` (`cli.py:81`) and
   `evaluation._first_retrieval_is_early` (`evaluation.py:199`) still use the legacy
   word-prefix simulator. Day 1b said to switch these to streams. The demo path now has
   a stream-based alternative (`play-stream`), but `run-demo` itself was not migrated.
3. **`multi_intent` streams are generated but never evaluated.** `eval --mode streaming`
   scores only `early_retrieval` (eligible) and `no_retrieval` (false-trigger). There is
   no G3 metric anywhere, and no `decomposer.py`.
4. **`sub_queries` is the single query-rewrite, not a decomposer.** `pipeline.py:59`
   appends one rewritten query. The architecture schema implies decomposition output.
5. **No `refinement`-category streams exist**, although the Day 1b stream schema lists
   the category. G5 has no data and no pipeline support.

Additional code-level facts the remaining work depends on:
- `RagResponse` (`models.py:40`) has no `version`, `applied_delta`, `latency_ms`, or
  `token_usage` fields.
- `providers.OpenAICompatibleChatClient.chat` (`providers.py:34`) returns `str` only —
  token usage is discarded at the source, so G6 requires a signature change.
- `StreamingRagPipeline.run` (`pipeline.py:31`) is stateless per call — no session
  object, so G5 needs a new entry point, not just a field.
- `prompts.md` has no Multi-Intent Decomposer template and no model-controller
  (Wait/Retrieve/Suppress) template.

### File-level remaining work (Days 3–5)

**Day 3a — decomposer (G3).** New `src/prism_live_rag/decomposer.py`: one provider call
returning `{"sub_queries": [...]}`, capped 2–4, near-identical dedup, deterministic
fallback `[query]`. Wire into `pipeline.run` after query rewrite; populate
`RagResponse.sub_queries` from it. Add `data-eval/compound.tsv` (labeled sub-intent
counts), `eval --mode decomposition` (target ≥0.70), `tests/test_decomposer.py` (cap,
dedup, fallback, compound sample). Prompt template: add to `prompts.md`.

**Day 3b — session refinement (G5).** Add an accumulated-session entry point on
`StreamingRagPipeline` (turn id + prior `RagResponse`). On a late constraint, emit a
**delta query only**, merge only affected claims, preserve unaffected answer text
byte-for-byte. Add `version` (int, increments per merge) and `applied_delta` to
`RagResponse`/`to_dict`. Add a `refinement` stream category in `stream.py` or a
hand-authored two-turn fixture. `tests/test_pipeline.py`: assert (a) retrieval event
trigger is `refinement`, (b) unaffected text preserved, (c) version incremented.

**Day 3c — telemetry (G6).** Extend `RagResponse.to_dict` with per-stage `latency_ms`
(controller, decompose, retrieval, extract, generate) and `token_usage`. Change
`ChatClient.chat` to expose last usage (return `(content, usage)` or a `last_usage`
attribute); deterministic mode records `usage: null`. Add a JSONL trace logger
(`eval-reports/trace.jsonl`). `tests/test_telemetry.py`: every stage has a
timestamp/latency; token cost recorded on the provider path (mocked).

**Day 4a — model controller.** `src/prism_live_rag/model_controller.py`: prompt-based
`decide(chunk)` on Groq, same contract as the rule controller, deterministic fallback
when the provider is down. `tests/test_model_controller.py`; run both controllers in
`eval --mode streaming` and diff. Prompt template: add to `prompts.md`.

**Day 4b — demo script.** `scripts/make_demo.py`: plays a curated stream with a
mid-stream revision **and** a late constraint, printing partials → decisions →
pre-final retrieval → revision → re-ground without restart → answer + citations +
telemetry. This is the video script.

**Day 4c — ablations.** `eval --mode retrieval` with `--use-dense` on/off →
`eval-reports/ablation_retrieval.tsv`; rule vs model controller in `--mode streaming`
→ `eval-reports/ablation_controller.tsv`.

**Day 4d — G4 evidence.** Sample ~20 provider answers, check each factual assertion is
entailed by a cited passage, save `eval-reports/grounding.json` (cite-valid,
claim-cover, hallucination rate). No full RAGChecker pipeline.

**Day 5a — report.** `docs/evaluation-report.md`: per gate, method → metric → result vs
target → known failure mode, plus both ablations, settling-time stats, and a
"how to reproduce" commands section.

**Day 5b — polish.** README: four judge commands, AI-use disclosure, no token-streaming
claim. Verify `docker compose up` on a clean volume. Record the video. Tag `v1.0`.

---

## Deadlines and submission format (read first)

- **Submission deadline:** 2026-09-27 end of day.
- **What is graded:** (1) one demo video, (2) the GitHub repo, (3) Docker setup.
- **Writer:** you (GPT 5.5) write the code. The reviewer (separate session) runs it,
  tests it, and files issues back to this doc.
- **API budget:** not a constraint. Prefer correctness and the demo story over saving
  calls, but never run provider calls inside tests/CI.

## Strategy (why we are doing this and not the alternative)

Theme 04 is "retrieves and grounds answers from a live full-duplex conversation
**before the user finishes speaking**." Every team will ship a decomposer, a
controller, and a grounding check. Almost none will have a *rigorous, demonstrated*
streaming story. That is the only place in the brief where we can be visibly better
than everyone, and it is currently our weakest point (streaming is a crude
word-prefix `simulate_chunks`).

So the build is shaped around **the demo video as the primary artifact**, with the
repo and Docker as the evidence that the demo is real and reproducible. The six gates
are the floor; the streaming narrative is the ceiling. We build toward the ceiling.

**Explicit cut-lines — do NOT build these at high quality (or at all):**
- No RAGChecker deep-dive. One sampled claim-entailment check is enough (see Day 4).
- No trained model-based controller. A cheap prompt-based drop-in satisfies the
  required ablation (see Day 4).
- No real-time ASR audio hookup. The stream is a rigorous simulator; document the
  ASR adapter as a swap point. (Decide Day 3 if a real-ASR demo is worth it.)
- No dense-vs-hybrid deep sweep. One afternoon table via the existing harness.
- No streaming *token* delivery of the answer. Out of scope for this deadline; do not
  claim it anywhere in docs/README/report.

---

## Day 1 — G1 reproducibility + streaming data foundation

### 1a. Make a clean clone actually run (G1, pass/fail, non-negotiable)

Current blockers on a fresh machine:
- `docker-compose.yml` has `env_file: [.env]`, but `.env` is gitignored → compose fails.
- `data/corpora/passage_level/{cloud,govt}.jsonl` (225MB) are gitignored → demo cannot
  run without them.

Tasks:
- [ ] Make `.env` optional in `docker-compose.yml` (compose must succeed with no `.env`;
  provider mode should then fall back to deterministic, which it already does).
- [ ] Add `scripts/fetch-corpus.py` (or `Makefile` target) that deterministically places
  the two JSONLs into `data/corpora/passage_level/`. If the source is not a public URL,
  it must read from a documented local path and fail with a clear message.
- [ ] Add `scripts/run_demo.sh`: fetch corpus (if missing) → build index → run demo.
  This is the one command the demo video shows at the start.
- [ ] Pin dependencies: commit a lockfile (`requirements.lock` or `uv lock`) so
  `pip install -e ".[dev]"` is reproducible. Keep `pyproject.toml` as the source of
  truth; add the lock as a second artifact.
- [ ] Add a G1 smoke check (not a pytest): a script that, on a fresh clone, runs
  `validate-data`, `run-demo` (deterministic), and `eval --mode retrieval` and asserts
  all exit 0.

**Acceptance:** `git clone <repo> && ./scripts/run_demo.sh` produces a grounded demo
on a machine with no `.env` and no pre-downloaded corpus.

### 1b. Replace `simulate_chunks` with a rigorous streaming simulator

This is the core differentiator. Put it in `src/prism_live_rag/stream.py` (new file).

Requirements:
- **TERTiUS-faithful format.** Emit partials as growing prefixes with irregular
  timestamps (word-length / speaking-rate driven, pauses at punctuation), ending in a
  final that **obsolesces** prior partials (a final result, not an append).
- **Revisions & noise (Tier 3).** A configurable fraction of streams must include a
  mid-stream correction — a prefix that revises earlier text (e.g. "working fire me"
  → "working from home") — plus optional hesitations/stutters. This is what forces the
  controller to Wait correctly; without it the demo looks fake.
- **LocalAgreement-n stability labels.** For every non-final chunk, compute/store
  `intent_stable: bool` = whether the last `n` consecutive partials agree on a text
  prefix (n configurable, default 2). This is the ground truth the controller is
  scored against — *not* "before the final chunk".
- **Settling-time measurement.** Record `settling_ms`: interval from first visible
  partial to intent stability. Report aggregate (median/p90) later.
- **Three gate-specific generators** built on the same primitives:
  - `early_retrieval`: single-intent streams (bulk).
  - `multi_intent`: compound utterances by concatenating 2–3 same-domain queries.
  - `no_retrieval`: presentation-only commands ("repeat that in two bullets") that
    must NOT trigger retrieval.
- **Deterministic.** Seedable RNG so the demo video and the eval reproduce byte-for-byte.

Output format: `data/simulated_streams/{cloud,govt}.jsonl`, one stream per line, schema:

```json
{
  "stream_id": "cloud-0001",
  "task_id": "task-xyz",
  "category": "early_retrieval|multi_intent|refinement|no_retrieval",
  "domain": "cloud",
  "qrel_passage_ids": ["..."],
  "stability_chunk_index": 3,
  "sub_intents": [{"text": "...", "stable_at": 2}],
  "chunks": [
    {"timestamp_s": 0.0, "text": "attached", "is_final": false, "confidence": 0.62},
    {"timestamp_s": 0.6, "text": "attached is", "is_final": false, "confidence": 0.71},
    {"timestamp_s": 2.15, "text": "attached is the draft.", "is_final": true, "confidence": 0.97}
  ]
}
```

- [ ] Add `load_streams(data_dir, domain)` to `data.py` returning typed stream objects.
- [ ] Add `prism-rag validate-streams` CLI: schema check, chunk-id monotonicity, per-
  category coverage, and that every `qrel_passage_ids` entry exists in the corpus.
- [ ] Update `evaluation.py` `_first_retrieval_is_early` (and the demo path) to consume
  streams, not `simulate_chunks`. Keep `simulate_chunks` only as a legacy fallback or
  delete it and fix callers.

**Acceptance:** `prism-rag validate-streams` passes on generated data for both domains;
`stream.py` produces a revision case whose `stability_chunk_index` is later than a
naive length-based fire.

---

## Day 2 — Controller rework (G2 early retrieval) + eval

### 2a. Intent-stability controller

Replace the hard `min_chars/min_tokens/min_confidence` thresholds in
`RuleBasedRetrievalController` with stability-aware logic:

- Fire `Retrieve` when the intent is *stable enough* per LocalAgreement-n, not merely
  long enough. Keep a low-confidence / low-agreement `Wait`.
- `No-Retrieval` for presentation-only / non-query turns (the `no_retrieval` streams).
- `Suppress` after a final that obsoletes a provisional retrieve (never act on a stale
  partial).
- **No thrash:** never re-retrieve on identical text (existing dedup must stay correct).
- Record **every** decision with reason + timestamp in the response (feeds G2 logging
  and G6 telemetry at once). Add a `decisions: [...]` field or extend
  `retrieval_events` with a `reason`.

- [ ] Keep the class named `RuleBasedRetrievalController` and the `decide(chunk)` contract
  so it stays a drop-in; add the model-based variant later on Day 4.

### 2b. Streaming eval + G2 metric

Add `prism-rag eval --mode streaming`:

- Load streams, run each through the pipeline, and score **early retrieval = first
  Retrieve fired at or before `stability_chunk_index`** (not "before the final chunk").
- Report: early-retrieval rate (target ≥0.80), false-trigger rate on `no_retrieval`
  streams (target low), settle-time distribution, and per-stream misses.
- Add `tests/test_controller.py` cases: stable-partial→Retrieve, presentation-only→
  No-Retrieval, revision→Wait-then-Retrieve, no-thrash on identical text.

**Acceptance:** `eval --mode streaming` prints early-retrieval rate ≥0.80 on the
generated cloud streams, with a low false-trigger rate on `no_retrieval`, and the unit
tests pass.

---

## Day 3 — G3 decomposer, G5 refinement, G6 telemetry (as streaming features) — ⬜ NOT STARTED

These are not standalone work — they are the evidence that the streaming story works.

### 3a. Multi-intent decomposer (G3)

- [ ] New `src/prism_live_rag/decomposer.py`: one LLM call returning
  `{"sub_queries": [...]}` capped at 2–4, dedup near-identical, deterministic fallback
  `[query]`. Wire into the pipeline after query rewrite; populate `sub_queries` from the
  decomposer (not just the single rewrite).
- [ ] Add `data-eval/compound.tsv` (labeled sub-intent counts) and
  `eval --mode decomposition` measuring sub-intent accuracy (target ≥0.70).
- [ ] `tests/test_decomposer.py`: cap, dedup, fallback, multi-intent sample.

### 3b. Session refinement (G5) — delta patch, not restart

- [ ] Pipeline accepts an accumulated session (turn id + prior `RagResponse`).
- [ ] On a late-arriving constraint: emit a **delta query only** for the new constraint,
  merge only affected claims, preserve unaffected prior answer text byte-for-byte.
- [ ] Add `version` (int, increments per merge) and `applied_delta` to `RagResponse.to_dict()`.
- [ ] `tests/test_pipeline.py` two-turn: initial answer → late constraint → assert (a) no
  full-corpus re-search (retrieval event trigger is `refinement`), (b) unaffected text
  preserved, (c) version incremented.

### 3c. Telemetry (G6)

- [ ] Extend `RagResponse.to_dict()`: per-stage `latency_ms` (controller, decompose,
  retrieval, extract, generate), `token_usage` estimate per call, every controller
  decision timestamped, answer `version`.
- [ ] Thread usage through provider clients (`chat()` returns `(content, usage)` or the
  client exposes last usage); deterministic mode records `usage: null`.
- [ ] Add a JSONL trace logger: `eval-reports/trace.jsonl` for any run.
- [ ] `tests/test_telemetry.py`: every stage has a timestamp/latency; token cost recorded
  on the provider path.

**Acceptance:** a provider run emits the full observability payload end-to-end.

---

## Day 4 — Model controller (cheap) + demo script + ablations + claim check — ⬜ NOT STARTED

### 4a. Model-based controller (satisfy the required ablation, minimally)

- [ ] `src/prism_live_rag/model_controller.py`: a prompt-based `decide(chunk)` using
  Groq (fast), same `DecideResult` contract as the rule controller; deterministic
  fallback to the rule controller when the provider is down.
- [ ] `tests/test_model_controller.py` + a run in `eval --mode streaming` comparing both.

### 4b. Demo script + video (the actual deliverable)

- [ ] `scripts/make_demo.py` (or extend the CLI) that plays back a curated stream —
  ideally one with a mid-stream revision and one late constraint — and prints a
  timestamped transcript of: arriving partials → controller decisions → retrieval
  fires **before** the final → revision → re-grounding without restart → grounded
  answer + citations + telemetry.
- [ ] The output must be screen-recordable and self-explanatory (or annotated). The
  video narrative, in order:
  1. `./scripts/run_demo.sh` on a clean-ish terminal (proves G1).
  2. Partial transcript arrives; system retrieves mid-speech.
  3. A revision ("working fire me" → "working from home") shows the system Wait, then
     re-retrieve — no false commit.
  4. A late constraint ("only for Sao Paulo") patches the answer without restart.
  5. Grounded answer + allowlisted citations + a telemetry trace.

### 4c. Ablations (both tables, cheap)

- [ ] Ablation A — hybrid vs dense-only: run `eval --mode retrieval` with `--use-dense`
  on/off → `eval-reports/ablation_retrieval.tsv`.
- [ ] Ablation B — rule vs model controller: run `eval --mode streaming` with each →
  `eval-reports/ablation_controller.tsv`.

### 4d. One sampled claim-entailment check (G4 evidence)

- [ ] For N sampled provider answers (say 20), verify each factual assertion is entailed
  by a cited passage (LLM-judge or lexical overlap). Save
  `eval-reports/grounding.json` (cite-valid, claim-cover, hallucination rate). Do NOT
  build a full RAGChecker pipeline.

**Acceptance:** `scripts/make_demo.py` output is recordable and tells the full story;
both ablation TSVs exist; `eval-reports/grounding.json` exists.

---

## Day 5 — Report + final polish + push — ⬜ NOT STARTED

### 5a. Evaluation report

- [ ] `docs/evaluation-report.md`: per gate, method → metric table → result vs target →
  known failure mode (cite `risks-and-open-problems.md` honestly). Include: G2 early-
  retrieval + false-trigger table, G3 sub-intent accuracy, G4 grounding artifact, G5
  two-turn output, G6 trace sample, both ablations, settling-time stats. Add a
  "how to reproduce everything" section (commands only).

### 5b. README + Docker + demo video final

- [ ] README: the four commands a judge will run (`run_demo.sh`, `run-demo --mode
  provider`, `eval --mode streaming`, `validate-streams`), plus the AI-use disclosure
  and a short architecture paragraph. Do NOT claim streaming token delivery.
- [ ] Docker: verify `docker compose up` runs the deterministic demo on a clean volume.
- [ ] Record the final video against the polished script.
- [ ] Tag the repo (`v1.0`), push.

**Acceptance:** a judge can, in under 10 minutes, clone, `docker compose up`, watch the
demo, and read a report whose numbers they can regenerate.

---

## Where the code stands today (verified 2026-09-22)

Green locally (33 passed; was 21 before the Day 1–2 work). Working: hybrid
retrieval (BM25 + expansion + RRF, dense via LanceDB), structural citation allowlist,
bounded provider retries (bad-JSON retry, empty-span rewrite retry), eval harness v1
(retrieval/provider modes), final-transcript refresh, the streaming simulator
(`stream.py`), the intent-stability controller, and `eval --mode streaming`.
`GROQ_MODEL`/`DEEPSEEK_MODEL` env overrides must remain (Groq model availability shifts
by date/tier).

Known honest gaps the report must reflect, not hide: streaming is simulated (not
real ASR), answer delivery is not token-streamed, recall@k ~0.42 true / success@k ~0.60
caps grounding, and abstention on genuinely unanswerable input is an unsolved problem
in the literature (~20–40% recall ceiling) — treat it as a measured limitation, not a
bug to paper over. See the Reality check above for the five code-vs-claim discrepancies
(Days 3–5 are still open).
