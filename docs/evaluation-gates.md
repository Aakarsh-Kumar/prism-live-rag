# Evaluation Gates Reference

Samsung's six acceptance gates, what they require, and how to build toward each.

## G1 — Reproducibility (Pass/Fail)

Requirement: container launches via a single command on a clean machine; automated
replay suite completes without manual intervention.

- Build `docker-compose up` (or equivalent single-command runner) from day one, not
  at the end. This is infrastructure debt that compounds if deferred.
- Pin all dependency versions (lockfiles), don't rely on floating versions.

**Status: done.** `.env` is optional in `docker-compose.yml`, `requirements.lock` pins
direct deps, and `scripts/run_demo.sh` / `scripts/g1_smoke.sh` provide the single
command. Clean-volume `docker compose up` verification is still pending (Day 5b).

## G2 — Early Retrieval (≥ 80% of eligible queries)

Requirement: retrieval commences before final transcript completion on held-out
streaming prompts, maintaining low false-trigger rates on no-retrieval cases.

- This is the Retrieval Controller's core job. See `architecture.md` §1.
- Build test cases explicitly covering the no-retrieval-needed case (e.g.,
  presentation-only turns: "repeat that in two bullets") — a false trigger here counts
  against you, not just a missed early trigger.
- Log every controller decision with a timestamp and trigger type for later analysis.
- Measured against the committed streams with
  `prism-rag eval --domain <cloud|govt> --mode streaming`: early-retrieval 0.97 cloud /
  0.95 govt at a 0.0 false-trigger rate on both domains. Streams whose final
  transcript is under 5 words are excluded from the denominator (they have no
  meaningful pre-final phase); see `approach-summary.md`.
- **Definition used:** an "early hit" is a provisional retrieval fired before the final
  chunk. The stricter variant — fired at or before `stability_chunk_index` — is not yet
  computed and must not be implied by the numbers above.
- **Status:** metric and controller implemented; secondary before-stability rate and
  `refinement`-category coverage still open.

## G3 — Multi-Intent Identification (≥ 70% of compound queries)

Requirement: accurately identifies and isolates at least distinct sub-intents in
compound test utterances.

- Build a labeled set of your own compound test utterances (with known sub-intent
  counts) to self-validate against before submission — don't rely purely on
  qualitative spot-checks.
- Watch for over-fragmentation: splitting a simple question into multiple
  near-identical queries is a documented pitfall that hurts this gate, not helps it.

**Status: not started.** There is no `decomposer.py`, no `data-eval/compound.tsv`, and
no `eval --mode decomposition`. `multi_intent` streams exist but are not scored.

## G4 — Factual Grounding (≥ 85% citation support, zero fabricated/hallucinated doc IDs)

Requirement: all sampled factual assertions supported by cited corpus chunks.

**This is the highest-risk gate — see `risks-and-open-problems.md`.**

- Evidence span extraction before generation is your primary defense (see
  `generation-grounding.md`).
- Consider RAGChecker or a similar claim-level entailment checker for your own
  benchmarking report — it separates retriever-side failures (claim recall, context
  precision) from generator-side failures (faithfulness, hallucination rate), which
  you'll need to explain in your evaluation report regardless.
- Never let the citation ID be generated freely by the LLM — enforce citation IDs are
  drawn from an allowlist of actually-retrieved chunk IDs at the code level, not just
  the prompt level. This eliminates fabricated Doc_IDs structurally rather than
  hoping the model doesn't hallucinate one.

**Status: partial.** The structural allowlist is implemented (citations come only from
extracted spans over retrieved passages), so fabricated IDs are prevented by
construction. There is **no measured citation-support artifact yet** — the ≥85%
citation-support claim is unproven until `eval-reports/grounding.json` exists (Day 4d).

## G5 — Session Refinement (verified state continuity)

Requirement: late-arriving constraints narrow or update existing responses without
clearing session state or re-executing full-corpus search.

- See `architecture.md` and `generation-grounding.md`, both §"Session refinement".
- Build explicit test cases with a two-turn sequence: initial answer, then a
  late-arriving constraint, and verify (a) the system doesn't re-run a full search,
  and (b) unaffected parts of the original answer are preserved.

**Status: not started.** `refine_on_final` refreshes retrieval on the final transcript,
but there is no session object, delta-query patch, `version` counter, or
`refinement`-category stream. The behavioral spec lives in `architecture.md`.

## G6 — Telemetry & Observability (100% trace coverage)

Requirement: structured logs/metrics capturing execution timestamps, retrieval
triggers, citations, answer version lineage, and token cost.

- Build this alongside every other component, not as a final pass. See
  `architecture.md` §5 for the specific fields to log.
- The structured output JSON schema in `architecture.md` should give you most of this
  "for free" if you build it correctly from the start.

**Status: partial.** `RagResponse.decisions[]` carries a timestamp, text, decision, and
reason per chunk, but there is no per-stage latency, no token usage (the provider
client returns `str` only), no answer `version`, and no JSONL trace log.

## General evaluation tooling worth adopting

- **RAGChecker** (open-source, pip-installable) — claim-level entailment evaluation,
  splits retriever vs. generator failure attribution. Useful for the required
  Benchmarking & Evaluation Report, especially for G4.
- **MT-RAG / MTRAGEval evaluation scripts** (IBM, open-source) — retrieval (nDCG) and
  generation (harmonic-mean composite) scoring scripts for a closely related
  benchmark; realistic target score ranges to sanity-check your own numbers against,
  even though this isn't the exact benchmark you're being evaluated on.
- For the two required ablation experiments (hybrid vs. dense-only retrieval;
  rule-based vs. model-based controller), structure them as clean before/after
  comparisons with the same eval harness — this is exactly the kind of ablation table
  the SemEval papers reviewed all include, and graders will expect similarly legible
  reporting.
