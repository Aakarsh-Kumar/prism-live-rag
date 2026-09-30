# Streaming Live RAG — Plan of Record

**Status:** historical response plan, retained for development context. It does
not supersede the refreshed `evaluation-gates.md` or the current benchmark report.
Counts, estimates and remaining-work claims below describe the earlier snapshot.

---

## 1. What is solid — do not rewrite any of this

- `pytest`: 44 passed, 2 skipped.
- Data join: 86 cloud + 105 govt queries, 122,049 passages, **504/504 gold qrel IDs
  resolve.** This is the most defensible asset in the repo. Protect it.
- LanceDB index builds on GPU, 122,049 passages, no OOM.
- `docker compose up` works.
- **The streaming controller genuinely works.** Retrieval fires at 0.8s on a partial
  transcript, before the utterance ends, then refines on the final chunk. That is the
  Theme 04 headline and it demonstrably does it.
- BM25 + dense + weighted RRF: real.
- Deterministic extractive grounding: **airtight.** Every sentence of a generated answer
  is a literal substring of its cited passage. This path cannot fabricate.

1,047 of 4,946 lines (21%) are dead: `controller_v2.py` (646) and
`SemanticRetrievalController` (~400), neither ever instantiated. Cut them.

---

## 2. Read this before anything else — one bug is inflating every bad number

`scripts/build_eval_dataset.py:110-111` builds the evidence spans for every evaluated
query:

```python
first_sentence = text.split('.')[0] + '.'
```

That is the first period-delimited fragment of the passage. On this corpus that is almost
always a URL, a version number, or a navigation header. The actual spans fed to DeepSeek,
pulled from the committed `data/eval_dataset.jsonl`:

| Query | Span handed to the model |
|---|---|
| "toolchain is not available in South America" | `'[Build your own toolchain](https://cloud'` — 40 chars, link cut mid-URL |
| "What happens to people's bodies when there isn't enough atmospheric pressure?" | `'StarChild: Space Wardrobe\nStarChild: Space Wardrobe\n\nSpace Wardrobe\n\nGuess what?\n…'` |
| "Are those the only steps?" | `'Partitioning inventory for CC pipeline \n\nBy default, the Continuous…'` |

The first span is shorter than the URL it contains.

So the model was given link fragments and duplicated nav headers labelled "the supplied
facts," and correctly refused to answer. That is the source of the **70% abstention rate**
(35 of 50 in `g4_answer_analysis.json`) and very likely the source of the **0.482
faithfulness with 6/20 scoring exactly 0.0** in `data/samsung_g4_report.json`.

Two consequences:

1. **`synthesis.py` is not involved.** The eval set is built in provider mode. Do not spend
   time on its term-overlap scorer to fix abstention — it changes zero measured queries.
2. **G4 and the abstention rate are probably one bug, not two.** Both numbers are
   currently measuring span extraction, not grounding.

**The fix** is the sentence splitter `synthesis.py:8` already uses:
`re.split(r'(?<=[.!?])\s+', text)`, then rank those sentences by query-term overlap and take
the top few. About five lines. **Do this before diagnosing or building anything**, because
rows 4, 6, 8 and 9 in the sequence below all depend on numbers this bug is producing.

---

## 3. Sequence, ordered by irreversibility

A row goes early if skipping it leaves something you cannot repair later. A row goes late if
skipping it just means less built, which is partial credit.

| # | Row | Time | If you stop here you still have |
|---|---|---|---|
| 0 | Commit the 4 dirty source files; tag a baseline release | 20 min | A submission exists. Re-tag at the end. |
| 1 | Delete the `expansion.py` hardcode | 10 min | No dataset-specific rewrite in any judged commit. |
| 2 | **Fix span extraction in `build_eval_dataset.py`; rebuild the eval set; re-measure abstention and faithfulness** | 1 h | Real numbers instead of artifacts. Everything below gets easier. |
| 3 | `retrieval.py:134` — `exists()` **warns loudly, does not raise** in hybrid mode; raise only for explicit `--retrieval-leg dense` | 30 min | No silent dense-leg dropout. Hybrid on a clean machine still runs. |
| 4 | Add top-level `LICENSE`; delete `passage_level_backup/`, `controller_v2.py`, `.cache/evaluation_report.json`, `.cache/g4_citation_evaluation.json`; delete or rewrite the drifting docs | 1–1.5 h | A clean repo a judge can't catch you in a false claim. |
| 5 | Cheap bug batch: `decomposer.py:232` `complete`→`chat`; `cli.py:88-100` pass the reranker flags; `retrieval.py:19-34` per-instance encoder cache; `retrieval.py:386` use `self.rrf_k`; `retrieval.py:460` falsy-zero; `providers.py:61-65` keep `payload["usage"]` | 1–1.5 h | Several concrete depth items fixed. `providers.py` unlocks all token accounting. **If squeezed, keep only `decomposer.py` and `providers.py`, defer the rest.** |
| — | **≈4-hour cut line.** Submission exists, tagged, no compliance risk, no silent failures, dead weight gone, real numbers. | | |
| 6 | G1 hardening: corpus fetch + index build as a required, checked step; add a checksum to `fetch-corpus.py` | 2 h | A clean-machine run either works or fails loudly. Never silently becomes a different system. |
| 7 | G5: scoped refinement (§4) | 4–6 h | The brief's namesake feature works live on a named example. |
| 8 | Honest meters (§5) | 3 h | Every reported number survives direct questioning. |
| — | **≈1-day cut line.** Everything through row 8. | | |
| 9 | Telemetry: `answer_version`, `session_id`, `run_id`, `index_version`, per-stage `latency_ms`, JSONL trace | 2–3 h | Answers Samsung's explicit ask for time-to-first-token and cost per turn. |
| 10 | Controller ablation: wire `SemanticRetrievalController` as `--controller-type semantic` **if** runway remains; otherwise write the reasoned-argument version | 30 min – 3 h | A defensible answer either way (§5). |
| 11 | Demo video, brief, final `PRISM_GENAI_HACKATHON_Y2026` re-tag | 3 h | Done. |

---

## 4. G5 — refine, don't restart

### Scope

The guide's travel-reimbursement example delivers **two** constraints in the late chunk
("the trip was international **and** the booking was made after travel"). A design that
handles one delta at a time fails its own acceptance test. So the minimum viable scope is:
**detect a modification, extract N≥1 deltas, patch claims per constraint.** No cross-session
memory, no conversation trees, no third constraint type. Build for the named case.

### Reuse, don't rebuild

`RuleBasedDecomposer` already finds N distinct actionable clauses in one utterance. Run it
against the **late chunk** instead of the first. Do not build new constraint-extraction
NLP. That is a real Parsimony argument for the brief: *we did not build a second
decomposition system for refinement.*

### State

Ephemeral, in memory, one per run, never persisted — satisfies Session-Bound State.

```python
@dataclass
class Claim:
    claim_id: str
    text: str
    citations: list[str]
    version_added: int
    status: Literal["active", "superseded"]
    version_superseded: int | None = None

@dataclass
class SessionState:
    session_id: str
    last_effective_query: str
    claims: list[Claim]
    answer_version: int = 1
```

### Decision

On a chunk arriving after an initial answer:

- **High content-word overlap with the open topic + reads as a declarative fact-addition**
  ("the trip was…", "actually…", "also…") → **MODIFY**
- **Low overlap + fresh interrogative** → **NEW TOPIC**; freeze claims, start fresh
- **Mixed or unclear** → **AMBIGUOUS**

On **MODIFY**: run the decomposer on the chunk, issue one narrow query per delta clause
(not the original query), mark overridden claims `superseded`, append new claims with their
own citations, increment `answer_version`. Untouched claims carry forward unchanged — not
re-cited, not re-verified. Compose as retained text + delta text.

On **AMBIGUOUS**, or zero usable deltas, or more than 2 consecutive patches without a full
re-retrieval → **degrade to full re-retrieval and log it as
`classification: "degraded_full_retrieval"`**. A documented degradation path is stronger
evidence of maturity than pretending refinement always works.

Note the contrast with today: `pipeline.py:121-122` does a full re-retrieval and
wholesale-replaces the passage set. There is no session object, no delta query, no version
counter, no preservation of unaffected content.

---

## 5. Cut this

| Item | Verdict |
|---|---|
| `controller_v2.py` (646 lines) | **Delete.** Never instantiated. |
| `SemanticRetrievalController` (~400) | **Conditional.** Wire it at row 10 if there's runway; otherwise cut and argue it. |
| `passage_level_backup/` (225 MB, byte-identical) | **Delete.** Local hygiene only. |
| `.cache/evaluation_report.json`, `.cache/g4_citation_evaluation.json` | **Delete.** Smoke-test artifacts with invented queries. |
| `docs/evaluation-results-summary.md` | **Delete.** Claims "80% success@10 — Meets Samsung 80% threshold"; no such threshold exists in any Samsung document. Wrong dataset counts. An unevidenced BGE-small-vs-large claim. |
| `docs/build-order.md`, `docs/evaluation-gates.md`, `docs/complete-evaluation-journey.md` | **Consolidate into one status doc.** All three state `decomposer.py` doesn't exist. It does. |
| `data/scripts/evaluation/` | **Keep.** A vendored benchmark harness with its LICENSE intact reads as rigor, not as passing off someone else's work. |
| Top-level `LICENSE` | **Add.** You redistribute 225 MB of Apache-2.0 IBM data. |

---

## 6. What to report — three defensible numbers beat six shaky ones

**G2.** Denominator: all streams, not the `--max-tasks 10` default. Stop grading on "fired
before the final chunk" — grade on `stability_chunk_index`, which is already captured and
never used. Negatives: don't invent more phrases. `reference.jsonl` already carries a
`Question Type` field with **22 `Non-Question`** and **3 `Conversational`** tasks — real
non-retrievable data instead of more things you wrote for your own filter. Report
false-trigger rate and early-fire-at-stability separately.

**G3.** Fix the classification bug first (`decomposer.py:80` lists `" or "` / `" plus "` as
strong indicators; `:126` only splits on `" and "` / `" also "` / `" both "`), so you don't
measure a known code defect as a capability gap. Then 30–50 hand-curated genuinely compound
utterances — combine *related* topics naturally, not `" . "`-glued. Hit = ≥2 sub-queries
**and** each sub-query's retrieval surfaces its own gold passage. If you can't build this
properly, report G3 as **unmeasured** with the reason. That's worth more than a number built
on non-examples.

**G4.** Two separate claims, not one blend:
- Deterministic path: "N/N answers had 100% of cited spans verified as literal substrings."
  Near-tautological, 100% defensible, lead with it.
- Provider path: report the **span-verification** rate — does the cited sentence actually
  occur in the passage body, not just is the `passage_id` on the allowlist. That check does
  not exist yet; `llm_steps.py:169-171` validates the ID only. Add it, then report it.
- Re-measure after row 2. Don't quote pre-fix numbers.

**Retrieval.** `hybrid + rerank` and `dense-only` are both one flag (`--retrieval-leg`).
Run both on the same 50 tasks and publish. Known-good figures to beat: 0.4715 / 0.66
(hybrid+rerank) and 0.447 / 0.64 (dense-only). **The published 56.85% / 80.0% figures are
wrong — I re-ran them.** Update every doc that quotes them.

**Controller ablation.** If you don't run it, say: *we built two controllers; the semantic one
was never wired because we prioritised the abstention and G5 work. Here is the trade-off we
expect and here is how we would measure it.* Do not claim a measurement you didn't make.

---

## 7. Demo — 5 minutes

Everything in the deterministic path runs local once the corpus and models are cached. No
network. Anything calling Groq/DeepSeek is pre-record-only.

| Time | Segment | Mode |
|---|---|---|
| 0:00–0:30 | Problem + architecture diagram | Pre-recorded |
| 0:30–1:30 | Early retrieval: chunk arrival, Wait→Retrieve at the documented timestamps | **Live** |
| 1:30–2:30 | G5: answer v1 → late chunk with both constraints → v2, preserved + delta citations, version incrementing | **Attempt live**, rehearsed take ready to cut to |
| 2:30–3:15 | Citation traceability: spans highlighted against literal passage text | **Live** — strongest visual claim |
| 3:15–3:45 | Presentation-command suppression: prior citations reused, no new query | **Live** |
| 3:45–4:30 | Telemetry trace: latency, token cost, version lineage | Live if row 9 landed, else **cut it** |
| 4:30–5:00 | Honest scorecard + known limitations | Pre-recorded |

Run the live portion twice with **wifi off** before you present. That is the actual test.

**Don't narrate a field that doesn't exist.** There is no `retrieval_required` in
`RagResponse` — it has six fields: `retrieval_events`, `sub_queries`, `decisions`, `answer`,
`citations`, `uncertainty`.

---

## 8. Open questions — decide, don't investigate

- **Do the G1–G6 gates come from Samsung's guide?** The copy in
  `All theme guidelines.zip` is a `<## NASCA DRM FILE - VER1.00 ##>` container with no PDF
  inside it, so it can't be read. If the gate table is real, cite the section number when
  you present them. If it isn't, drop the framing — the official criteria are the five
  percentage weights. **Check this in a viewer, not a parser.**
- **Corpus manifest mismatch.** Local `cloud` is 72,442 passages; the MTRAG README says
  61,022. Local `govt` (49,607) exactly equals the README's FiQA count. Content sampling
  says the labels are right. Don't chase it — just add a checksum so a judge's fetch is
  verified.
- ~~**Corpus cleaning: explicitly out of scope for now.**~~ **Done — see
  `docs/corpus-spec.md` §3.** Implemented in `src/prism_live_rag/corpus_clean.py` at load
  time, so the on-disk files stay byte-identical to MTRAG-UN. 20,286 passages removed
  (16.6%), 504/504 gold preserved, index rebuilt to 101,763. Four rules the measurement
  forced, recorded in `corpus-spec.md` §3.1.0–3.1.4:
  1. 12 gold IDs share identical text with another gold ID, so a duplicate group containing
     two or more gold IDs must keep **every** gold copy.
  2. Gold protection is now **automatic** in `iter_passages()`. It was opt-in, and the first
     index build silently dropped those 12 (101,751 instead of 101,763).
  3. Chrome must be stripped **before** link extraction — link extraction normalises
     whitespace, which collapses the line structure chrome detection keys on.
  4. The chrome threshold was **guessed at 0.30 and that was wrong**. Swept against BM25
     recall: 0.30 cost 1.6 points of R@1 and R@10 by deleting real content. Now 0.60,
     where recall matches chrome-off. The threshold was a hyperparameter, not a constant.

  **Do not claim cleaning improved retrieval.** Measured (§3.1.0): neutral, +0.0 R@1, +0.5
  R@10 (one task), +0.0 R@100, zero gold tasks lost. It buys corpus hygiene, a smaller
  index, and an audit trail. That is the honest claim.
- **Two bugs the measurement caught that reading the code did not.** `SparseIndex` ignored
  `Passage.links`, so link extraction *destroyed* URL vocabulary instead of relocating it
  (−1.0 R@1). And `\s+` whitespace normalisation flattened every passage to one line,
  discarding paragraph structure. Both are now regression-tested. Lesson worth keeping:
  for a refactor justified by quality, measure the quality.
- **Row 2's G4 numbers are now doubly stale.** They were measured with both the broken span
  extractor and the uncleaned index. Regenerate after the span fix; do not quote 70%
  abstention or 0.482 faithfulness.
- **Full-stream consumption shipped** (prerequisite for row 7 / G5). `run()` no longer
  breaks at the first retrieval, so the final chunk is finally reached. See
  `corpus-spec.md` §11.0.
- **Answer relevance is a known open weakness.** The grounding guarantee holds, but the demo
  answer for "is the toolchain available in South America" is four topically-adjacent IBM
  sentences that never address the question. A grounded non-answer still scores zero on
  functionality (30%). Row 2's sentence-ranking helps; judge whether more is needed.
- **How long is the extension, and are you solo?** Determines whether row 10 is a 3-hour
  ablation or a paragraph of argument.
