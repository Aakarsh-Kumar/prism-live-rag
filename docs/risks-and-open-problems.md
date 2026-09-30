# Risks & Open Problems

Current release risk (30 September): default offline extraction was uncertain or
uncited on 59/100 answerable cases in the CPU audit. Historical grounding means
exclude abstentions and do not establish end-to-end answer quality. See the
[benchmark report](benchmark-evaluation-report.md) and [security policy](../SECURITY.md).
Research findings and the dated 22 September measurements below retain their
original scope; they are not a claim that later implementation work is absent.

Read this before marking any of the following "done." These are not solved problems
in the published literature — treat them as genuine engineering risk, not checkbox
items.

## 1. Abstention / uncertainty-flagging is empirically unsolved (affects Gate G4)

Two independent top-ranked systems on the closest comparable benchmark
(SemEval-2026 Task 8 / MTRAGEval) both show the same pattern:

- One system: 83.9% overall answerability accuracy, but only **21.8% recall on
  genuinely unanswerable questions** — meaning it confidently answered ~78% of
  questions it should have declined.
- Another system: scored under 0.4 on ambiguous/underspecified questions across every
  model tested (including the benchmark's winning system), versus ~0.75+ on clear
  questions.
- This structural bias toward confident answering over correct abstention held across
  every prompt variant and voting/judging rule tested — it is not a fixable prompt bug
  in any single implementation reviewed.

**Implication for this project:** don't treat "add an uncertainty flag" as a
one-prompt task. Budget real iteration time here, build explicit test cases for
ambiguous/underspecified queries early, and set realistic internal expectations —
matching the ~20–40% recall ceiling seen elsewhere would already put you in line with
state-of-the-art, not behind it. A system that reliably says "I need more information"
on genuinely ambiguous input is a strong outcome here, not a fallback.

## 2. The streaming/speculative-retrieval controller is extrapolated, not validated

Everything in `retrieval.md` and `generation-grounding.md` is validated against
**non-streaming, turn-complete conversational RAG** (the SemEval benchmark evaluates
complete utterances, not incremental transcript chunks). The core streaming
requirement — retrieval triggered before the utterance finishes, based on partial,
possibly-unstable transcript chunks — is not something any of the reviewed benchmark
systems actually had to solve.

The research this component rests on (Stream RAG's Trigger/Threads/Reflector framing,
Adaptive-RAG's complexity classifier, the tool-intent-stabilization paper's timing
bound) is theoretically sound but has not been validated end-to-end against a
grounding/generation stack as rigorous as the one described in the other reference
files.

**Implication:** this is the component most likely to need real iteration during the
hackathon, and the one where your own test cases and telemetry matter most — you
can't lean on someone else's ablation table here the way you can for retrieval fusion
or grounding. Instrument it heavily from the start so you can actually see where it's
failing.

## 3. Hyperparameters from the literature are corpus-specific

Values like RRF's `k` and `α`, extractiveness target bands, answerability confidence
thresholds, and passage-count sweet spots were all tuned on specific corpora (English,
four domains: encyclopedic, financial forum posts, government docs, technical docs).
They transferred reasonably well *across those four domains* in the papers reviewed,
but Samsung's actual corpus is unknown. Treat every specific number in the other
reference files as a starting point for your own tuning sweep, not a final answer.

## 4. Test-time distribution shift is a documented failure mode, not a hypothetical

One reviewed system's own limitations section notes that hyperparameters tuned on a
development set (lower unanswerable rate, more first-turn queries) did not transfer
cleanly to a test set with a 3× higher unanswerable rate and no first-turn "easy"
queries — and that this mismatch was the primary driver of their weakest subtask
score. If Samsung's held-out benchmark differs systematically from whatever you
develop against (more edge cases, harder turn positions, different unanswerable
rate), expect a similar gap. Mitigate by deliberately stress-testing on harder,
later-turn, higher-ambiguity examples during development rather than only on easy
first-turn cases.

## 5. Measured results and metric caveats (audited 2026-09-22)

Recorded so the evaluation report does not overclaim:

- **Retrieval is the quality ceiling.** Local `eval --mode retrieval` measures ~0.42
  recall@k and ~0.60 success@k on the MTRAG-UN working set. Grounding can only be as
  good as what retrieval surfaces; the G4 citation-support target is bounded by this
  until retrieval improves.
- **The G2 early-retrieval number is definition-sensitive.** `eval --mode streaming`
  reports 0.97 cloud / 0.95 govt, but an "early hit" is currently *any provisional
  retrieval before the final chunk*, not the stricter "before the ground-truth
  `stability_chunk_index`". Do not quote it as a before-stability result. Streams under
  5 words are excluded as ineligible.
- **False-trigger rate is 0.0 on the committed `no_retrieval` fixtures**, but those are
  ten templated phrases — this is a small, easy negative set and should not be read as
  robust false-trigger performance on natural conversational fillers.
- **Abstention remains the unsolved gate.** No measured artifact exists yet; see §1.
  The structural citation allowlist prevents fabricated IDs but does not prove claim
  support.
- **The controller is validated against simulated partials, not real ASR** (see §2).
  Settling-time and revision behavior are faithful to the cited methodology but are
  not measured against a live ASR engine.
