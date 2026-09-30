# Dataset & Augmentation Strategy — Streaming Live RAG

Compiled for Samsung Theme 04. Covers what a generic search would surface, then what
required knowing specific benchmark names, patent databases, and very recent
(days-old) hackathon work to find.

---

## 0. Current local working set

The local `data/` directory has been cleaned down to the MTRAG-UN working set needed
for this project:

- Primary domain: `data/corpora/passage_level/cloud.jsonl`
- Optional secondary domain: `data/corpora/passage_level/govt.jsonl`
- Retrieval qrels:
  `data/mtragun-human/retrieval_tasks/qrels/cloud.tsv` and
  `data/mtragun-human/retrieval_tasks/qrels/govt.tsv`
- Generation/query source:
  `data/mtragun-human/generation_tasks/reference.jsonl`

No standalone MTRAG-UN query file exists. Query text is recovered by joining
`reference.jsonl.task_id` to each qrels `query-id` and taking the final user turn in
`input`. This was validated for Cloud and Govt with 100% query coverage and 100%
qrel passage-ID coverage against the extracted passage corpora.

Banking and Telco are still "coming soon" in the cloned IBM benchmark. Do not build
against them unless those files are later added.

## 1. Primary corpus: MTRAG-UN, not plain MTRAG

Most teams researching this problem will find **MTRAG** (IBM, TACL 2025 — 110
conversations, 4 domains: ClapNQ, FiQA, Govt, Cloud). That's the training-side
benchmark. Far fewer will find its purpose-built evaluation successor:

**[MTRAG-UN](https://arxiv.org/abs/2602.23184)** ("A Benchmark for Open Challenges in
Multi-Turn RAG Conversations," IBM Research, ACL 2026 Findings — published Feb 2026,
recent enough that most generic searches for "RAG benchmark" still surface only the
older MTRAG paper).

Why this is the better base corpus for your submission, not just a reference:

- It is **explicitly built around the four phenomena in its own name** —
  UNanswerable, UNderspecified, NONstandalone questions, and UNclear responses — which
  map almost one-to-one onto Samsung's own gates: G4 (grounding/uncertainty) needs
  unanswerable examples, G2/G3 (controller, decomposition) need underspecified and
  non-standalone examples, G5 (refinement) needs the conversational drift these
  create. You are not adapting a generic QA set to fit the brief; this corpus was
  built to stress the same failure modes Samsung is grading.
- **666 tasks, 2,800+ turns, 6 domains** — the original 4 (ClapNQ, FiQA, Govt, Cloud)
  plus two new ones: **Banking and Telco**, added specifically to represent
  enterprise-deployed chatbot use cases.
- Documented finding from the paper itself: retrieval recall is measurably **lower on
  Banking and Telco specifically, due to longer and more complex documents** — meaning
  these two domains are a built-in stress test for your retrieval stage, not just
  more of the same.
- Same corpus family as MTRAG, so your retrieval/chunking pipeline doesn't need
  separate code paths if you also want to reference MTRAG for extra volume.
- Available at the same repo as MTRAG: `github.com/IBM/mt-rag-benchmark` — look for
  the `mtragun-human` README specifically, not just the top-level `mtrag-human` one
  most people stop at.

**Practical recommendation for this repo:** develop primarily against **Cloud**
because it is technical documentation and exists in the current clone. Keep **Govt**
as an optional secondary domain for distribution-shift checks. Banking and Telco
remain attractive future stress-test domains, but they are not present locally yet.

---

## 2. The gap neither MTRAG nor MTRAG-UN covers: streaming/partial-transcript structure

Both benchmarks evaluate **complete, turn-final utterances**. Neither has any notion
of a partial, still-forming transcript — the entire premise of Samsung's Gate G2
(retrieval triggered before the utterance finishes). This is genuine augmentation
territory. Three specific, non-obvious sources make the augmentation itself rigorous
rather than arbitrary:

### a) Borrow the actual data format from TERTiUS

**[TERTiUS / "Toward Interactive Dictation"](https://arxiv.org/pdf/2307.04008)** is a
real research dataset that logs streaming ASR output in exactly the format Samsung's
own problem statement example uses:

```
0:00.00: attached
0:00.30: attached is
0:00.60: attached is the
0:01.05: attached is the draft
0:02.15: Attached is the draft.   ← final result, obsoletes prior partials
```

Don't invent your own timestamped-partial format from scratch — replicate this
structure when converting MTRAG-UN's static questions into simulated streaming input.
It also documents an important detail most people miss: **a final ASR result
obsolesces all prior partials** — your Retrieval Controller's state management needs
to handle a partial being superseded, not just appended to.

### b) Use LocalAgreement-n as your stability-detection policy, not a hand-tuned heuristic

Rather than inventing an ad hoc "wait until the transcript looks stable" rule, there's
a named, citable policy from real-time speech translation research:
**LocalAgreement-n** (from CUNI-KIT's IWSLT 2022 streaming system, popularized further
by Whisper-streaming implementations) — if *n* consecutive incremental updates agree
on a text prefix, that prefix is confirmed/committed. This gives your Retrieval
Controller's "Wait" decision a principled, literature-backed trigger condition instead
of an arbitrary confidence threshold, and it's citable in your architecture brief as
prior art rather than something you made up under time pressure.

### c) A measurement framework from a hackathon that finished days ago

**[SETTLE — "When Streaming ASR Stops Changing"](https://lablab.ai/submissions/r6jyfodk7677jxyi0ih62su5)**
(built at a Qualcomm hackathon dated **September 14, 2026** — this is about as fresh
and unindexed as competitive intelligence gets; it won't show up in any established
"streaming RAG resources" list because it's a week-old hackathon submission, not a
paper). It introduces **"settling time"**: the measurable interval during which a
partial transcript is visible and plausible but still being revised, distinct from
final word-error-rate. Their headline finding: settling p99 scales ~3.5× depending on
the ASR engine's max-delay setting, and a naive system that acts on unsettled partials
can fire on a transcript before it's corrected (their demo: "working fire me"
misheard, triggering a false alarm before correcting to the right phrase).

**Why this matters for your build:** it gives you a concrete metric — settling time —
to report in your architecture brief and benchmarking report alongside G2's
early-retrieval-rate requirement. You can directly frame your Retrieval Controller's
Wait/Retrieve decision as trading off against measured settling time on your own
simulated partial-transcript data, which is a more rigorous framing than "we picked a
confidence threshold that seemed to work."

**Build status (2026-09-22).** This section's methodology is now implemented in
`src/prism_live_rag/stream.py`: TERTiUS-style growing prefixes with a final that
obsolesces prior partials, mid-stream ASR revisions, LocalAgreement-n stability labels
(`agreement_n=2`, `min_stable_words=3`), and per-stream `settling_ms`. The three
generators are `early_retrieval`, `multi_intent`, and `no_retrieval`; fixtures are
committed under `data/simulated_streams/`. The `refinement` category (recipe step 5)
is **not** generated yet, and the human "stable at this prefix" labels (step 3) are
approximated by the LocalAgreement-n computation rather than independently annotated —
a known limitation to state in the report.

---

## 3. Supporting citation for the underlying concept (patent, not a paper)

**[US Patent 10,102,851 — "Incremental utterance processing and semantic stability
determination"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10102851)**
describes an NLU module that determines intent stability from incremental ASR results
*while the user is still speaking*, and feeds that stability signal back to give the
user real-time feedback (e.g., a UI indicating the system is "ready to act"). This is
functionally your Retrieval Controller's job, described in patent-claim form years
before this became a research topic. Not something a generic AI-tool query surfaces —
patent databases aren't in most people's search habits — but useful as evidence that
the "intent stability" framing has prior art outside academia, and as a source of
implementation vocabulary (their "stability score," feedback loop to the ASR module)
you can reuse in your own architecture brief.

---

## 4. Concrete augmentation recipe

Putting the above together — the actual thing to build, not just read about:

1. **Take real questions from MTRAG-UN** (prioritize Cloud, mix in Govt for a
   secondary-domain check) as your ground-truth
   answer targets — this keeps citations and grounding checkable against real corpus
   passages, unlike a fully synthetic corpus.
2. **For each question you want to test streaming behavior on**, manually (or
   scripted, using an LLM) split it into a timestamped partial-transcript sequence in
   the TERTiUS format above — a handful of growing prefixes ending in the final
   utterance.
3. **Label each prefix** with whether a human would consider the intent
   "stable" at that point — this becomes your ground truth for evaluating the
   Retrieval Controller's Wait/Retrieve timing, and can be cross-checked against a
   LocalAgreement-n implementation as a sanity baseline.
4. **For multi-intent test cases**, construct compound utterances by combining 2–3
   real MTRAG-UN questions from the same domain/conversation into one streamed
   utterance — since the underlying facts are still real corpus content, your
   citations remain checkable.
5. **For late-refinement test cases**, take a real MTRAG-UN multi-turn conversation
   and insert an additional constraint-bearing turn mid-conversation that isn't in the
   original data — verify your system patches rather than restarts.
6. **Measure and report settling time** (per SETTLE's framing) alongside Gate G2's
   early-retrieval-rate metric in your benchmarking report — this is very unlikely to
   appear in a competing team's report, since it requires knowing SETTLE exists.

This keeps ~80% of your test corpus traceable to a real, citable, gold-labeled public
benchmark (MTRAG-UN) — so your grounding/citation scores mean something — while the
~20% you hand-augment is built on named, literature-backed methodology (TERTiUS's
format, LocalAgreement-n, settling time) rather than arbitrary invention.
