# Generation & Grounding Reference

## Non-negotiable constraints

These two were previously stated only in the root file with no detail — moved here
since they're specific to this component, not project-wide:

- **No fabricated citations, ever.** If retrieved evidence doesn't support a claim,
  the claim doesn't go in the answer. Prefer the deterministic "insufficient
  evidence" fallback below over a model-generated abstention.
- **No parametric/world knowledge in factual claims.** Every factual assertion must
  trace to a corpus chunk ID. This is a hard constraint from the problem statement,
  not a style preference — Samsung's Gate G4 checks for exactly this.

## Core principle: extract evidence before generating

Extract verbatim supporting spans from retrieved passages *before* calling the
generator, and feed the generator only those spans (not full passages). This acts as
a context bottleneck that bounds hallucination exposure — documented as the single
largest quality lever in one team's full pipeline ablation. Don't skip this step to
save a model call; it's cheaper than it looks (one extraction call, short output) and
the highest-ROI grounding intervention found in the literature reviewed.

## Deterministic fallback for insufficient evidence

If retrieval genuinely returns no relevant passages (or confidence is below your
threshold), **do not let the generator decide whether to abstain.** Hard-code the
fallback as a rule:

- Detect: retrieval returned zero passages, or top passage relevance score is below a
  tuned threshold.
- Action: skip generation entirely, return a deterministic "insufficient information"
  response.

This was the single highest-ROI intervention in the winning SemEval Task B system —
contributed roughly a third of that system's total score, for zero inference cost.
Cheaper and more reliable than hoping the generator abstains correctly on its own.

## Extractiveness shaping — cheap, deterministic grounding control

As an alternative or supplement to an LLM-based grounding checker: measure 4-gram
overlap between the generated answer and the extracted evidence spans, and apply an
asymmetric penalty function.

```
r4(answer, spans) = |4grams(answer) ∩ 4grams(spans)| / |4grams(answer)|

φ(r4):
  < 0.28        → strong penalty (under-extractive, hallucination risk)
  0.28–0.38     → reward (ideal band)
  0.38–0.50     → smaller reward (acceptable)
  > 0.50        → minimal reward (over-extractive, too verbatim/robotic)
```

Target band should be calibrated against your own reference answers (typical value
found in one study: mean 36% overlap in gold references). The penalty is deliberately
asymmetric — under-grounding is penalized more heavily than over-copying, because
hallucination is worse than mild verbatim repetition. Use this as a scoring signal to
select between candidate generations, or as an automated check in your evaluation
harness — it requires no extra model call.

## Answerability / uncertainty classification threshold

If implementing an explicit answerable / partial / unanswerable classifier ahead of
generation:
- Sweep the confidence threshold on a dev set and pick by macro-F1 across the three
  classes rather than guessing a value.
- Expect a real precision/recall trade-off: too low a threshold over-refuses (hurts
  answerable-turn recall), too high under-refuses (accepts weak evidence, hallucinates
  on genuinely unanswerable turns). A commonly landed-on value in one study was ~0.7,
  but this is corpus- and prompt-dependent — retune on your own data.
- **Do not assume this is a solved problem.** See `risks-and-open-problems.md` — even
  well-tuned systems in the closest comparable benchmark show a strong structural bias
  toward confident answering over correct abstention.

## Context sizing

- Passage count fed to the generator: more is not better. One study found a sweet
  spot around 3–5 retrieved passages for reference-grounded generation, with quality
  dropping again at 7–10 due to noise from marginally relevant content diluting the
  extracted evidence. In an end-to-end (retrieval-included) setting, fewer passages
  (~3) outperformed more, since retrieval noise compounds with generation noise.
- Conversation history in the generation prompt: 2–4 most recent turns is a reasonable
  default; including the *entire* history can slightly degrade performance by
  displacing evidence spans from the effective context window.

## Session refinement — patch, don't restart

See `architecture.md` §"Session refinement" for the behavioral spec. Implementation
approach: treat this as an uncertainty-triggered targeted-retrieval loop (FLARE's
pattern, repurposed) — detect that a new constraint changes an existing claim, issue a
targeted query for just that delta, regenerate only the affected portion of the
answer, and increment a version counter rather than discarding prior state.

## Post-processing: forbidden-phrase filtering

A cheap, high-value post-processing step: strip hedging/refusal phrases from the final
answer when evidence actually supports a full response (e.g. "I'm not sure", "it's
unclear", "I cannot say") via case-insensitive string matching. These phrases
disproportionately fire on partially-answerable turns where the generator hedges
despite available evidence, and measurably hurt naturalness/appropriateness scores.
Do NOT apply this filtering to genuinely unanswerable turns — only clean up hedging
language on turns where the deterministic/classifier answerability check has already
confirmed sufficient evidence exists.

## Prompt-iteration safety check

**Every time you revise a synthesis/generation prompt, re-test it against your
no-evidence test cases before keeping the revision.** A documented failure mode: an
LLM-driven "make this prompt sound more helpful" revision pass suppressed abstention
behavior and cost one team 10.9 points on their overall score. Optimizing for
perceived helpfulness trades directly against faithfulness — treat any prompt change
that increases "helpfulness" language as a regression risk to verify, not a free win.

## Model routing for cost control

In offline or non-streaming RAG, route cheap/fast models to extraction,
classification, and judging steps; reserve your strongest model for final answer
generation. One team achieved a validated 3× cost reduction this way for under 1
point of quality loss versus using their best model uniformly across every stage.

For this project, the final answer is part of the full-duplex streaming hot path, so
latency takes priority over raw model tier. Use the best fast streaming model that
meets the grounding bar for final generation, and spend stronger/slower model calls
on quality-heavy non-streaming work such as query rewriting and evidence span
extraction.

See `prompts.md` for adaptable prompt templates for each of these stages.
