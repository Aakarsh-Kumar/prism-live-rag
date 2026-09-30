# Prompt Templates

These are adapted from published, benchmarked system prompts (SemEval-2026 Task 8
system papers). They're starting points, not final prompts — adjust for your own
corpus, domain, and model. Do not copy verbatim into a production system without
testing against your own eval set.

## Model-based Retrieval Controller (rule-vs-model ablation)

Live-path decision (stage ①): given only the partial transcript so far, decide whether
to retrieve now. Must be fast (Groq) and conservative — a false trigger on a
presentation-only turn counts against G2, it is not a free retrieval.

```
You are the retrieval controller of a live voice assistant. You see the partial
transcript of what the user is saying so far. Decide whether to search the knowledge
base NOW, wait for more speech, or not retrieve at all.

Partial transcript so far: {partial_transcript}
Previous decision: {previous_decision}

Rules:
- Retrieve only when the user's information need is stable and specific enough that a
  search would return the right passages.
- Wait when the transcript is still too short, unstable, or mid-correction.
- No-Retrieval for presentation-only or conversational turns that ask for no new
  information (e.g. "repeat that", "speak more slowly", "that's all thank you").

Output format (JSON):
{"decision": "Retrieve|Wait|No-Retrieval", "reason": "brief explanation"}
```

Deterministic fallback: if the provider is unavailable or returns malformed JSON, fall
back to `RuleBasedRetrievalController` rather than defaulting to `Retrieve`.

## Multi-Intent Decomposer (cap 2–4 sub-queries)

Stage ②. Split a compound utterance into independent, search-ready sub-queries. Do not
split a single-intent question, and do not decompose dependent multi-hop questions
(they need sequential resolution, not parallel search).

```
Break the user's request into independent search queries, at most {max_sub_queries}
and at least 1. Each sub-query must stand alone and target one distinct information
need. If the request has a single intent, return exactly one query. Do not split a
question whose parts depend on each other. Preserve names, numbers, and constraints.

User request: {query}

Output format (JSON):
{"sub_queries": ["...", "..."]}
```

Deterministic fallback: `[query]` (the single rewritten query) when the provider is
unavailable or returns malformed JSON. Dedup near-identical sub-queries before
retrieval.

## Query rewriting (standalone-query resolution)

```
Rewrite the last user question only into a standalone question that can be
understood without the conversation history. Use earlier turns only to resolve
ambiguity (pronouns, ellipsis, vague references). Output exactly one complete
question. Preserve the original intent and interrogative structure. Do not
introduce new concepts; do not merge multiple questions; do not copy the
conversation history. Remove conversational fillers.

Conversation history:
{history}

Current question: {question}

Output format (JSON):
{"class": "standalone|non-standalone", "rewritten_version": "..."}
```

## Evidence span extraction (run before generation)

```
Extract sentences from the passages below that answer the question. Copy EXACT
sentences — do not paraphrase. Include names, numbers, dates, and key facts.
Extract at most 8 sentences. Prioritize earlier passages if there's a tie.

Question: {question}
Conversation history: {history}

PASSAGE 1: {passage_1_text}
PASSAGE 2: {passage_2_text}
...

Output format (JSON):
{"extracted_spans": [{"passage_id": 1, "sentence": "exact text"}]}
```

If zero spans are extracted, do not proceed to generation — trigger the
deterministic insufficient-evidence fallback instead (see `generation-grounding.md`).

## Grounded generation (from extracted spans, not full passages)

```
Generate a natural answer using ONLY the facts below. Do not use outside knowledge.

FACTS:
{extracted_spans_as_bullet_list}

Conversation context: {history}
Question: {question}

Rules:
- Use ONLY information from FACTS above.
- Copy exact phrases for names, numbers, dates, and technical terms.
- Aim for meaningful overlap with the facts (avoid excessive paraphrasing — this
  reduces groundedness) but don't copy mechanically (this hurts naturalness).
- No hedging language (seems, possibly, maybe) unless evidence is genuinely
  ambiguous.
- No meta-phrases ("based on the documents", "according to the passages").
- Target length: {target_words} words.

Answer:
```

## Insufficient-evidence response (deterministic trigger, not model-decided)

Only call this when retrieval/extraction has already determined evidence is
insufficient — this should be a rule-based trigger, not a generation-time decision:

```
System: No information is available for this question.
Question: {question}

Write a short response (under 25 words) stating the information is not available in
the retrieved corpus. Do not speculate. Do not apologize excessively.
```

## Late-constraint refinement (patch, don't restart)

```
The user has added a new detail or constraint to a question you already answered.
Do NOT discard your previous answer or re-search everything from scratch.

Previous answer: {previous_answer}
Previous citations: {previous_citations}
New constraint from user: {new_constraint}

New targeted evidence (from a query issued only for this new constraint):
{new_extracted_spans}

Task: Update ONLY the parts of the previous answer that are affected by the new
constraint. Preserve everything else exactly as it was. State clearly what changed
and why (e.g., "the standard rule still applies, however the new constraint means...").

Updated answer:
```

## Answerability classification

```
Given the user question and retrieved passages, classify answerability:
- ANSWERABLE: passages contain sufficient information to fully answer.
- PARTIAL: passages contain some relevant information but incomplete.
- UNANSWERABLE: no relevant information in passages.

Question: {question}
Passages: {passages}

Output format (JSON):
{"class": "ANSWERABLE|PARTIAL|UNANSWERABLE", "confidence": 0.0-1.0}
```

Tune the confidence threshold for triggering a refusal on a dev set (see
`generation-grounding.md` §"Answerability / uncertainty classification threshold") —
don't hard-code an untested value.

## Judge / candidate-selection prompt (if generating multiple candidates)

Only worth implementing if your latency budget allows generating 2 candidates in
parallel (e.g., one greedy/low-temperature, one higher-temperature) and picking the
better one — skip this if every millisecond counts more than marginal quality gains.

```
Compare two candidate answers for quality against the source facts.

Question: {question}
Facts: {extracted_spans}

Candidate A: {answer_a}
Candidate B: {answer_b}

Evaluate on: faithfulness to facts, completeness, naturalness.

Output format (JSON):
{"winner": "A|B", "reason": "brief explanation"}
```
