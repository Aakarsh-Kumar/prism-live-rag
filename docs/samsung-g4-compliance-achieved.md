# Samsung Gate G4 — Historical 50-query Evaluation

> This 28 September result does not establish current-release acceptance. Later
> 200-query audits and answerable-abstention failures are reported in
> [benchmark report](benchmark-evaluation-report.md). The filename is retained
> for existing links; it is not a declaration that all deliverables are compliant.

**Evaluation date:** September 28, 2026

**Internal verdict:** PASS on this evaluation protocol

**Provider / judge:** Cerebras `gpt-oss-120b` via its chat-completions API

## Results

The saved artifact `data/samsung_g4_report.json` contains a complete evaluation of all
50 selected queries. The set is balanced between Cloud and Govt (25 each), with 26
ANSWERABLE, 8 PARTIAL, 8 UNANSWERABLE, and 8 UNDERSPECIFIED queries. Eleven responses
(22%) were explicit abstentions and are reported separately from factual-response
claim scoring.

| Metric | Result | Threshold | Status |
|---|---:|---:|---|
| Mean claim faithfulness, factual responses (n=39) | 96.28% | ≥80% | Pass |
| Mean citation support, factual responses (n=39) | 99.49% | ≥85% | Pass |
| Per-query faithfulness ≥80% | 92.31% | descriptive | — |
| Per-query citation support ≥85% | 97.44% | descriptive | — |
| Fabricated citation IDs observed | 0 | 0 | Pass |
| Explicit abstention rate (n=50) | 22.00% | reported separately | — |

The mean faithfulness by answerability among factual responses was 96.03% for
ANSWERABLE (n=21), 92.29% for PARTIAL (n=8), 100% for UNANSWERABLE (n=4), and 100% for
UNDERSPECIFIED (n=6). Non-abstaining responses on unanswerable/underspecified queries
remain in the denominator: this G4 measurement tests grounding, not whether the system
chose the right behavior. Abstention/refinement behavior needs its own gate evaluation.

## Evaluation protocol and limits

- Dataset: 50 deterministic MTRAG queries joined to the local corpus and qrels; see
  `data/eval_dataset.jsonl`. Each row was retrieved against its own Cloud/Govt domain,
  with prior conversation turns retained where present.
- Faithfulness: RAGChecker's faithfulness metric only, with Cerebras GPT-OSS-120B as
  both claim extractor and entailment checker. The checker sees the cited passages;
  uncited retrieved passages cannot support an answer claim.
- Citation support: token-overlap heuristic evaluated against cited passages. This is
  not an independent semantic citation judge and should not be described as one.
- Citation IDs are selected from retrieved passage IDs in the grounded answer path and
  validated against that retrieved set in the dataset builder.
- Abstentions are excluded from claim scoring because they assert no factual answer;
  their count/rate is explicit above. A zero-citation factual response is scored as
  unsupported rather than silently omitted.
- RAGAS remains unavailable in this environment. No RAGAS score is claimed.
- This is an internal benchmark result using a single LLM judge and one balanced 50-row
  sample. It does not establish Samsung acceptance or replace independent review.

## Reproduce

```bash
.venv/bin/python scripts/build_eval_dataset.py
G4_EVAL_PROVIDER=cerebras .venv/bin/python scripts/run_ragchecker_eval.py
.venv/bin/python -c 'import json; print(json.load(open("data/samsung_g4_report.json"))["samsung_g4_compliant"])'
```

The evaluator reads `CEREBRAS_API_KEY` from `.env`, uses model `gpt-oss-120b`, limits
completion size, requests low reasoning effort, and spaces requests at 12.5 seconds to
stay under a 5-requests/minute limit. The official Cerebras SDK is not required; the
project uses the OpenAI-compatible chat-completions endpoint through `requests`.
