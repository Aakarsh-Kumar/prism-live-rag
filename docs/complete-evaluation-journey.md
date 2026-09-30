# Evaluation Journey — Samsung Theme 04

> Historical evaluation journey. The current version-specific evidence and open
> issues are in [benchmark report](benchmark-evaluation-report.md).

## Historical snapshot — September 28, 2026

G4 passed the internal 50-query protocol at that historical snapshot, recorded in
`data/samsung_g4_report.json`: 96.28% mean faithfulness and 99.49% mean citation
support across 39 factual responses. Eleven explicit abstentions (22%) are tracked
separately. The sample contains 25 Cloud and 25 Govt queries across four answerability
labels.

Faithfulness uses RAGChecker with Cerebras GPT-OSS-120B, checking only passages actually
cited. Citation support is a token-overlap heuristic over cited passages, not an
independent semantic citation judge. One LLM judge and one 50-query sample do not
establish Samsung acceptance. See `samsung-g4-compliance-achieved.md` for details and
reproduction instructions.

## Evaluation work completed

1. Restored and retained the complete 122,049-passage corpus and rebuilt its LanceDB
   index with fixed-length CUDA embedding batches.
2. Corrected G4 query selection to use only locally supported IBM Cloud and Govt
   collections, join answerable/partial examples with qrels, validate passage IDs, and
   preserve prior conversational turns.
3. Made the evaluator report the full 50-query sample and fail on provider/evaluation
   errors instead of silently producing a partial or zero-filled report.
4. Routed G4 judging to Cerebras `gpt-oss-120b`; bounded completion size/reasoning and
   paced requests at 12.5 seconds. The evaluator uses the chat-completions endpoint, so
   no Cerebras SDK install is required.
5. Narrowed RAGChecker to its faithfulness metric, which is the relevant claim-level
   G4 measurement. Other metrics that consume the same provider quota are not run here.
6. Made the faithfulness evidence scope match user-visible evidence: cited passages only.
   Explicit abstentions skip claim judging; a factual response without citations scores
   unsupported.

## G4 result

| Metric | Result | Target |
|---|---:|---:|
| Mean faithfulness | 96.28% (n=39) | ≥80% |
| Mean citation support | 99.49% (n=39) | ≥85% |
| Faithfulness pass rate per query | 92.31% | descriptive |
| Citation pass rate per query | 97.44% | descriptive |
| Explicit abstentions | 11/50 (22%) | reported separately |
| Fabricated citation IDs observed | 0 | 0 |
| Internal G4 verdict | PASS | both mean thresholds |

The 50-query answerability distribution is ANSWERABLE 26, PARTIAL 8, UNANSWERABLE 8,
UNDERSPECIFIED 8. Of 39 non-abstaining factual responses, the breakdown is 21, 8, 4,
and 6 respectively. Non-abstaining responses to negative/underspecified cases remain
in the faithfulness denominator: G4 evaluates grounding, while abstention correctness
is a separate behavior question.

## Reproduce

```bash
.venv/bin/python scripts/build_eval_dataset.py
G4_EVAL_PROVIDER=cerebras .venv/bin/python scripts/run_ragchecker_eval.py
.venv/bin/python -c 'import json; print(json.load(open("data/samsung_g4_report.json"))["samsung_g4_compliant"])'
```

## Remaining evaluation work

G4 is not a proxy for the other Samsung gates. Consult `evaluation-gates.md` for current
G1–G6 status; in particular, multi-intent identification, session refinement, and
complete telemetry still require their own evidence.
