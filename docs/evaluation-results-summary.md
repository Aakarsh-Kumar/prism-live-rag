# Evaluation Results Summary

> Historical 28 September evaluation, not current-release gate acceptance. See
> [benchmark report](benchmark-evaluation-report.md) and
> [evaluation gates](evaluation-gates.md) for later 200-query evidence and failures.

## Historical G4 result — September 28, 2026

The historical complete G4 artifact is `data/samsung_g4_report.json`, generated from all
50 rows in `data/eval_dataset.jsonl` using Cerebras GPT-OSS-120B for RAGChecker
faithfulness evaluation.

| Measurement | Result |
|---|---:|
| Dataset | 50 queries: 25 Cloud, 25 Govt |
| Answerability mix | 26 answerable, 8 partial, 8 unanswerable, 8 underspecified |
| Factual responses scored | 39 |
| Explicit abstentions | 11 (22%) |
| Mean RAGChecker faithfulness | 96.28% (target ≥80%) |
| Mean citation support | 99.49% (target ≥85%) |
| No fabricated citation IDs | Verified by retrieved-ID allowlist |
| Internal G4 verdict | PASS |

Faithfulness is judged only against passages actually cited. Citation support is a
token-overlap heuristic against those cited passages; it is not an independent semantic
judge. Correct abstentions are excluded from claim scoring and reported separately.
Non-abstaining responses to unanswerable/underspecified cases remain in the factual
denominator, so this result measures grounding rather than abstention correctness.

See the [benchmark report](benchmark-evaluation-report.md) for current protocols
and limitations. Older results in the working
history used different samples/providers and are superseded by this artifact; do not
compare them as a controlled before/after experiment.

## Other gates

G2/G3/G5/G6 are tracked separately in [`evaluation-gates.md`](evaluation-gates.md).
This document intentionally does not restate retrieval or streaming figures from older
evaluation runs, since those require their own current artifacts and definitions.
