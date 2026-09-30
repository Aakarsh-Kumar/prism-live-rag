# Curated Grounding Dataset

This is an additive, corpus-derived evaluation suite. It does not replace MTRAG-UN,
and its results must never be presented as official MTRAG-UN results.

## Targets and current status

| split | answerable | partial | unanswerable | underspecified | total |
|---|---:|---:|---:|---:|---:|
| train | 300 | 90 | 120 | 90 | 600 |
| dev | 60 | 18 | 24 | 18 | 120 |
| test | 100 | 30 | 40 | 30 | 200 |

The combined review pool contains **218 cases**: 100 ANSWERABLE, 48 PARTIAL, 40
UNANSWERABLE, and 30 UNDERSPECIFIED. The extra PARTIAL cases include a Cloud-only backup
pool so reviewers were not forced to accept weak cases merely to maintain domain
coverage. The project owner reviewed and approved the 200-case shortlist on 2026-09-28.
The accepted test suite is frozen at `test.jsonl`; `manifest.json` records SHA-256
`1f5a13a6a0d6eac5bb927f0b56cc599e85fc8cb9f697647b14f1369874615e78`.

- `candidates/test-mtragun-negative.jsonl`: 40 UNANSWERABLE and 30
  UNDERSPECIFIED cases retaining the benchmark's labels and full conversation.
- `candidates/test-mtragun-positive.jsonl`: 100 ANSWERABLE and 22 PARTIAL cases
  retaining benchmark answers and gold passage IDs. Their support spans are lexical
  candidates and require human verification.
- `reviews/test-mtragun.csv`: review sheet for the first negative candidate set.
- `candidates/test-generated-partial.jsonl`: 16 DeepSeek-generated PARTIAL candidates
  whose evidence spans passed exact-substring validation.
- `candidates/test-generated-partial-cloud-backup.jsonl`: 10 additional Cloud PARTIAL
  candidates generated after the first audit exposed a domain-specific shortage.
- `candidates/test-all.jsonl` and `reviews/test-all.csv`: the unified candidate pool and
  review sheet to use for final test selection.
- `candidates/test-shortlist.jsonl`, `candidates/test-audit.json`, and
  `reviews/test-shortlist.csv`: the target-sized 200-case review set, deterministic audit
  signals, and the review sheet that should be completed.

All test cases now have review status `approved`, the exact target counts are met,
`curated-data validate` passes, and the manifest freezes the test-file hash. Results
must still be labeled as curated-suite results, never official MTRAG-UN results.

The accepted suite has a validated ASR simulation at `streams/test.jsonl`: 200
timestamped streams, each with growing partial
hypotheses plus one final result, and 29 streams containing an explicit mid-utterance
revision. The final chunk records that it supersedes every preceding partial.

The full 200-case deterministic pipeline evaluation was run on the RTX 3050 with
BGE-small CUDA embeddings, hybrid dense+sparse retrieval, and cross-encoder reranking.
It measured 83.85% pre-final retrieval, 0.77% retrieval at or before the labeled
stability point, and 65.38% qrel citation hits over answerable/partial cases. The
behavior proxy was 42.5% overall; both UNANSWERABLE abstention and UNDERSPECIFIED
clarification proxy rates were 0%. These numbers expose open controller and response-
policy failures and do not establish G4 or G5 compliance. The complete per-case report
is `evaluation/test-streaming.json`.

## Workflow

Use Python 3.11–3.13 through `.venv`.

```bash
# Authoritative benchmark candidates
prism-rag curated-data seed-benchmark --split test --kind negative
prism-rag curated-data seed-benchmark --split test --kind positive

# Generate overcomplete corpus-derived candidates. This calls DeepSeek and can be
# resumed by writing each split to a distinct output file.
prism-rag curated-data generate --split train --multiplier 2
prism-rag curated-data generate --split dev --multiplier 2
prism-rag curated-data generate --split test --multiplier 2

# A bounded generation run can fill one class without paying for the whole pool.
prism-rag curated-data generate \
  --split test --case-class partial --count 16 \
  --output data/curated_dataset/candidates/test-generated-partial.jsonl

# Optionally mine top-ranking non-gold passages. Add --use-dense only after the
# current cleaned LanceDB index has been built.
prism-rag curated-data add-distractors \
  --input data/curated_dataset/candidates/train.jsonl \
  --output data/curated_dataset/candidates/train-with-distractors.jsonl

# Human review
prism-rag curated-data validate-candidates \
  --input data/curated_dataset/candidates/test-all.jsonl
prism-rag curated-data shortlist \
  --split test \
  --input data/curated_dataset/candidates/test-all.jsonl \
  --output data/curated_dataset/candidates/test-shortlist.jsonl \
  --audit-output data/curated_dataset/candidates/test-audit.json
prism-rag curated-data export-review \
  --input data/curated_dataset/candidates/test-shortlist.jsonl \
  --output data/curated_dataset/reviews/test-shortlist.csv
prism-rag curated-data import-review \
  --input data/curated_dataset/candidates/test-shortlist.jsonl \
  --review data/curated_dataset/reviews/test-shortlist.csv \
  --output data/curated_dataset/reviewed/test.jsonl

# Convert only the current user turn into interval-based ASR partials. Prior turns stay
# in context_turns; they are not replayed as part of the current utterance.
prism-rag curated-data generate-streams \
  --split test \
  --input data/curated_dataset/candidates/test-shortlist.jsonl \
  --output data/curated_dataset/streams/test-shortlist.jsonl
prism-rag curated-data validate-streams \
  --cases data/curated_dataset/candidates/test-shortlist.jsonl \
  --streams data/curated_dataset/streams/test-shortlist.jsonl

# Freeze accepted splits only after enough approved cases exist
prism-rag curated-data finalize \
  --split test --input data/curated_dataset/reviewed/test.jsonl
prism-rag curated-data validate
prism-rag curated-data manifest

# Evaluate actual interval-delivered chunks. This reports the gate-aligned pre-final
# rate and the stricter retrieval-by-stability rate separately.
prism-rag curated-data evaluate-streams \
  --cases data/curated_dataset/test.jsonl \
  --streams data/curated_dataset/streams/test.jsonl \
  --use-dense --use-reranker \
  --output data/curated_dataset/evaluation/test-streaming.json
```

Use `curated-data merge` when benchmark and generated candidate files need one review
sheet. It rejects duplicate case IDs.

## Review rules

For every test case, a human reviewer must verify the label, query naturalness,
dialogue context, reference answer, and exact evidence spans. An approval requires a
reviewer identifier. In particular:

- ANSWERABLE answers may contain only claims entailed by the recorded spans.
- PARTIAL answers must identify both the supported part and the requested detail that
  is absent.
- UNANSWERABLE cases must warrant abstention, not merely have weak retrieval.
- UNDERSPECIFIED cases must need the recorded targeted clarification.

The test negatives are required by validation to retain MTRAG-UN provenance. Generated
negatives are training/development material only because absence from one source passage
does not prove absence from the full 101,763-passage retrieval corpus.

## Leakage and provenance

Corpus-generated cases are assigned to a split by a seeded hash of normalized source
URL (or document-prefix fallback) before generation. Validation rejects a source group
that appears in multiple accepted splits, missing passage IDs, non-exact evidence spans,
gold/distractor overlap, duplicate case IDs, incomplete class counts, and unreviewed test
cases.

The manifest records split hashes and labels the suite as curated. Continue to report
the untouched MTRAG-UN evaluation separately, including the historical 48.2% G4 result
until a frozen replacement run exists.

## Streaming invariant

The current user utterance is never supplied to the pipeline as one complete line. It is
emitted as timestamped, confidence-bearing ASR hypotheses at speaking intervals. Partials
may revise previously emitted words; the final hypothesis restores the canonical query and
obsoletes all earlier partials. Conversation history remains available as prior context but
is not concatenated into a fake single utterance.
