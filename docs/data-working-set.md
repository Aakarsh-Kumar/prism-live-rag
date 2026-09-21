# Data Working Set

The repo is configured so code/docs can be committed without checking in the large
local corpus files or generated LanceDB indexes.

## Required Local Files

Keep these files locally under `data/`:

- `data/corpora/passage_level/cloud.jsonl`
- `data/corpora/passage_level/govt.jsonl`
- `data/mtragun-human/generation_tasks/reference.jsonl`
- `data/mtragun-human/retrieval_tasks/qrels/cloud.tsv`
- `data/mtragun-human/retrieval_tasks/qrels/govt.tsv`

The two passage-level JSONL corpus files are intentionally ignored by Git because
they are large. Share or recreate them from the cleaned IBM MTRAG-UN clone rather
than attaching them to small code-review commits.

## Query Join

MTRAG-UN does not ship a separate query file for these retrieval tasks. The loader
joins `qrels.query-id` to `reference.jsonl.task_id` and uses the final user turn in
`input` as query text.

Verified working-set coverage:

- Cloud: 86 tasks, 248 qrel passages, 72,442 corpus passages.
- Govt: 105 tasks, 256 qrel passages, 49,607 corpus passages.

## Generated Files

`prism-rag index` writes LanceDB tables under `.cache/lancedb/`. This directory is
ignored and should be regenerated locally or through Docker.

