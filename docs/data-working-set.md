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

`prism-rag index` writes LanceDB tables under `.cache/lancedb/`. Table names are
scoped by encoder (e.g. `passages__BAAI-bge-small-en-v1.5-384` for neural,
`passages__hash128` for the fallback), so both indexes can coexist. The neural
encoder also caches its model under `.cache/fastembed/`. Both directories are
ignored and should be regenerated locally or through Docker.

## Committed Stream Fixtures

`data/simulated_streams/{cloud,govt}.jsonl` are deterministic, generated fixtures
(cloud: 106 streams, govt: 125) that are **committed** so `eval --mode streaming` and
`play-stream` reproduce byte-for-byte without regenerating. Rebuild with
`prism-rag generate-streams --domain <cloud|govt>` (fixed seed); validate with
`prism-rag validate-streams`. Each line follows the `SimulatedStream` schema in
`src/prism_live_rag/stream.py` (chunks with timestamps/confidence, `stability_chunk_index`,
`settling_ms`, `sub_intents`, `qrel_passage_ids`). The `refinement` category described in
`dataset-and-augmentation.md` is not generated yet.

