# Commit Guide

Use small, sequential commits so each step is easy to review or share with another
developer.

## Suggested Commit Order

1. Secret/artifact exclusions and repository hygiene.
2. Provider configuration and paced, observable requests.
3. Non-destructive corpus preparation and accepted temporal test fixtures.
4. Hybrid retrieval, reranking and controller behavior.
5. Decomposition, grounded synthesis and session continuity.
6. Evaluation, structured telemetry and CLI integration.
7. Local dashboard and browser/audit tooling.
8. Bundled CPU Docker build and submission packaging.
9. Judge-facing documentation and honest, dated evidence.

Use `type(scope): concise description`, with an explanatory body where needed.
These are logical feature groups, not a fabricated historical timeline; do not
backdate commits or claim each intermediate snapshot was independently tested.
Keep accepted query/stream fixtures, but do not commit large raw corpus files,
`.cache/`, `.venv/`, `.env`, archives or nested `data/.git/`.

## Before Committing

Run:

```bash
.venv/bin/pytest
.venv/bin/prism-rag validate-data
docker compose run --rm app pytest
```

For dense retrieval smoke checks, use a separate scratch index/cache. Never run
a limited indexing command against the full submission index. Validate without
rebuilding it:

```bash
.venv/bin/prism-rag run-demo --domain cloud --corpus-limit 1000 --use-dense
```

If sharing over mail or chat, send source/config/docs first. Avoid attaching
`data/corpora/passage_level/*.jsonl`, `.cache/`, `.venv/`, or `.env`.
