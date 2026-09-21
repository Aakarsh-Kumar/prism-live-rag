# Prism Live RAG

Streaming Live RAG baseline for Samsung Theme 04. The project is built in Python
with LanceDB selected as the local vector database.

## Current Commands

Use Python 3.11-3.13. The Docker image uses Python 3.11. Host Python 3.14 is not a
supported LanceDB runtime for this project.

Install locally when `pip` is available:

```bash
python3 -m pip install -e ".[dev]"
```

Validate the narrowed Cloud/Govt working set:

```bash
PYTHONPATH=src python3 -m prism_live_rag.cli validate-data
```

Build the LanceDB passage index:

```bash
prism-rag index
```

If LanceDB hangs or times out on your host Python, run indexing in Docker:

```bash
docker compose run --rm app prism-rag index
```

Run the deterministic streaming demo:

```bash
PYTHONPATH=src python3 -m prism_live_rag.cli run-demo --domain cloud
```

After building the LanceDB index, include dense retrieval in the demo:

```bash
prism-rag run-demo --domain cloud --use-dense
```

Run tests after installing dev dependencies:

```bash
pytest
```

Run through Docker:

```bash
docker compose up --build
```

Docker does not bake the corpus into the image. `docker-compose.yml` mounts the
local `data/` directory read-only and writes indexes/caches under the mounted
`.cache/` directory as your host UID/GID.

## Environment

Preferred key names in `.env`:

```bash
GROQ_API_KEY=...
DEEPSEEK_API_KEY=...
```

The loader also accepts the dashed aliases currently used locally.

## Repository Layout

```text
src/prism_live_rag/      Python package: data loading, retrieval, controller, pipeline
tests/                   Unit/smoke tests for data joins, controller, pipeline shape
docs/                    Architecture notes, build order, approach, commit/data guides
data/                    Local cleaned MTRAG-UN working set
.cache/                  Generated LanceDB/pytest caches, ignored
.venv/                   Local virtual environment, ignored
```

Large local corpus files under `data/corpora/passage_level/*.jsonl` are ignored by
Git. See `docs/data-working-set.md` before sharing or rebuilding the dataset.
