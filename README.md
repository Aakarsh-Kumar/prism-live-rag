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

Run the provider-backed synthesis path after adding `GROQ_API_KEY` and
`DEEPSEEK_API_KEY` to `.env`:

```bash
prism-rag run-demo --domain cloud --mode provider --rewrite-query
```

Provider mode keeps the early provisional retrieval event, refreshes retrieval when
the final transcript arrives, uses DeepSeek for optional query rewriting and evidence
span extraction, then uses Groq for the final grounded answer. If a provider is
unavailable or returns malformed JSON, the command falls back to deterministic
extraction.
Provider mode makes live API calls; keep it out of tests/CI until caching and
rate-limit guards are added.

After building the LanceDB index, include dense retrieval in the demo:

```bash
prism-rag run-demo --domain cloud --use-dense
```

To test final-transcript refresh without spending API calls:

```bash
prism-rag run-demo --domain cloud --refine-on-final
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
# Optional overrides:
GROQ_MODEL=openai/gpt-oss-120b
DEEPSEEK_MODEL=deepseek-chat
PROVIDER_TIMEOUT_S=12
BM25_K1=1.2
BM25_B=0.75
```

The loader also accepts the dashed aliases currently used locally.

## Repository Layout

```text
src/prism_live_rag/      Python package: data loading, retrieval, controller, LLM steps, pipeline
tests/                   Unit/smoke tests for data joins, controller, pipeline shape
docs/                    Architecture notes, build order, approach, commit/data guides
data/                    Local cleaned MTRAG-UN working set
.cache/                  Generated LanceDB/pytest caches, ignored
.venv/                   Local virtual environment, ignored
```

Large local corpus files under `data/corpora/passage_level/*.jsonl` are ignored by
Git. See `docs/data-working-set.md` before sharing or rebuilding the dataset.
