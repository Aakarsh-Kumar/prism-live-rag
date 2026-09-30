# AGENTS.md — Streaming Live RAG (Samsung Hackathon, Theme 04)

Event-driven streaming RAG engine that retrieves and grounds answers from a live,
full-duplex conversation before the user finishes speaking, per Samsung's Theme 04
problem statement.

**Runtime / storage choices:** Python 3.11-3.13 is the implementation runtime
(Docker uses Python 3.11; do not rely on host Python 3.14 for LanceDB indexing).
LanceDB is the vector database for local retrieval indexes.

**Package manager / build commands:**

- `pip install -e ".[dev]"` — install the Python package and test tools locally.
- `bash scripts/setup_embeddings.sh` — detect a GPU and install the matching ONNX
  Runtime build (`onnxruntime-gpu[cuda,cudnn]` on NVIDIA GPUs, CPU wheel otherwise).
  `scripts/run_demo.sh` calls it automatically; the `neural-gpu` extra in
  `pyproject.toml` is the manual equivalent.
- `prism-rag validate-data` — verify Cloud/Govt corpus, qrels, and query joins.
- `prism-rag index` — build the LanceDB passage index under `.cache/lancedb`.
  The dense encoder is chosen by `--embedding-backend auto|hash|fastembed`
  (auto prefers neural when fastembed is installed) and runs on the device from
  `--embedding-device auto|cpu|cuda`. Batch size and fixed sequence length are
  tunable via `--embedding-batch-size` / `--embedding-fixed-length` (or the
  `EMBEDDING_*` env vars) — fixed-length padding keeps CUDA memory stable on
  small GPUs during full-corpus builds.
- `prism-rag run-demo --domain cloud` — run a deterministic streaming RAG demo.
- `pytest` — run the Python test suite.
- `docker compose up --build` — clone-and-run judge path: automatically fetches
  and checksum-verifies the public release pinned in `release-assets.json`, then
  reuses its corpus/index/models in a named asset volume. No local staging or
  index rebuild is required. A configured Cerebras key selects provider routing.
- `bash scripts/run_judge.sh` — wrapper for the same build/start path. The container
  is non-root; assets and telemetry use named volumes, not host corpus/cache mounts.
  The maintainer must publish the pinned release once; see `docs/release-publication.md`.

If a host `prism-rag index` fails with `Permission denied` (or LanceDB hangs at connect), a previous `docker compose run ... prism-rag index` left a root-owned `.cache/lancedb`; clear it with `sudo rm -rf .cache/lancedb` and rebuild.

Preferred `.env` names are `GROQ_API_KEY` and `DEEPSEEK_API_KEY`; offline G4
faithfulness evaluation also supports `CEREBRAS_API_KEY` with
`G4_EVAL_PROVIDER=cerebras` (model defaults to `gpt-oss-120b`). The current loader
also accepts the dashed aliases already used locally. Embedding settings
honor `EMBEDDING_BACKEND`, `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`,
`EMBEDDING_BATCH_SIZE`, `EMBEDDING_FIXED_LENGTH`, `EMBEDDING_CACHE_DIR` and
`EMBEDDING_LOCAL_FILES_ONLY` (plus dashed aliases).

Read the relevant doc below before implementing that part of the system. Each one
encodes findings from actual SemEval-2026 competition system papers, not general RAG
practice — don't substitute general knowledge where a specific doc exists.

## Docs

- [Build order](docs/build-order.md) — what to build, in what sequence
- [Approach summary](docs/approach-summary.md) — concise expert-facing summary of the technical approach and current working set
- [Data working set](docs/data-working-set.md) — local corpus files, query join, and ignored generated artifacts
- [Commit guide](docs/commit-guide.md) — suggested commit grouping and what not to share
- [Architecture](docs/architecture.md) — the four pipeline stages, structured output schema
- [Retrieval](docs/retrieval.md) — hybrid search, cross-encoder reranking policy, fusion, chunking, what NOT to build
- [Generation & grounding](docs/generation-grounding.md) — citations, abstention, session refinement
- [Prompts](docs/prompts.md) — reusable prompt templates per stage
- [Evaluation gates](docs/evaluation-gates.md) — Samsung's G1–G6, what each requires
- [Risks & open problems](docs/risks-and-open-problems.md) — what's genuinely unsolved; read before calling anything "done"
- [Dataset & augmentation](docs/dataset-and-augmentation.md) — which corpus, why, how to extend it
- [LLM providers](docs/llm-providers.md) — latency-first routing for live-path stages, strongest affordable models for offline quality stages
- [Agent handoff](docs/agent-handoff.md) — working plan and progress ledger for the build

Full research map with citations: `streaming-rag-research-map.md` (shared team
artifact, lives outside this repo).
