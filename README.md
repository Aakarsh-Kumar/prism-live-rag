# Prism Live RAG

Event-driven streaming retrieval-augmented generation for Samsung Theme 04.
Timestamped text chunks simulate evolving ASR hypotheses: the controller can
retrieve before the user finishes speaking, while the final answer is grounded
in retrieved Cloud/Govt passages.

## Run the judge dashboard

The supplied Linux AMD64 Docker image includes the corpus, LanceDB index and CPU
models. No GPU, API key or model download is required at runtime.

1. Extract the source bundle and place `prism-live-rag-judge.tar.gz` in its root.
2. Start Docker and run:

```bash
bash scripts/run_judge.sh
```

Open **http://localhost:8080**. Allow at least 6 GB RAM for Docker; startup took
about 24–27 seconds on the development PC, excluding image loading. If port 8080
is already occupied, use `PRISM_PORT=8081 bash scripts/run_judge.sh`.

For a source build, extract the separate asset bundle into `submission-assets/`
first, then run `docker compose up --build -d`. A bare Git clone does not contain
the large corpus, index or model files. See [judge quickstart](docs/judge-quickstart.md)
for offline loading, asset preparation and the optional unattended 200-query replay.

The dashboard provides all 200 accepted test queries, filters and search, streamed
partials/revisions, controller reasons, actual search dispatches, returned passages
and ranks, citations, provider-call events, latency/cost fields and raw traces.
Completed answers can be followed by a presentation request or a delta refinement.
“Multi-intent” filtering is a rule preview, not a human gold label.

## Design

1. Rule-based streaming controller: Wait / Retrieve / No-Retrieval on each chunk;
   semantic controllers remain opt-in.
2. Bounded multi-intent decomposition with rule and optional provider paths.
3. BGE-small dense retrieval in LanceDB plus BM25, weighted/nested RRF fusion,
   and optional cross-encoder reranking. A generative LLM is not the reranker.
4. Evidence-based synthesis, allowlisted citation IDs, abstention and versioned
   session refinement.

See the [architecture brief](docs/system-architecture-brief.md),
[retrieval reference](docs/retrieval.md) and [telemetry schema](docs/telemetry-schema.md).

## Verification and honest limits

The release verification recorded **275 tests passed, two skipped**. Subsequent
local-request security and packaging checks passed **36 focused tests**; the full
suite was not rerun after that hardening.

The completed CPU dashboard audit contains **200 unique executions and 200 valid
traces**, zero invalid citation IDs, and actual pre-final search in **118/130
eligible cases (90.77%)**. The final image also completed two queries with Docker
networking disabled and no host asset mounts, keys or GPU.

These are execution/timing results, not an overall answer-quality pass:

- Default offline extraction was uncertain or uncited on **59/100 answerable
  cases**. Three of 40 unanswerable cases returned citations. Quality remains open.
- Historical provider faithfulness scores exclude abstentions and refer to earlier
  runtime snapshots. They do not establish that the latest release clears every gate.
- Input is simulated ASR text, not live audio. Answers arrive whole, not token-streamed.
- Session refinement has controlled tests and targeted corpus checks, not broad
  semantic accuracy validation.
- Linux AMD64 was exercised; ARM64 and arbitrary judge hardware were not verified.

The [benchmark report](docs/benchmark-evaluation-report.md) separates each protocol,
runtime version, denominator and failure. Raw generated evidence and checksums ship
in submission bundles rather than Git history. The human records the final video
using the [recording script](docs/demo-video-script.md).

## Optional provider mode

Copy `.env.example` to `.env` and set `CEREBRAS_API_KEY` locally.
The dashboard uses `CEREBRAS_MODEL=gpt-oss-120b`, paced at five requests/minute.
Routing determines whether a query needs a provider call; not every query does.
Keys stay server-side. The default is provider-free; provider mode is quota-dependent
and slower. Groq/DeepSeek remain available for legacy CLI paths.

## Local development

Use Python **3.11–3.13**, not host Python 3.14.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/prism-rag validate-data
.venv/bin/prism-rag serve --port 8080
```

Local execution requires the corpus described in
[data working set](docs/data-working-set.md). Neural retrieval additionally requires
model files and a matching index. `bash scripts/setup_embeddings.sh` selects ONNX
Runtime for the available GPU/CPU. `prism-rag index` builds an index; it is not a
startup step for the supplied image. Do not replace the full submission index with
a small smoke-test index.

```bash
.venv/bin/python -m pytest
.venv/bin/prism-rag run-demo --domain cloud
.venv/bin/prism-rag eval --domain cloud --mode retrieval --max-tasks 50
.venv/bin/prism-rag eval --domain cloud --mode streaming --max-tasks 200
```

Real provider evaluations are explicit and separate from unit tests/CI.

## Security and data handling

This is a **local demonstration server, not a public production service**.
Compose publishes only on loopback. Do not expose it through a public tunnel:
there is no authentication, and traces contain query text, context and evidence.
The server rejects non-loopback Host names and cross-origin run requests, validates
JSON bodies, and sends browser security headers. These controls are not a complete
security audit. See [security policy](SECURITY.md).

`.env`, model/index caches, generated evaluation reports, backup corpora and
submission archives are ignored by Git. Retain upstream dataset/model licenses
when redistributing assets; the raw corpus remains 122,049 passages and the
pre-existing cleaned index contains 101,763 rows.

## AI-use disclosure

Developed with AI coding assistance under human direction. The human set
requirements, approved implementation and verification, and owns final technical
decisions. AI assistance contributed code, diagnostics, documentation and executed
verification. Provider requests are confined to explicit runtime/evaluation paths.

## Repository layout

- `src/prism_live_rag/`: pipeline, providers, telemetry and dashboard assets.
- `tests/`: deterministic unit and regression checks.
- `scripts/`: preparation, audits, packaging and launch helpers.
- `docs/`: design references, dated research notes and versioned evaluation reports.
- `data/`: tracked query/stream fixtures and local ignored corpus/evaluation files.
