# Prism Live RAG

Event-driven streaming retrieval-augmented generation for Samsung Theme 04.
Timestamped text chunks simulate evolving ASR hypotheses: the controller can
retrieve before the user finishes speaking, while the final answer is grounded
in retrieved Cloud/Govt passages.

## Samsung hackathon submission

Theme 04 · Binary Bits · SRM Institute of Science and Technology.
The final submission tag is `PRISM_GENAI_HACKATHON_Y2026`.

- [Presentation PDF](SRMIST_BinaryBits_Submission.pdf)
- [Demo video on Google Drive](https://drive.google.com/drive/folders/1kKADF41rKLjVKMlLcp7u8tu87hfRkfLn?usp=sharing) — 4 minutes 52 seconds; publicly readable.
- [AI-use disclosure PDF](LangAI3.0_AI_Disclosure.pdf)

The presentation and disclosure are included in the tagged commit. The demo video
is hosted on Drive and referenced here at the owner's request.

## Run the judge dashboard

Install/start Docker, clone this repository, and provide your own Cerebras key:

```bash
git clone https://github.com/Aakarsh-Kumar/prism-live-rag.git
cd prism-live-rag
cp .env.example .env
# Edit .env and set CEREBRAS_API_KEY.
docker compose up
```

On Windows, copy `.env.example` using your editor/file manager. Open
**http://localhost:8080** when startup logs show the dashboard URL.
Docker automatically pulls the prebuilt image `aakarshkumar25/prism-live-rag:judge`
from Docker Hub, downloads the pinned corpus/index/model release asset, verifies
archive and file checksums, and caches it in a persistent volume.
**No manual asset download, extraction, GPU, host Python, or local build is required.**
First run pulls the prebuilt image and downloads assets in ~2-4 minutes; subsequent starts
are instantaneous. (To build from source instead, append `--build`).
Allocate at least 6 GB RAM to Docker. See [judge quickstart](docs/judge-quickstart.md).

The pinned [judge asset release](https://github.com/Aakarsh-Kumar/prism-live-rag/releases/tag/PRISM_GENAI_HACKATHON_Y2026)
is public. On 30 September 2026, anonymous download and a fresh-volume Docker
bootstrap passed archive/manifest checks and corpus/query validation. Judges do
not need GitHub authentication or a local asset archive.

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
[pipeline architecture](docs/architecture.md) and [telemetry schema](docs/telemetry-schema.md).

## Verification and honest limits

The release verification recorded **275 tests passed, two skipped**. Subsequent
bootstrap, security, provider and packaging checks passed **59 focused tests on
the host and 59 inside Docker**;
the full suite was not rerun after these changes. The new image built without a
local corpus/model build context, installed the pinned archive into a fresh asset
volume and completed two cited queries with valid traces. Cached restart succeeded
without networking or an archive mount; configured credentials selected provider
mode automatically without making an inference request during that startup check.

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
in submission bundles rather than Git history.

## Cerebras routing

Copy `.env.example` to `.env` and set `CEREBRAS_API_KEY` locally.
The dashboard uses `CEREBRAS_MODEL=gpt-oss-120b`, paced at five requests/minute.
Routing determines whether a query needs a provider call; not every query does.
Keys stay server-side. With a key, the dashboard/API default to automatic provider
routing. No-key operation is explicitly provider-free and not the recommended
quality demonstration. Provider mode is quota-dependent and slower. Groq/DeepSeek
remain available for legacy CLI paths.

## Local development

Use Python **3.11–3.13**, not host Python 3.14.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/prism-rag validate-data
.venv/bin/prism-rag serve --port 8080
```

Local execution requires the corpus described in
[corpus README](data/corpora/README.md). Neural retrieval additionally requires
model files and a matching index. `bash scripts/setup_embeddings.sh` selects ONNX
Runtime for the available GPU/CPU. `prism-rag index` builds an index; it is not a
startup step for the Docker judge path. Do not replace the full submission index with
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
