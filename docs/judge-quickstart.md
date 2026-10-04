# Judge quickstart

## Clone, add a Cerebras key, run

```bash
git clone https://github.com/Aakarsh-Kumar/prism-live-rag.git
cd prism-live-rag
cp .env.example .env
# Edit .env: set CEREBRAS_API_KEY to your own Cerebras key.
docker compose up --build
```

On Windows, copy `.env.example` to `.env` using your editor or file manager.
No host Python, GPU, corpus download, model installation, archive extraction or
manual index build is required.

Open **http://localhost:8080** once the logs print the dashboard URL. Leave the
Compose terminal running; Ctrl+C stops the app. Add `-d` for background operation.

The first build needs internet for pinned dependencies. First startup automatically
downloads the approximately 610 MiB release asset pinned in `release-assets.json`,
verifies its size and SHA-256, validates/extracts its file manifest, and caches the
corpus, existing index and CPU models in a Docker volume. Logs show preparation and
download progress. Startup never rebuilds the index or shrinks the corpus.

With a configured key, the dashboard and API default to **Automatic / Cerebras
GPT-OSS-120B**, paced at five requests/minute. Routing still skips the LLM when it
is unnecessary. No separate provider toggle is required. An invalid/expired key
or provider quota limit is a runtime error, not proof that a query is unanswerable.
Without a key, the dashboard explicitly falls back to provider-free extraction.

## Resources and troubleshooting

Allow at least 6 GB RAM for Docker (16 GB system RAM recommended) and roughly
8 GB free disk for the image, download/extraction cache and build overhead.
The service targets Linux AMD64; Docker Desktop on ARM hosts may use emulation,
which was not tested and may be slower.

First startup can take several minutes, depending on network speed and CPU.
Subsequent starts verify cached files without downloading them. If a transfer is
interrupted, restart; partial downloads are resumed where the server supports it.
The cache fails closed on corrupt files, invalid checksums or unsafe archive entries.

If port 8080 is occupied, set `PRISM_PORT=8081` in `.env` before launch.
Watch progress with `docker compose logs -f app`. Health is available at
`/api/health` after asset preparation and corpus initialization.

The dashboard remains a local demonstration. Compose publishes on loopback only;
do not use public tunnels. See [security policy](../SECURITY.md).
Avoid `docker compose down -v`: it deletes cached assets and persisted telemetry.
Changing the key requires `docker compose up -d --force-recreate`.

## Dashboard

All 200 accepted test queries are selectable and searchable. Transcript chunks
arrive at their scheduled intervals; revised/final hypotheses supersede earlier
partials. Inspect controller reasons, actual search dispatch, returned passages,
rank signals, citations and provider-call events. Completed answers support
presentation-only requests and targeted session refinements. Answers arrive whole,
not token-streamed. Citation overlap diagnostics are not semantic verdicts.

## Maintainer publication prerequisite

The pinned [asset release](https://github.com/Aakarsh-Kumar/prism-live-rag/releases/tag/PRISM_GENAI_HACKATHON_Y2026)
is published and publicly readable. Its URL, size, archive SHA-256 and manifest
SHA-256 are pinned in `release-assets.json`. A missing/private release is reported clearly; the app does
not silently run with missing corpus/models or ask judges for GitHub credentials.

On 30 September 2026, anonymous download returned HTTP 200 with the pinned size.
A separate empty Docker asset volume downloaded the public archive without a
local mount or GitHub credentials, verified archive/file checksums, and completed
`prism-rag validate-data`. The local dashboard loaded 101,763 cleaned passages and
completed two UI replays with citations and valid traces. These replays took about
177 seconds cold and 85 seconds warm on the Windows owner's CPU setup; latency
on this machine remains a limitation.

## Evidence and quality scope

Bootstrap/security/provider/packaging tests passed 59 focused checks on the host
and 59 inside the corrected Docker image. The new image
built without local asset COPY instructions or a corpus/model build context.
A fresh asset volume installed the complete pinned archive and loaded 101,763
cleaned passages with CPU hybrid retrieval; two replay queries returned citations
and valid exact-ID traces. The raw corpus still contains 122,049 passages.

Historical CPU coverage remains 200/200 traces and 118/130 early searches (90.77%).
Offline extraction was uncertain or uncited on 59/100 answerable cases; the
bootstrap fix does not erase that quality gap or establish a new semantic pass.
See [benchmark report](benchmark-evaluation-report.md).
