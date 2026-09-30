# Judge quickstart

The submission targets Linux AMD64 containers (Docker Desktop is suitable on an
AMD64 Windows/macOS host). GPU drivers and API keys are not required. ARM64 is not
yet verified. Model files, the supplied corpus and the existing index are bundled;
startup does not rebuild the index or download models.

Allow at least 6 GB RAM for Docker (16 GB system RAM recommended), and sufficient
disk for the roughly 4.1 GB uncompressed image plus the supplied archive. GPU
hardware is not required; this portable image uses CPU inference. The full raw
122,049-passage corpus is included; the existing cleaned index has 101,763 rows.

## Prebuilt image

Place the supplied image archive in the repository root and launch with one command:

```bash
bash scripts/run_judge.sh
```

For an unattended provider-free 200-query replay with persisted checkpoints and
exact trace-ID coverage validation, use `bash scripts/run_judge.sh --verify`.
This is deliberately longer than a quick interactive demonstration. Results are
stored in the telemetry volume as `judge-replay.jsonl` and
`judge-replay-report.json`; no API requests are needed.

Or load the archive and launch manually:

```bash
docker load -i prism-live-rag-judge.tar.gz
docker compose up --no-build -d
```

Open http://localhost:8080. If that port is occupied, use
`PRISM_PORT=8081 docker compose up --no-build -d` and open port 8081.
Watch readiness with `docker compose logs -f app`; health is available at
`/api/health` after corpus initialization. The initial CPU image reached readiness
in 26.7 seconds on the development PC without a GPU or host asset mounts. This
excludes loading the image archive and is not a guarantee for an unknown machine.

## Source build

Extract the supplied asset bundle into `submission-assets/` at repository root,
then run `docker compose up --build -d`. Builders need internet for pinned Python
dependencies; the prebuilt-image path avoids this step.

Maintainers can stage existing artifacts with:

```bash
.venv/bin/python scripts/prepare_judge_assets.py
docker compose build app
docker save -o prism-live-rag-judge.tar prism-live-rag:judge
```

The staging command never rebuilds or changes source corpus/index files and writes
SHA-256 checksums into `submission-assets/manifest.json`. Do not include `.env` in
either archive. Redistributing corpus/model assets remains subject to their
original licenses; retain upstream provenance with the submission.

## Dashboard

All 200 evaluated scenarios remain selectable. Transcript chunks arrive at their
own scheduled intervals; final ASR hypotheses supersede partials. Inspect the
timeline for controller decisions, retrieval-before-final, decomposed queries,
retrieved passages and observed provider calls. The answer arrives as a complete
response, not token-streamed output.

Default no-key operation is deterministic extraction. Optional `.env` values
`CEREBRAS_API_KEY` and `CEREBRAS_MODEL=gpt-oss-120b` enable query-selective provider
mode at five requests/minute. Provider mode is slower and quota-dependent; no key
is sent to the browser. Benchmark proxies are not semantic correctness scores.

The telemetry volume contains execution traces. Avoid `docker compose down -v`
unless you intentionally want to delete that volume.

## Verification status

Clean-directory Compose launch without `.env` or host data/cache mounts was
exercised. The release reached readiness in 24.1 seconds on the development PC;
the final image also completed two cited queries with Docker networking disabled,
no mounts, no API keys and no GPU, with both traces validated by execution ID.
first measured browser runs ranged roughly 6–21 seconds per pipeline execution
at accelerated transcript pacing, with additional cold-reranker overhead.
The CPU audit completed 200 unique runs with 200 valid traces and no invalid
citation IDs. Actual early search was 118/130 eligible cases (90.77%). However,
59/100 answerable cases were uncertain or uncited in default offline mode.
Launch and trace coverage do not prove answer-quality completion. See the
benchmark report for version-specific results and limitations.
