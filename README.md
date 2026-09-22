# Prism Live RAG

Streaming Live RAG baseline for Samsung Theme 04. The project is built in Python
with LanceDB selected as the local vector database.

The live path is driven by a rigorous streaming simulator
(`src/prism_live_rag/stream.py`): irregular word delivery, mid-stream ASR revisions,
LocalAgreement-n intent stability, and per-stream settling time. The rule-based
controller fires only once the partial transcript is stable, and `eval --mode streaming`
measures early retrieval before the user finishes speaking.

## Project Status & Known Limitations

The streaming foundation (simulator, intent-stability controller, streaming eval, G1
reproducibility) is implemented and tested. The multi-intent decomposer, session
refinement, full telemetry, model-based controller, ablations, and evaluation report
are still in progress. Honest limitations a reader should assume:

- **Streaming input is simulated, not real ASR.** `stream.py` reproduces partials,
  revisions, and stability timing, but no audio/ASR adapter is wired in. Treat it as a
  rigorous simulator with a documented swap point.
- **Answer delivery is not token-streamed.** Retrieval fires early; the final answer is
  returned whole. Do not read the streaming story as token-level streaming.
- **The early-retrieval rate measures a provisional fire before the final chunk**, not
  a fire before the ground-truth `stability_chunk_index`. See
  `docs/approach-summary.md` for the exact definition.
- **Retrieval recall is the current quality ceiling** (~0.42 recall@k / ~0.60
  success@k on the local working set), which caps grounding.
- **Abstention on genuinely unanswerable input is unsolved** in the literature; see
  `docs/risks-and-open-problems.md`.

## AI-Use Disclosure

This project was developed with AI coding assistance under human direction: the human
set the architecture, reviewed and edited all code, ran every experiment, and owns the
final technical decisions. AI was used as an implementation and documentation aid, not
as an autonomous author. Provider-backed features (Groq/DeepSeek) are used only in
explicit demo/eval paths and never in tests or CI.

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

Run offline retrieval/controller evaluation:

```bash
prism-rag eval --domain cloud --mode retrieval --max-tasks 50
```

Run streaming evaluation (early-retrieval, false-trigger, settling time) against the
committed simulated streams:

```bash
prism-rag eval --domain cloud --mode streaming --max-tasks 200
```

Regenerate or validate the simulated streams (deterministic, no API calls):

```bash
prism-rag generate-streams --domain cloud
prism-rag validate-streams --domain cloud
```

Play back a single stream to see each partial, the controller decision and reason, the
mid-stream revision, and the final grounded answer:

```bash
prism-rag play-stream --domain cloud --stream-id cloud-0044
```

Run a bounded real-provider smoke evaluation:

```bash
prism-rag eval --domain cloud --mode provider --max-tasks 5 --rewrite-query
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
