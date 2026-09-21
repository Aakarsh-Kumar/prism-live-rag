# Approach Summary

## Problem Framing

We are building a Streaming Live RAG engine for Samsung Theme 04: a system that can
listen to a live, full-duplex conversation, decide when the user's intent is stable
enough to retrieve, and start grounding an answer before the user has fully finished
speaking. The core challenge is not just retrieval quality; it is balancing early
retrieval against the risk of acting on unstable partial transcripts.

## Technical Stack

- **Runtime:** Python 3.11-3.13, with Docker pinned to Python 3.11.
- **Vector database:** LanceDB for local persistent vector indexes.
- **Entrypoints:** `prism-rag validate-data`, `prism-rag index`, and
  `prism-rag run-demo --domain cloud`.
- **Container path:** `docker compose up --build` runs the demo service.
- **Primary corpus:** IBM MTRAG-UN Cloud domain, using passage-level technical
  documentation.
- **Secondary corpus:** MTRAG-UN Govt domain for optional distribution-shift testing.
- **Evaluation focus:** Samsung gates G1-G6, especially early retrieval, factual
  grounding, session refinement, and observability.

## Dataset Strategy

The local `data/` directory has been cleaned to the working set needed for the
pipeline. The large passage-level corpus files are kept locally but ignored by Git:

- `data/corpora/passage_level/cloud.jsonl`
- `data/corpora/passage_level/govt.jsonl`
- `data/mtragun-human/retrieval_tasks/qrels/cloud.tsv`
- `data/mtragun-human/retrieval_tasks/qrels/govt.tsv`
- `data/mtragun-human/generation_tasks/reference.jsonl`

MTRAG-UN does not provide a separate retrieval query file. We join qrels
`query-id` to `reference.jsonl.task_id` and use the final user turn as query text.
This join has been verified for Cloud and Govt, including passage-ID coverage against
the extracted passage corpora.

## Pipeline Approach

The pipeline has four main stages:

1. **Retrieval Controller:** watches incremental transcript chunks and decides
   `Wait`, `Retrieve`, or `No-Retrieval` based on intent stability.
2. **Multi-Intent Decomposer:** splits compound utterances into 2-4 independent
   search queries when needed.
3. **Corpus Retrieval & Fusion:** uses hybrid dense+sparse retrieval, LanceDB-backed
   dense search, weighted/nested RRF fusion, and optional cross-encoder reranking
   when latency allows.
4. **Session-Aware Synthesis:** extracts evidence spans before generation, enforces
   citation grounding, applies deterministic insufficient-evidence fallback, and
   patches late constraints instead of restarting the session.

The current implementation is a deterministic baseline: hash embeddings for initial
local dense indexing, in-memory sparse scoring, weighted RRF fusion, a rule-based
streaming controller, and extraction-based synthesis. By default the demo uses the
fast sparse/RRF path; after running `prism-rag index`, `--use-dense` includes the
LanceDB dense index. Groq and DeepSeek keys are loaded from `.env` for the next
provider-backed iteration, but the baseline can run without spending API calls.

Local note: LanceDB indexing should be run under the supported Python range or via
Docker. Host Python 3.14 is currently not treated as a supported LanceDB runtime.

## Key Design Choices

We allow cross-encoder reranking, but ban generative LLM-as-judge reranking because
the reviewed SemEval systems found it slower, costlier, and lower quality than RRF
on the same candidate pool.

The live path is latency-first. The controller, decomposer, and final streamed answer
should use the fastest acceptable provider to minimize time-to-first-token. Slower,
stronger models can be reserved for non-streaming quality work such as query
rewriting and evidence span extraction.

Grounding is treated as a structural constraint, not a prompt preference. The system
must only cite retrieved chunk IDs, avoid parametric factual claims, and return a
deterministic insufficient-evidence response when supporting passages are missing.
