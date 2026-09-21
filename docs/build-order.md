# Build Order

## Current progress

- Runtime selected: **Python**.
- Vector database selected: **LanceDB**.
- Python package, CLI, Dockerfile, `docker-compose`, and baseline tests are
  scaffolded.
- Repo hygiene is configured for sequential commits: local secrets, venvs, caches,
  generated LanceDB indexes, nested dataset Git history, and large corpus JSONL files
  are ignored.
- Working corpus prepared under `data/`: Cloud passage corpus and qrels as the
  primary domain; Govt passage corpus and qrels as optional secondary domain.
- MTRAG-UN query text is sourced from
  `data/mtragun-human/generation_tasks/reference.jsonl` by joining
  `task_id == qrels.query-id`; Cloud and Govt coverage was verified at 100%.

## Next build sequence

1. Python repo scaffolding, `docker-compose`, one-command run script — complete for
   the deterministic baseline; keep hardening for [Gate G1](evaluation-gates.md).
2. Transcript chunking simulator (own stub data; swap in real transcripts later) —
   base format on [dataset-and-augmentation.md](dataset-and-augmentation.md).
3. Data loader for the cleaned MTRAG-UN working set: Cloud first, Govt optional.
4. LanceDB ingestion for passage-level Cloud/Govt corpus files — initial CLI exists;
   run it under Python 3.11-3.13 or Docker, then pass `--use-dense` in demos.
5. Hybrid retrieval + weighted RRF, fixed-window chunking — initial deterministic
   dense+sparse baseline exists; see
   [retrieval.md](retrieval.md) for exact config to start from.
6. Rule-based Retrieval Controller (heuristic Wait/Retrieve/Suppress). **Provider
   choice follows the latency-first live-path routing rule in
   [llm-providers.md](llm-providers.md).**
7. Multi-Intent Decomposer (LLM-prompted, capped at 2–4 sub-queries). Use the same
   latency-first live-path routing rule.
8. Evidence span extraction + grounded generation + deterministic fallback — see
   [generation-grounding.md](generation-grounding.md) and [prompts.md](prompts.md).
   **Use stronger/slower models for evidence extraction when needed, and fast
   streaming models for final answer delivery.**
9. Model-based Retrieval Controller (classifier) — for the required rule-based vs.
   model-based ablation.
10. Late-refinement patch logic (delta update, not restart).
11. Evaluation harness — wire up against [evaluation-gates.md](evaluation-gates.md).
12. Telemetry/observability — should already be logging from step 1 onward, not
    bolted on at the end.
