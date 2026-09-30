#!/usr/bin/env python3
"""Matched retrieval ablations on one qrel set and one existing LanceDB index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_live_rag.config import load_settings
from prism_live_rag.data import load_query_tasks
from prism_live_rag.embeddings import build_encoder
from prism_live_rag.retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("cloud", "govt"), default="cloud")
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    encoder = build_encoder(
        "fastembed", model=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=True, device=args.device,
        batch_size=settings.embedding_batch_size,
        fixed_length=settings.embedding_fixed_length,
    )
    retriever = HybridRetriever(
        settings.data_dir, settings.lancedb_dir, settings.table_name,
        encoder, settings.rrf_k, settings.sparse_weight,
        use_dense=True, use_reranker=True,
        bm25_k1=settings.bm25_k1, bm25_b=settings.bm25_b,
    )
    if not retriever.dense.exists():
        raise SystemExit(f"Missing index {retriever.dense.table_name}; no index will be built by this benchmark")
    tasks = [t for t in load_query_tasks(settings.data_dir, args.domain) if t.qrel_passage_ids][:args.max_tasks]
    strategies = (
        ("dense_only_reranked", "dense", True),
        ("hybrid_unreranked", "hybrid", False),
        ("hybrid_reranked", "hybrid", True),
    )
    observations: dict[str, list[dict]] = {name: [] for name, _, _ in strategies}
    for index, task in enumerate(tasks, start=1):
        gold = set(task.qrel_passage_ids)
        for name, leg, rerank in strategies:
            retriever.retrieval_leg = leg
            retriever.use_reranker = rerank
            hits = retriever.search(task.query, task.domain, limit=5, candidate_limit=30)
            hit_ids = [hit.passage.id for hit in hits]
            matched = gold.intersection(hit_ids)
            first_rank = next((i for i, pid in enumerate(hit_ids, start=1) if pid in gold), None)
            observations[name].append({
                "task_id": task.task_id, "query": task.query,
                "gold_ids": sorted(gold), "retrieved_ids": hit_ids,
                "matched_ids": sorted(matched), "hit_rank": first_rank,
                "recall_at_5": len(matched) / len(gold),
            })
        if index % 10 == 0:
            print(f"processed {index}/{len(tasks)} tasks", flush=True)

    metrics = {}
    for name, rows in observations.items():
        count = len(rows)
        metrics[name] = {
            "tasks": count,
            "success_at_5": round(sum(r["hit_rank"] is not None for r in rows) / count, 4),
            "recall_at_5": round(sum(r["recall_at_5"] for r in rows) / count, 4),
            "mrr_at_5": round(sum(1 / r["hit_rank"] for r in rows if r["hit_rank"]) / count, 4),
        }
    report = {
        "protocol": "same first N qrel-bearing tasks, same cleaned full corpus, same query expansion, top 5; dense-only + rerank vs hybrid + rerank, and hybrid without vs with rerank",
        "domain": args.domain,
        "encoder": encoder.name,
        "device": encoder.device,
        "index_table": retriever.dense.table_name,
        "sparse_passages": len(retriever.sparse.passages),
        "max_tasks": args.max_tasks,
        "metrics": metrics,
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
