#!/usr/bin/env python3
"""
Evaluate retrieval quality and calculate metrics against the full corpus.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism_live_rag.config import load_settings
from prism_live_rag.data import load_query_tasks
from prism_live_rag.embeddings import build_encoder
from prism_live_rag.retrieval import HybridRetriever

DATA_DIR = Path("data")


def main():
    domain = "cloud"
    max_tasks = 50
    top_k = 5

    print(f"Loading {domain} tasks...")
    tasks = load_query_tasks(DATA_DIR, domain=domain)
    if max_tasks:
        tasks = tasks[:max_tasks]
    print(f"Loaded {len(tasks)} tasks")

    print("Initializing retriever...")
    settings = load_settings()
    encoder = build_encoder(
        backend=settings.embedding_backend,
        model=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        fixed_length=settings.embedding_fixed_length,
    )
    retriever = HybridRetriever(
        settings.data_dir,
        settings.lancedb_dir,
        settings.table_name,
        encoder,
        settings.rrf_k,
        settings.sparse_weight,
        use_dense=True,
        use_reranker=True,
    )
    print(f"  Encoder: {encoder.name}")
    print(f"  Device: {encoder.device}")
    print(f"  Dimensions: {encoder.dim}")

    print(f"\nEvaluating retrieval on {len(tasks)} queries...")
    recall_at_k = 0
    success_at_k = 0
    total = 0

    results = []

    for i, task in enumerate(tasks, 1):
        query = task.query
        qrels = task.qrel_passage_ids
        if not query or not qrels:
            continue

        # Retrieve
        retrieved = retriever.search(query, domain=domain, limit=top_k, candidate_limit=top_k * 6)
        retrieved_ids = [hit.passage.id for hit in retrieved]

        # Calculate metrics
        qrels_set = set(qrels)
        retrieved_set = set(retrieved_ids)
        overlap = len(qrels_set & retrieved_set)

        # Recall: proportion of relevant docs retrieved
        recall = overlap / len(qrels_set)
        recall_at_k += recall

        # Success: at least one relevant doc retrieved
        if overlap > 0:
            success_at_k += 1

        total += 1

        results.append({
            "task_id": task.task_id,
            "query": query,
            "qrels": qrels,
            "retrieved": retrieved_ids,
            "recall": recall,
            "success": overlap > 0,
        })

        if i % 10 == 0:
            print(f"  Processed {i}/{len(tasks)} queries...")

    # Calculate averages
    avg_recall = (recall_at_k / total * 100) if total > 0 else 0
    success_rate = (success_at_k / total * 100) if total > 0 else 0

    # Print results
    print("\n" + "=" * 80)
    print("RETRIEVAL METRICS")
    print("=" * 80)
    print("Configuration:")
    print(f"  Encoder: {encoder.name}")
    print(f"  Device: {encoder.device}")
    print(f"  Top-k: {top_k}")
    print()
    print("Results:")
    print(f"  Total queries: {total}")
    print(f"  Recall@{top_k}: {avg_recall:.2f}%")
    print(f"  Success@{top_k}: {success_rate:.2f}%")
    print("=" * 80)

    # Save results
    output_file = Path("data/retrieval_eval_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "encoder": encoder.name,
                "device": encoder.device,
                "top_k": top_k,
                "reranker": retriever.reranker_model,
            },
            "metrics": {
                "total_queries": total,
                "recall_at_k": avg_recall,
                "success_at_k": success_rate,
            },
            "results": results,
        }, f, indent=2)

    print(f"\n✓ Detailed results saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())