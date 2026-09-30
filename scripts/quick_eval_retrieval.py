#!/usr/bin/env python3
"""Quick retrieval quality evaluation using existing infrastructure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism_live_rag.config import load_settings
from prism_live_rag.data import load_query_tasks
from prism_live_rag.embeddings import build_encoder
from prism_live_rag.retrieval import LanceIndex


def main():
    domain = "cloud"
    max_tasks = 50
    top_k = 5

    print("=" * 80)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 80)

    # Load tasks
    print(f"\nLoading {domain} tasks...")
    tasks = load_query_tasks(Path("data"), domain=domain)[:max_tasks]
    print(f"✓ Loaded {len(tasks)} tasks")

    # Build encoder
    print("\nInitializing GPU-accelerated encoder...")
    settings = load_settings()
    encoder = build_encoder(
        backend=settings.embedding_backend,
        model=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        fixed_length=settings.embedding_fixed_length,
    )
    print(f"✓ Encoder: {encoder.name}")
    print(f"  Device: {encoder.device}")
    print(f"  Dimensions: {encoder.dim}")

    # Load index
    print("\nLoading LanceDB index...")
    index = LanceIndex(settings.lancedb_dir, settings.table_name, encoder)
    print(f"✓ Index loaded: {index.table_name}")

    # Evaluate
    print(f"\nEvaluating retrieval on {len(tasks)} queries...")
    recall_sum = 0
    success_count = 0
    total = 0

    for i, task in enumerate(tasks, 1):
        query = task.query
        qrels = task.qrel_passage_ids

        if not query or not qrels:
            continue

        # Retrieve using dense (neural) search
        results = index.search(query, domain=domain, limit=top_k)
        retrieved_ids = [passage.id for passage in results]

        # Calculate metrics
        qrels_set = set(qrels)
        retrieved_set = set(retrieved_ids)
        overlap = len(qrels_set & retrieved_set)

        # Recall: proportion of relevant docs retrieved
        recall = overlap / len(qrels_set)
        recall_sum += recall

        # Success: at least one relevant doc retrieved
        if overlap > 0:
            success_count += 1

        total += 1

        if i % 10 == 0:
            print(f"  Processed {i}/{len(tasks)}...")

    # Calculate metrics
    avg_recall = (recall_sum / total * 100) if total > 0 else 0
    success_rate = (success_count / total * 100) if total > 0 else 0

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS - GPU-Accelerated Neural Retrieval")
    print("=" * 80)
    print("Configuration:")
    print(f"  Model: {encoder.name}")
    print(f"  Device: {encoder.device}")
    print(f"  Top-k: {top_k}")
    print()
    print("Metrics:")
    print(f"  Total queries: {total}")
    print(f"  Recall@{top_k}: {avg_recall:.2f}%")
    print(f"  Success@{top_k}: {success_rate:.2f}%")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)