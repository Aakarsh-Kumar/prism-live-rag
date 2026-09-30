#!/usr/bin/env python3
"""Build evaluation dataset from MTRAG queries for Samsung G4 compliance testing.

Extracts queries with ground truth answers from MTRAG reference data,
runs the RAG pipeline, and saves results for batch RAGChecker evaluation.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism_live_rag.retrieval import HybridRetriever
from prism_live_rag.config import load_settings
from prism_live_rag.embeddings import build_encoder
from prism_live_rag.data import iter_passages, load_qrels
from prism_live_rag.controller import simulate_chunks
from prism_live_rag.pipeline import StreamingRagPipeline

# Only collections present in the local corpus can be evaluated. The MT-RAG 2.0
# reference contains clapnq/fiqa too, but those passages are not in the corpus.
COLLECTION_TO_DOMAIN = {"ibmcloud": "cloud", "govt": "govt"}


def extract_evaluation_queries(
    reference_path: Path,
    qrels_by_domain: dict[str, dict[str, list[str]]],
    passage_ids_by_domain: dict[str, set[str]],
    max_queries: int = 50,
    *,
    seed: int = 20260928,
):
    """Select a deterministic, domain-balanced, qrel-joined MTRAG sample."""
    pools: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for line_num, line in enumerate(reference_path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{reference_path}:{line_num}: invalid JSON: {exc}") from exc
        collection = data.get("Collection", "unknown")
        domain = COLLECTION_TO_DOMAIN.get(collection)
        task_id = str(data.get("task_id", ""))
        if domain is None:
            continue
        if task_id in seen:
            raise ValueError(f"Duplicate MTRAG task_id: {task_id}")
        seen.add(task_id)
        inputs = data.get("input") or []
        last_user_index = max(
            (i for i, turn in enumerate(inputs) if turn.get("speaker") == "user"),
            default=-1,
        )
        targets = data.get("targets") or []
        answerability = str((data.get("answerability") or ["UNKNOWN"])[0]).upper()
        query = str(inputs[last_user_index].get("text", "")).strip() if last_user_index >= 0 else ""
        answer = str(targets[0].get("text", "")).strip() if targets else ""
        qrel_ids = qrels_by_domain[domain].get(task_id, [])
        if answerability in {"ANSWERABLE", "PARTIAL"} and not qrel_ids:
            continue
        if not query or not answer or answerability not in {
            "ANSWERABLE", "PARTIAL", "UNANSWERABLE", "UNDERSPECIFIED"
        }:
            continue
        missing_qrels = set(qrel_ids) - passage_ids_by_domain[domain]
        if missing_qrels:
            raise ValueError(
                f"{task_id}: qrels refer to passage IDs absent from {domain} corpus: "
                f"{sorted(missing_qrels)[:3]}"
            )
        context_ids = {
            str(context.get("document_id", ""))
            for context in data.get("contexts") or []
            if context.get("document_id")
        }
        if context_ids - passage_ids_by_domain[domain]:
            raise ValueError(f"{task_id}: benchmark context IDs are absent from {domain} corpus")
        context_turns = [
            {"speaker": str(turn.get("speaker", "")), "text": str(turn.get("text", ""))}
            for turn in inputs[:last_user_index]
            if turn.get("speaker") in {"user", "agent"} and turn.get("text", "").strip()
        ][-4:]
        pools[domain].append(
            {
                "query_id": task_id,
                "conversation_id": str(data.get("conversation_id", "")),
                "query": query,
                "domain": domain,
                "context_turns": context_turns,
                "gt_answer": answer,
                "gold_passage_ids": sorted(set(qrel_ids)),
                "has_qrels": bool(qrel_ids),
                "answerability": answerability,
                "collection": collection,
            }
        )

    # 25 examples per supported collection, covering all four answerability labels.
    per_domain = max_queries // len(COLLECTION_TO_DOMAIN)
    base_quotas = {
        "ANSWERABLE": 13,
        "PARTIAL": 4,
        "UNANSWERABLE": 4,
        "UNDERSPECIFIED": 4,
    }
    if per_domain != sum(base_quotas.values()):
        base_quotas = None
    rng = random.Random(seed)
    selected = []
    for domain in sorted(qrels_by_domain):
        if domain not in pools:
            continue
        pool = pools[domain]
        rng.shuffle(pool)
        if base_quotas:
            domain_rows = []
            for label, quota in base_quotas.items():
                candidates = [row for row in pool if row["answerability"] == label]
                if len(candidates) < quota:
                    raise ValueError(f"{domain}: need {quota} {label} queries, found {len(candidates)}")
                domain_rows.extend(candidates[:quota])
            selected.extend(domain_rows)
        else:
            selected.extend(pool[:per_domain])
    if len(selected) < max_queries:
        raise ValueError(f"Requested {max_queries} G4 queries, selected only {len(selected)}")
    return selected[:max_queries]


def run_rag_pipeline(queries, retriever, domain: str, limit: int = 10):
    """Run G4 cases through the streaming application pipeline used by the UI."""
    results = []
    pipeline = StreamingRagPipeline(
        retriever,
        synthesis_mode="deterministic",
        refine_on_final=True,
        enable_multi_intent=True,
    )

    for i, q in enumerate(queries, 1):
        print(f"Processing {i}/{len(queries)}: {q['query'][:60]}...")

        try:
            context_turns = tuple(
                (turn["speaker"], turn["text"])
                for turn in q.get("context_turns", [])
            )
            retrieval_events = []
            response = pipeline.run(
                simulate_chunks(q["query"]),
                q.get("domain", domain),
                context_turns=context_turns,
                trace=True,
                on_event=retrieval_events.append,
            )
            final_retrieval = next(
                (
                    event for event in reversed(retrieval_events)
                    if event.get("type") == "retrieval_done"
                ),
                {},
            )
            passages_by_id = {
                passage["id"]: passage
                for event in retrieval_events
                if event.get("type") == "retrieval_done"
                for passage in event.get("passages", [])
            }
            citation_ids = response.citations
            answer = response.answer
            uncertainty = response.uncertainty
            citations = set(citation_ids)
            retrieved_ids = set(passages_by_id)
            if not citations <= retrieved_ids:
                raise ValueError(f"{q['query_id']}: synthesis cited a passage not retrieved")
            print(f"  Extractive: {answer[:80]}...")

            results.append({
                "query_id": q["query_id"],
                "query": q["query"],
                "domain": q["domain"],
                "collection": q["collection"],
                "answerability": q["answerability"],
                "gold_passage_ids": q["gold_passage_ids"],
                "has_qrels": q["has_qrels"],
                "qrel_retrieval_hit": bool(
                    set(q["gold_passage_ids"]) & retrieved_ids
                ),
                "retrieval_query": final_retrieval.get("query", q["query"]),
                "sub_queries": response.sub_queries,
                "response": answer,
                "citations": sorted(citations),
                "uncertainty": uncertainty,
                "abstained": not citation_ids and uncertainty is not None,
                "gt_answer": q["gt_answer"],
                "retrieved_context": [
                    {
                        "doc_id": passage_id,
                        "text": passage["text"],
                        "cited": passage_id in citations,
                        "score": passage.get("score"),
                    }
                    for passage_id, passage in passages_by_id.items()
                ],
                "context_turns": q.get("context_turns", []),
                "trace_id": response.trace_id,
                "decisions": response.decisions,
                "retrieval_events": [
                    event for event in response.to_dict().get("retrieval_events", [])
                ],
            })

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return results


def main():
    """Build evaluation dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/eval_dataset.jsonl"))
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument(
        "--input-queries",
        type=Path,
        help="Reuse an existing JSONL query set (query_id/query/domain/context_turns/golds).",
    )
    args = parser.parse_args()
    print("Building evaluation dataset from MTRAG queries...")

    # Paths
    reference_path = Path("data/mtragun-human/generation_tasks/reference.jsonl")
    output_path = args.output

    if args.input_queries is None and not reference_path.exists():
        print(f"Error: Reference file not found: {reference_path}")
        return 1

    # Extract queries
    settings = load_settings()
    if args.input_queries is not None:
        print(f"\n1. Loading the selected query set from {args.input_queries}...")
        queries = [
            json.loads(line)
            for line in args.input_queries.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:args.max_queries]
        required = {"query_id", "query", "domain", "collection", "answerability"}
        if any(not required <= row.keys() for row in queries):
            raise ValueError(f"Input query rows must contain {sorted(required)}")
        query_ids = [str(row["query_id"]) for row in queries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("Input query file contains duplicate query_id values")
    else:
        print(f"\n1. Extracting queries from {reference_path}...")
        print("   (clapnq/fiqa collections excluded - their passages are not in the corpus)")
        qrels = {domain: load_qrels(settings.data_dir, domain) for domain in COLLECTION_TO_DOMAIN.values()}
        passage_ids = {
            domain: {passage.id for passage in iter_passages(settings.data_dir, domain)}
            for domain in COLLECTION_TO_DOMAIN.values()
        }
        queries = extract_evaluation_queries(
            reference_path, qrels, passage_ids, max_queries=args.max_queries
        )
    print(f"   Extracted {len(queries)} queries")

    if not queries:
        print("Error: no corpus-matched queries found")
        return 1

    # Initialize retriever
    print("\n2. Initializing retriever with reranking...")
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
        reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        reranker_top_k=50,
    )

    # Run RAG pipeline on each query's own corpus domain
    print("\n3. Running retrieval with extractive grounded synthesis...")
    results = run_rag_pipeline(queries, retriever, domain="cloud", limit=10)
    print(f"   Processed {len(results)} queries successfully")
    if len(results) != len(queries):
        raise RuntimeError(f"Only {len(results)}/{len(queries)} G4 queries completed")

    # Save results
    print(f"\n4. Saving results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n✓ Evaluation dataset saved: {len(results)} queries")
    print(f"  Output: {output_path}")

    # Print stats
    answerability_counts = {}
    collection_counts = {}
    for r in results:
        answerability_counts[r["answerability"]] = answerability_counts.get(r["answerability"], 0) + 1
        collection_counts[r["collection"]] = collection_counts.get(r["collection"], 0) + 1

    print(f"\n  Answerability distribution:")
    for k, v in sorted(answerability_counts.items()):
        print(f"    {k}: {v}")

    print(f"\n  Collection distribution:")
    for k, v in sorted(collection_counts.items()):
        print(f"    {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
