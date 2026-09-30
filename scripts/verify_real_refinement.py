#!/usr/bin/env python3
"""One real full-corpus, two-turn G5 replay using the existing GPU index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_live_rag.config import load_settings
from prism_live_rag.controller import TranscriptChunk
from prism_live_rag.embeddings import build_encoder
from prism_live_rag.models import ConversationSession
from prism_live_rag.pipeline import StreamingRagPipeline
from prism_live_rag.retrieval import HybridRetriever


class RecordingRetriever:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever
        self.queries: list[str] = []
        self.use_reranker = retriever.use_reranker

    def search(self, query: str, domain: str):
        self.queries.append(query)
        return self.retriever.search(query, domain)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    settings = load_settings()
    encoder = build_encoder(
        "fastembed", model=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir, local_files_only=True,
        device="cuda", batch_size=settings.embedding_batch_size,
        fixed_length=settings.embedding_fixed_length,
    )
    underlying = HybridRetriever(
        settings.data_dir, settings.lancedb_dir, settings.table_name,
        encoder, settings.rrf_k, settings.sparse_weight,
        use_dense=True, use_reranker=True,
        bm25_k1=settings.bm25_k1, bm25_b=settings.bm25_b,
    )
    if not underlying.dense.exists():
        raise SystemExit("Missing BGE-small LanceDB index")
    retriever = RecordingRetriever(underlying)
    pipeline = StreamingRagPipeline(retriever)
    session = ConversationSession()
    first = pipeline.run([
        TranscriptChunk(0.5, "Does IBM Cloud Transit Gateway", False, 0.83),
        TranscriptChunk(1.1, "Does IBM Cloud Transit Gateway perform encryption between VPCs?", True, 0.99),
    ], session=session)
    initial_queries = list(retriever.queries)
    retriever.queries.clear()
    second = pipeline.run([
        TranscriptChunk(0.5, "for transit gateway DDoS", False, 0.83),
        TranscriptChunk(0.9, "for transit gateway DDoS", False, 0.86),
        TranscriptChunk(1.4, "for transit gateway DDoS protection only", True, 0.99),
    ], session=session, refinement=True)
    assert first.version == 1 and first.citations
    assert second.previous_version == 1 and second.version == session.version == 2
    assert second.applied_delta == "for transit gateway DDoS protection only"
    assert retriever.queries and retriever.queries[-1] == second.applied_delta
    assert all("encryption between VPCs" not in query for query in retriever.queries)
    assert "DDoS" in second.answer
    assert second.answer.count("does not perform encryption") <= 1
    assert "How do you prevent" not in second.answer
    result = {
        "protocol": "full cleaned corpus, BGE-small CUDA, hybrid plus cross-encoder, deterministic synthesis; one two-turn smoke, not a G5 accuracy benchmark",
        "corpus_passages": len(underlying.sparse.passages),
        "first": {"queries": initial_queries, "answer": first.answer, "citations": first.citations, "version": first.version},
        "refinement": {"queries": retriever.queries, "answer": second.answer, "citations": second.citations, "previous_version": second.previous_version, "version": second.version, "applied_delta": second.applied_delta, "uncertainty": second.uncertainty},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "first_citations": first.citations, "refined_citations": second.citations, "version": second.version, "uncertainty": second.uncertainty}, indent=2))


if __name__ == "__main__":
    main()
