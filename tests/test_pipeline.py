from pathlib import Path

import pytest

from prism_live_rag.config import Settings
from prism_live_rag.controller import TranscriptChunk, simulate_chunks
from prism_live_rag.data import corpus_path
from prism_live_rag.models import Passage, RetrievedPassage
from prism_live_rag.pipeline import StreamingRagPipeline
from prism_live_rag.retrieval import HybridRetriever


def corpus_available(domain: str = "cloud") -> bool:
    return corpus_path(Path("data"), domain).exists()


@pytest.mark.skipif(not corpus_available(), reason="cloud corpus is gitignored and absent")
def test_pipeline_returns_required_shape() -> None:
    settings = Settings(data_dir=Path("data"), lancedb_dir=Path(".cache/test-lancedb"))
    retriever = HybridRetriever(
        settings.data_dir,
        settings.lancedb_dir,
        settings.table_name,
        settings.embedding_dim,
        settings.rrf_k,
        settings.sparse_weight,
        corpus_limit=300,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
    )
    pipeline = StreamingRagPipeline(retriever)
    response = pipeline.run(simulate_chunks("I heard the toolchain is not available in South America."), "cloud")
    payload = response.to_dict()
    assert set(payload) == {"retrieval_events", "sub_queries", "answer", "citations", "uncertainty"}
    assert payload["sub_queries"]
    assert payload["answer"]


class FakeRetriever:
    def __init__(self, final_has_passages: bool = True) -> None:
        self.queries = []
        self.final_has_passages = final_has_passages

    def search(self, query: str, domain: str):
        self.queries.append(query)
        if "South America" in query and self.final_has_passages:
            return [
                RetrievedPassage(
                    Passage(
                        id="cloud-final",
                        domain=domain,
                        text="The toolchain service is available in South America.",
                    ),
                    score=1.0,
                )
            ]
        if "South America" in query:
            return []
        return [
            RetrievedPassage(
                Passage(
                    id="cloud-provisional",
                    domain=domain,
                    text="The toolchain service is available.",
                ),
                score=1.0,
            )
        ]


def test_pipeline_refines_retrieval_on_final_transcript() -> None:
    retriever = FakeRetriever()
    pipeline = StreamingRagPipeline(retriever, refine_on_final=True)
    response = pipeline.run(simulate_chunks("I heard the toolchain is not available in South America."), "cloud")
    assert retriever.queries == [
        "I heard the toolchain is not available in",
        "I heard the toolchain is not available in South America.",
    ]
    assert [event.trigger for event in response.retrieval_events] == ["provisional", "final"]
    assert response.citations == ["cloud-final"]


def test_pipeline_keeps_provisional_passages_when_final_refresh_is_empty() -> None:
    retriever = FakeRetriever(final_has_passages=False)
    pipeline = StreamingRagPipeline(retriever, refine_on_final=True)
    response = pipeline.run(simulate_chunks("I heard the toolchain is not available in South America."), "cloud")
    assert retriever.queries == [
        "I heard the toolchain is not available in",
        "I heard the toolchain is not available in South America.",
    ]
    assert response.citations == ["cloud-provisional"]


def test_pipeline_deduplicates_identical_retrieval_chunks() -> None:
    retriever = FakeRetriever()
    chunks = [
        TranscriptChunk(0.8, "I heard the toolchain is not available in", False, 0.8),
        TranscriptChunk(1.6, "I heard the toolchain is not available in", True, 0.95),
    ]
    pipeline = StreamingRagPipeline(retriever, refine_on_final=True)
    pipeline.run(chunks, "cloud")
    assert retriever.queries == ["I heard the toolchain is not available in"]
