from pathlib import Path

from prism_live_rag.config import Settings
from prism_live_rag.controller import simulate_chunks
from prism_live_rag.pipeline import StreamingRagPipeline
from prism_live_rag.retrieval import HybridRetriever


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
    )
    pipeline = StreamingRagPipeline(retriever)
    response = pipeline.run(simulate_chunks("I heard the toolchain is not available in South America."), "cloud")
    payload = response.to_dict()
    assert set(payload) == {"retrieval_events", "sub_queries", "answer", "citations", "uncertainty"}
    assert payload["sub_queries"]
    assert payload["answer"]

