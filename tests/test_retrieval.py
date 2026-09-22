from pathlib import Path

import pytest

from prism_live_rag.config import Settings
from prism_live_rag.data import corpus_path, load_query_tasks
from prism_live_rag.expansion import expand_query
from prism_live_rag.retrieval import HybridRetriever


def test_expand_query_maps_south_america_to_cloud_region_terms() -> None:
    assert expand_query("Is the toolchain available in South America?") == (
        "is the toolchain available in sao paulo br sao"
    )


@pytest.mark.skipif(not corpus_path(Path("data"), "cloud").exists(), reason="cloud corpus is gitignored and absent")
def test_sparse_retrieval_finds_toolchain_south_america_qrel() -> None:
    settings = Settings(data_dir=Path("data"), lancedb_dir=Path(".cache/test-lancedb"))
    task = load_query_tasks(settings.data_dir, "cloud")[0]
    retriever = HybridRetriever(
        settings.data_dir,
        settings.lancedb_dir,
        settings.table_name,
        settings.embedding_dim,
        settings.rrf_k,
        settings.sparse_weight,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
    )
    hits = retriever.search(task.query, "cloud", limit=5, candidate_limit=100)
    assert task.qrel_passage_ids[0] in {hit.passage.id for hit in hits}
