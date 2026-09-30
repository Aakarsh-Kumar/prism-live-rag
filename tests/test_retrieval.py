from pathlib import Path
import threading

import pytest

from prism_live_rag.config import Settings
from prism_live_rag.data import corpus_path, load_query_tasks
from prism_live_rag.embeddings import build_encoder
from prism_live_rag.expansion import expand_query
from prism_live_rag.retrieval import HybridRetriever, timeout_after


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
        build_encoder("hash"),
        settings.rrf_k,
        settings.sparse_weight,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
    )
    hits = retriever.search(task.query, "cloud", limit=5, candidate_limit=100)
    assert task.qrel_passage_ids[0] in {hit.passage.id for hit in hits}


def test_sparse_index_indexes_links_that_cleaning_lifted_out_of_prose() -> None:
    """Loader-side cleaning moves URLs from ``text`` into ``links``.

    If the sparse leg ignores ``links`` that vocabulary is destroyed rather than
    relocated, which measured as ~1 point of BM25 R@1 and R@10 on the 191 qrel tasks.
    """
    from prism_live_rag.models import Passage
    from prism_live_rag.retrieval import SparseIndex

    passage = Passage(
        id="p1",
        domain="cloud",
        text="Configure the content delivery network for your domain.",
        links=("https://cloud.ibm.com/docs/CDN?topic=hotlink-protection",),
    )
    index = SparseIndex([passage])
    hits = index.search("hotlink-protection", "cloud", limit=5)
    assert [h.id for h in hits] == ["p1"]


def test_timeout_context_is_safe_in_dashboard_worker_thread() -> None:
    errors = []

    def run_in_worker() -> None:
        try:
            with timeout_after(1, "worker operation"):
                pass
        except Exception as exc:  # captured so the assertion runs in this thread
            errors.append(exc)

    worker = threading.Thread(target=run_in_worker)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []
