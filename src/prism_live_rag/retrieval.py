from __future__ import annotations

import math
import signal
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .data import iter_passages
from .embeddings import Encoder, tokenize
from .expansion import expand_query
from .models import Passage, RetrievedPassage


RETRIEVAL_LEGS = ("hybrid", "sparse", "dense")


class LanceDbTimeout(RuntimeError):
    pass


def _writable_or_absent(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        test = path / ".probe"
        test.touch()
        test.unlink()
        return True
    except OSError:
        return False


@contextmanager
def timeout_after(seconds: int, label: str):
    def _raise_timeout(_signum, _frame):
        raise LanceDbTimeout(f"{label} timed out after {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class LanceIndex:
    """LanceDB dense index. The table name is scoped by the encoder so that
    different backends/dimensions can coexist without silently colliding."""

    def __init__(self, db_dir: Path, table_name: str, encoder: Encoder) -> None:
        self.db_dir = db_dir
        self.encoder = encoder
        self.dim = encoder.dim
        self.table_name = f"{table_name}__{encoder.name}"

    def build(
        self,
        data_dir: Path,
        domains: list[str],
        limit: int | None = None,
        timeout_s: int = 300,
        progress: bool = False,
    ) -> int:
        import lancedb

        self.db_dir.mkdir(parents=True, exist_ok=True)
        passages: list[Passage] = []
        for domain in domains:
            passages.extend(iter_passages(data_dir, domain, limit=limit))
        texts = [passage.text for passage in passages]
        vectors: list[list[float]] = []
        chunk = 20000
        for start in range(0, len(texts), chunk):
            vectors.extend(self.encoder.encode_passages(texts[start : start + chunk]))
            if progress:
                done = min(start + chunk, len(texts))
                print(
                    f"  embedded {done}/{len(texts)} passages on {self.encoder.device}",
                    flush=True,
                )
        rows = [
            {
                "id": passage.id,
                "domain": passage.domain,
                "title": passage.title,
                "url": passage.url,
                "text": passage.text,
                "vector": vector,
            }
            for passage, vector in zip(passages, vectors)
        ]
        with timeout_after(timeout_s, "LanceDB index build"):
            if not _writable_or_absent(self.db_dir):
                raise LanceDbTimeout(
                    f"{self.db_dir} is not writable (often a root-owned ./cache left behind by a "
                    "Docker index build). Fix with: sudo rm -rf .cache/lancedb"
                )
            db = lancedb.connect(str(self.db_dir))
            if self.table_name in db.table_names():
                db.drop_table(self.table_name)
            db.create_table(self.table_name, data=rows)
        return len(rows)

    def exists(self) -> bool:
        if not self.db_dir.exists():
            return False
        try:
            import lancedb

            with timeout_after(10, "LanceDB table check"):
                db = lancedb.connect(str(self.db_dir))
                return self.table_name in db.table_names()
        except Exception:
            return False

    def search(self, query: str, domain: str, limit: int) -> list[Passage]:
        import lancedb

        with timeout_after(10, "LanceDB dense search"):
            db = lancedb.connect(str(self.db_dir))
            table = db.open_table(self.table_name)
            rows = (
                table.search(self.encoder.encode_query(query))
                .where(f"domain = '{domain}'", prefilter=True)
                .limit(limit)
                .to_list()
            )
        return [
            Passage(
                id=str(row["id"]),
                domain=str(row["domain"]),
                title=str(row.get("title", "")),
                url=str(row.get("url", "")),
                text=str(row.get("text", "")),
            )
            for row in rows
        ]


class SparseIndex:
    def __init__(self, passages: list[Passage], bm25_k1: float = 1.2, bm25_b: float = 0.75) -> None:
        self.passages = passages
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.doc_terms = [Counter(tokenize(passage.title + " " + passage.text)) for passage in passages]
        self.doc_lengths = [sum(terms.values()) for terms in self.doc_terms]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freq: Counter[str] = Counter()
        for terms in self.doc_terms:
            self.doc_freq.update(terms.keys())

    def search(self, query: str, domain: str, limit: int) -> list[Passage]:
        query_terms = Counter(tokenize(query))
        total_docs = max(len(self.passages), 1)
        scored: list[tuple[float, Passage]] = []
        for passage, terms, doc_len in zip(self.passages, self.doc_terms, self.doc_lengths):
            if passage.domain != domain:
                continue
            score = 0.0
            normalized_length = 1.0 - self.bm25_b + self.bm25_b * (doc_len / max(self.avg_doc_length, 1.0))
            for term, query_tf in query_terms.items():
                if term not in terms:
                    continue
                idf = math.log(1.0 + (total_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                term_tf = terms[term]
                saturated_tf = (term_tf * (self.bm25_k1 + 1.0)) / (term_tf + self.bm25_k1 * normalized_length)
                score += idf * saturated_tf * query_tf
            if score > 0:
                scored.append((score, passage))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [passage for _, passage in scored[:limit]]


def weighted_rrf(
    dense: list[Passage],
    sparse: list[Passage],
    k: int = 60,
    sparse_weight: float = 0.35,
    limit: int = 5,
) -> list[RetrievedPassage]:
    by_id: dict[str, Passage] = {}
    scores: dict[str, float] = {}
    dense_rank: dict[str, int] = {}
    sparse_rank: dict[str, int] = {}

    for rank, passage in enumerate(dense, start=1):
        by_id[passage.id] = passage
        dense_rank[passage.id] = rank
        scores[passage.id] = scores.get(passage.id, 0.0) + 1.0 / (k + rank)
    for rank, passage in enumerate(sparse, start=1):
        by_id[passage.id] = passage
        sparse_rank[passage.id] = rank
        scores[passage.id] = scores.get(passage.id, 0.0) + sparse_weight / (k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [
        RetrievedPassage(
            passage=by_id[pid],
            score=score,
            dense_rank=dense_rank.get(pid),
            sparse_rank=sparse_rank.get(pid),
        )
        for pid, score in ranked
    ]


class HybridRetriever:
    def __init__(
        self,
        data_dir: Path,
        lancedb_dir: Path,
        table_name: str,
        encoder: Encoder,
        rrf_k: int,
        sparse_weight: float,
        corpus_limit: int | None = None,
        use_dense: bool = False,
        retrieval_leg: str = "hybrid",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> None:
        self.data_dir = data_dir
        self.encoder = encoder
        self.dense = LanceIndex(lancedb_dir, table_name, encoder)
        self.use_dense = use_dense
        if retrieval_leg not in RETRIEVAL_LEGS:
            raise ValueError(f"Unknown retrieval leg: {retrieval_leg!r}")
        self.retrieval_leg = retrieval_leg
        passages = []
        for domain in ("cloud", "govt"):
            passages.extend(iter_passages(data_dir, domain, limit=corpus_limit))
        self.sparse = SparseIndex(passages, bm25_k1=bm25_k1, bm25_b=bm25_b)
        self.rrf_k = rrf_k
        self.sparse_weight = sparse_weight

    def search(self, query: str, domain: str, limit: int = 5, candidate_limit: int = 30) -> list[RetrievedPassage]:
        expanded_query = expand_query(query)
        dense: list[Passage] = []
        sparse: list[Passage] = []
        if self.retrieval_leg in ("hybrid", "sparse"):
            sparse = self.sparse.search(expanded_query, domain, candidate_limit)
        if self.retrieval_leg in ("hybrid", "dense"):
            if self.use_dense and self.dense.exists():
                dense = self.dense.search(expanded_query, domain, candidate_limit)
            elif self.retrieval_leg == "dense":
                raise RuntimeError(
                    f"Dense retrieval requested but index table '{self.dense.table_name}' is "
                    "missing. Run `prism-rag index --embedding-backend <backend>` first."
                )
        return weighted_rrf(dense, sparse, self.rrf_k, self.sparse_weight, limit)
