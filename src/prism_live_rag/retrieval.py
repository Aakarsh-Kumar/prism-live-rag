from __future__ import annotations

import math
import signal
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .data import iter_passages
from .embeddings import Encoder, tokenize
from .expansion import expand_query
from .models import Passage, RetrievedPassage


RETRIEVAL_LEGS = ("hybrid", "sparse", "dense")


# Lazy-load cross-encoder to avoid import overhead when not needed
_cross_encoder = None


def get_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    *,
    local_files_only: bool = False,
):
    """Lazy-load cross-encoder reranker."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(
                model_name, max_length=512, local_files_only=local_files_only
            )
        except ImportError:
            raise RuntimeError(
                "sentence-transformers required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )
    return _cross_encoder


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
    # SIGALRM is process-wide and Python permits installing it only from the
    # main thread. Dashboard runs execute in a worker; previously the resulting
    # ValueError was swallowed by LanceIndex.exists(), making a healthy dense
    # table look absent and silently forcing sparse-only retrieval. LanceDB's
    # synchronous calls cannot be safely interrupted from a timer thread, so
    # retain the hard timeout on the main thread and avoid corrupting worker use.
    if threading.current_thread() is not threading.main_thread():
        yield
        return

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
        # Loader-side cleaning lifts URLs out of the prose into ``Passage.links``
        # (docs/corpus-spec.md §3.5). They must be indexed here or the lexical leg loses
        # that vocabulary outright: measured on 191 qrel tasks, omitting them cost
        # ~1 point of BM25 R@1 and R@10.
        self.doc_terms = [
            Counter(tokenize(passage.title + " " + passage.text + " " + " ".join(passage.links)))
            for passage in passages
        ]
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
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        reranker_top_k: int = 50,
        reranker_local_files_only: bool = False,
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

        # Cross-encoder reranking
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.reranker_top_k = reranker_top_k
        self.reranker_local_files_only = reranker_local_files_only
        self.reranker_status = "configured; awaiting candidates" if use_reranker else "disabled"
        self._reranker = None
        self.last_search_stats = {"dense_results": 0, "sparse_results": 0, "queries": 0}
        self.last_intent_results: list[list[RetrievedPassage]] = []

    def _get_reranker(self):
        """Lazy-load reranker on first use."""
        if self._reranker is None:
            self._reranker = get_cross_encoder(
                self.reranker_model,
                local_files_only=self.reranker_local_files_only,
            )
        return self._reranker

    def _rerank_passages(
        self,
        query: str,
        passages: list[RetrievedPassage],
        top_k: int,
    ) -> list[RetrievedPassage]:
        """Rerank passages using cross-encoder.

        Args:
            query: User query
            passages: Initial retrieved passages
            top_k: Number of top results to return after reranking

        Returns:
            Reranked passages with updated scores
        """
        if not passages:
            return []

        reranker = self._get_reranker()

        # Prepare query-passage pairs for cross-encoder
        pairs = [(query, p.passage.text) for p in passages]

        # Get relevance scores from cross-encoder
        scores = reranker.predict(pairs)

        # Create new RetrievedPassage objects with reranker scores
        reranked = []
        for passage, score in zip(passages, scores):
            reranked.append(RetrievedPassage(
                passage=passage.passage,
                score=float(score),  # Cross-encoder relevance score
                dense_rank=passage.dense_rank,
                sparse_rank=passage.sparse_rank,
            ))

        # Sort by new scores and return top_k
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

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

        # Get initial candidates via RRF
        self.last_search_stats = {
            "dense_results": len(dense),
            "sparse_results": len(sparse),
            "queries": 1,
        }
        initial_limit = self.reranker_top_k if self.use_reranker else limit
        candidates = weighted_rrf(dense, sparse, self.rrf_k, self.sparse_weight, initial_limit)

        # Apply cross-encoder reranking if enabled
        if self.use_reranker and candidates:
            try:
                reranked = self._rerank_passages(query, candidates, limit)
                self.reranker_status = "applied"
                return reranked
            except (RuntimeError, OSError, ImportError) as exc:
                # The dashboard must remain usable offline and must never fetch
                # model weights as a side effect of a judge pressing Run.
                self.use_reranker = False
                self.reranker_status = f"unavailable; RRF fallback ({type(exc).__name__})"

        return candidates[:limit]

    def search_multi_query(
        self,
        queries: list[str],
        domain: str,
        limit: int = 5,
        candidate_limit: int = 30,
        query_weights: list[float] | None = None,
    ) -> list[RetrievedPassage]:
        """Parallel multi-query retrieval with nested RRF fusion.

        Samsung Theme 04 specification: Handle 2-4 parallel sub-queries with
        weighted fusion across both query-level and method-level results.

        Args:
            queries: List of sub-queries (2-4 per Samsung constraint)
            domain: Search domain (cloud/govt)
            limit: Final result count
            candidate_limit: Candidates per query per method
            query_weights: Optional weights per query (default: equal weighting)

        Returns:
            Fused results with nested RRF scoring
        """
        if not queries:
            self.last_intent_results = []
            return []

        # Default equal weighting
        if query_weights is None:
            query_weights = [1.0 / len(queries)] * len(queries)
        elif len(query_weights) != len(queries):
            raise ValueError(f"Query weights ({len(query_weights)}) must match queries ({len(queries)})")

        # Step 1: Parallel retrieval for each sub-query
        all_query_results: list[list[RetrievedPassage]] = []
        query_stats: list[dict[str, int]] = []

        for query in queries:
            # Each query gets hybrid retrieval (dense + sparse → RRF)
            query_results = self.search(query, domain, limit=candidate_limit, candidate_limit=candidate_limit)
            all_query_results.append(query_results)
            query_stats.append(dict(self.last_search_stats))

        self.last_search_stats = {
            "dense_results": sum(item["dense_results"] for item in query_stats),
            "sparse_results": sum(item["sparse_results"] for item in query_stats),
            "queries": len(query_stats),
        }

        # Keep each intent's already-ranked evidence available to synthesis.
        # A global top-k must not erase a narrower intent merely because its
        # passages do not also rank highly for the other questions.
        self.last_intent_results = [rows[:limit] for rows in all_query_results]
        # Step 2: Nested RRF fusion across queries
        return self._nested_rrf_fusion(all_query_results, query_weights, limit)

    def _nested_rrf_fusion(
        self,
        query_results: list[list[RetrievedPassage]],
        query_weights: list[float],
        limit: int,
        k: int = 60,
    ) -> list[RetrievedPassage]:
        """Implement nested RRF fusion for multi-query results.

        Nested RRF: First-level RRF within each query (dense+sparse),
        Second-level RRF across queries with weighting.

        This handles the "multi-intent fusion" requirement from Samsung docs.
        """
        if not query_results:
            return []

        # Collect all unique passages with their cross-query rankings
        passage_data: dict[str, dict] = {}

        for query_idx, results in enumerate(query_results):
            query_weight = query_weights[query_idx]

            for rank, retrieved_passage in enumerate(results, start=1):
                passage_id = retrieved_passage.passage.id

                if passage_id not in passage_data:
                    passage_data[passage_id] = {
                        'passage': retrieved_passage.passage,
                        'query_ranks': {},
                        'query_scores': {},
                        'dense_ranks': [],
                        'sparse_ranks': [],
                    }

                # Store this query's rank and score for the passage
                passage_data[passage_id]['query_ranks'][query_idx] = rank
                passage_data[passage_id]['query_scores'][query_idx] = retrieved_passage.score

                # Accumulate method-level ranks for observability
                if retrieved_passage.dense_rank is not None:
                    passage_data[passage_id]['dense_ranks'].append(retrieved_passage.dense_rank)
                if retrieved_passage.sparse_rank is not None:
                    passage_data[passage_id]['sparse_ranks'].append(retrieved_passage.sparse_rank)

        # Calculate nested RRF scores
        final_scores: list[tuple[str, float]] = []

        for passage_id, data in passage_data.items():
            # Nested RRF: sum weighted reciprocal ranks across queries
            total_score = 0.0

            for query_idx, rank in data['query_ranks'].items():
                query_weight = query_weights[query_idx]
                # RRF formula: weight / (k + rank)
                total_score += query_weight / (k + rank)

            final_scores.append((passage_id, total_score))

        # Sort by score and create final results
        final_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for passage_id, score in final_scores[:limit]:
            data = passage_data[passage_id]

            # Calculate representative method ranks for observability
            avg_dense_rank = (
                sum(data['dense_ranks']) / len(data['dense_ranks'])
                if data['dense_ranks'] else None
            )
            avg_sparse_rank = (
                sum(data['sparse_ranks']) / len(data['sparse_ranks'])
                if data['sparse_ranks'] else None
            )

            results.append(RetrievedPassage(
                passage=data['passage'],
                score=score,  # Nested RRF score
                dense_rank=int(avg_dense_rank) if avg_dense_rank else None,
                sparse_rank=int(avg_sparse_rank) if avg_sparse_rank else None,
            ))

        return results
