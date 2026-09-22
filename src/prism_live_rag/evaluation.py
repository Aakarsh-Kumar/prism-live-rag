from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from statistics import mean
from typing import Callable

from .controller import RuleBasedRetrievalController, simulate_chunks
from .data import iter_passages
from .models import QueryTask, RagResponse


@dataclass(frozen=True)
class RetrievalEvalRow:
    task_id: str
    query: str
    qrels: tuple[str, ...]
    retrieved: tuple[str, ...]
    hit_rank: int | None
    early_retrieval: bool


@dataclass(frozen=True)
class ProviderEvalRow:
    task_id: str
    query: str
    citations: tuple[str, ...]
    citation_ids_valid: bool
    qrel_citation_hit: bool
    abstained: bool
    early_retrieval: bool
    latency_ms: int
    uncertainty: str | None


def evaluate_retrieval(
    tasks: list[QueryTask],
    search: Callable[[str], list[str]],
    *,
    limit: int,
    controller: RuleBasedRetrievalController | None = None,
) -> dict:
    controller = controller or RuleBasedRetrievalController()
    rows: list[RetrievalEvalRow] = []
    for task in tasks[:limit]:
        retrieved = tuple(search(task.query))
        qrels = set(task.qrel_passage_ids)
        hit_rank = next((index for index, pid in enumerate(retrieved, start=1) if pid in qrels), None)
        early_retrieval = _first_retrieval_is_early(task.query, controller)
        rows.append(
            RetrievalEvalRow(
                task_id=task.task_id,
                query=task.query,
                qrels=task.qrel_passage_ids,
                retrieved=retrieved,
                hit_rank=hit_rank,
                early_retrieval=early_retrieval,
            )
        )
    task_count = len(rows)
    hit_count = sum(row.hit_rank is not None for row in rows)
    reciprocal_ranks = [1.0 / row.hit_rank for row in rows if row.hit_rank]
    true_recalls = [
        len(set(row.retrieved) & set(row.qrels)) / len(row.qrels)
        for row in rows
        if row.qrels
    ]
    return {
        "mode": "retrieval",
        "tasks": task_count,
        "success_at_k": _ratio(hit_count, task_count),
        "recall_at_k": _ratio(sum(true_recalls), len(true_recalls)),
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "early_retrieval_rate": _ratio(sum(row.early_retrieval for row in rows), task_count),
        "misses": [
            {
                "task_id": row.task_id,
                "query": row.query,
                "qrels": list(row.qrels),
                "retrieved": list(row.retrieved),
            }
            for row in rows
            if row.hit_rank is None
        ],
    }


def evaluate_provider(
    tasks: list[QueryTask],
    run_pipeline: Callable[[str], RagResponse],
    *,
    corpus_ids: set[str],
    limit: int,
) -> dict:
    rows: list[ProviderEvalRow] = []
    for task in tasks[:limit]:
        started = time.perf_counter()
        response = run_pipeline(task.query)
        latency_ms = int((time.perf_counter() - started) * 1000)
        citations = tuple(response.citations)
        citation_ids_valid = all(citation in corpus_ids for citation in citations)
        qrel_citation_hit = bool(set(citations) & set(task.qrel_passage_ids))
        rows.append(
            ProviderEvalRow(
                task_id=task.task_id,
                query=task.query,
                citations=citations,
                citation_ids_valid=citation_ids_valid,
                qrel_citation_hit=qrel_citation_hit,
                abstained=not citations and response.uncertainty is not None,
                early_retrieval=bool(response.retrieval_events and response.retrieval_events[0].trigger == "provisional"),
                latency_ms=latency_ms,
                uncertainty=response.uncertainty,
            )
        )
    task_count = len(rows)
    return {
        "mode": "provider",
        "tasks": task_count,
        "citation_valid_rate": _ratio(sum(row.citation_ids_valid for row in rows), task_count),
        "qrel_citation_hit_rate": _ratio(sum(row.qrel_citation_hit for row in rows), task_count),
        "abstention_rate": _ratio(sum(row.abstained for row in rows), task_count),
        "early_retrieval_rate": _ratio(sum(row.early_retrieval for row in rows), task_count),
        "latency_ms": {
            "mean": int(mean([row.latency_ms for row in rows])) if rows else 0,
            "max": max([row.latency_ms for row in rows], default=0),
        },
        "failures": [
            {
                "task_id": row.task_id,
                "query": row.query,
                "citations": list(row.citations),
                "citation_ids_valid": row.citation_ids_valid,
                "qrel_citation_hit": row.qrel_citation_hit,
                "abstained": row.abstained,
                "uncertainty": row.uncertainty,
            }
            for row in rows
            if not row.citation_ids_valid or row.abstained
        ],
    }


@lru_cache(maxsize=4)
def corpus_id_set(data_dir, domain: str) -> set[str]:
    return {passage.id for passage in iter_passages(data_dir, domain)}


def _first_retrieval_is_early(query: str, controller: RuleBasedRetrievalController) -> bool:
    for chunk in simulate_chunks(query):
        if controller.decide(chunk) == "Retrieve":
            return not chunk.is_final
    return False


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
