from __future__ import annotations

import time
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from statistics import mean
from typing import Callable

from .controller import RuleBasedRetrievalController, TranscriptChunk, simulate_chunks
from .controller_v2 import ProductionSemanticController
from .data import iter_passages
from .decomposer import MultiIntentDecomposer
from .embeddings import tokenize
from .models import QueryTask, RagResponse
from .synthesis import CONTENT_STOPWORDS


def _normalize_asr_hypothesis(text: str) -> str:
    """Normalize harmless ASR casing, spacing, and punctuation-only revisions."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


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
    controller: RuleBasedRetrievalController | ProductionSemanticController | None = None,
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


def evaluate_streaming(
    streams,
    run_pipeline,
    *,
    limit: int | None = None,
    min_words: int = 5,
) -> dict:
    rows: list[dict] = []
    for stream in streams[:limit] if limit is not None else streams:
        response = run_pipeline(list(stream.chunks))
        first_retrieve = None
        for event in response.retrieval_events:
            if event.trigger == "provisional":
                first_retrieve = event.timestamp_s
                break
        rows.append(
            {
                "stream_id": stream.stream_id,
                "category": stream.category,
                "stability_chunk_index": stream.stability_chunk_index,
                "settling_ms": stream.settling_ms,
                "first_retrieve_timestamp": first_retrieve,
                "retrieval_events": len(response.retrieval_events),
                "word_count": len(stream.chunks[-1].text.split()) if stream.chunks else 0,
            }
        )
    eligible = [r for r in rows if r["category"] == "early_retrieval" and r["word_count"] >= min_words]
    negatives = [r for r in rows if r["category"] == "no_retrieval"]
    early_hits = [r for r in eligible if r["first_retrieve_timestamp"] is not None]
    false_triggers = [r for r in negatives if r["retrieval_events"] > 0]
    settling = [r["settling_ms"] for r in eligible if r["settling_ms"] >= 0]
    return {
        "mode": "streaming",
        "tasks": len(rows),
        "min_words": min_words,
        "eligible": len(eligible),
        "early_retrieval_rate": _ratio(len(early_hits), len(eligible)),
        "false_trigger_rate": _ratio(len(false_triggers), len(negatives)),
        "settling_ms": {
            "mean": int(mean(settling)) if settling else 0,
            "max": max(settling, default=0),
        },
        "misses": [r for r in eligible if r["first_retrieve_timestamp"] is None],
        "false_triggers": [r["stream_id"] for r in false_triggers],
    }


def evaluate_decomposition(
    streams,
    decomposer: MultiIntentDecomposer,
    *,
    single_queries: list[dict] | None = None,
    limit: int | None = None,
    min_intent_f1: float = 0.5,
    resume_rows: dict[str, dict] | None = None,
    on_case: Callable[[dict], None] | None = None,
    retry_fallbacks: bool = False,
) -> dict:
    """Score final intent isolation while exercising each streamed retrieval point.

    Accepts both ``SimulatedStream`` objects and JSONL-style case dictionaries. New
    benchmark rows should provide ``chunks`` and ``expected_intents``; legacy
    ``multi_intent`` streams and separate single-query files remain supported.
    """
    candidates = list(streams)
    if limit is not None:
        candidates = candidates[:limit]
    if single_queries:
        candidates.extend(single_queries if limit is None else single_queries[:limit])

    rows = []
    resumed_case_ids = set()
    for case in candidates:
        case_id = str(_decomposition_value(case, "case_id", _decomposition_value(case, "stream_id", "unknown")))
        cached_case = (resume_rows or {}).get(case_id)
        cached_has_fallback = bool(cached_case) and (
            cached_case.get("final_provider_status") == "fallback"
            or any(
                item.get("provider_status") == "fallback"
                for item in cached_case.get("retrieval_observations", [])
            )
        )
        if cached_case and (not retry_fallbacks or not cached_has_fallback):
            rows.append(cached_case)
            resumed_case_ids.add(case_id)
            continue
        if cached_case:
            resumed_case_ids.add(case_id)
        cached_observations = {
            _normalize_asr_hypothesis(item["query"]): item
            for item in (cached_case or {}).get("retrieval_observations", [])
            if item.get("provider_status") == "success"
        }
        category = _decomposition_value(case, "category", "single_intent")
        category = "multi_intent" if category in {"compound", "multi_intent"} else "single_intent"
        expected = _decomposition_expected(case, category)
        chunks = _decomposition_chunks(case)
        controller = RuleBasedRetrievalController()
        observations = []
        seen_hypotheses = set()
        for chunk in chunks:
            decision = controller.decide(chunk)
            if decision == "Retrieve":
                normalized = _normalize_asr_hypothesis(chunk.text)
                if normalized in seen_hypotheses:
                    continue
                seen_hypotheses.add(normalized)
                cached_observation = cached_observations.get(normalized)
                if cached_observation is not None:
                    observations.append(cached_observation)
                    continue
                result = decomposer.decompose(chunk.text)
                observations.append({
                    "timestamp_s": chunk.timestamp_s,
                    "is_final": chunk.is_final,
                    "query": chunk.text,
                    "predicted_intents": result.sub_queries,
                    "is_multi_intent": result.is_multi_intent,
                    "reason": result.decomposition_reason,
                    "processing_time_ms": result.processing_time_ms,
                    "provider_status": getattr(result, "provider_status", "not_applicable"),
                    "provider_error": getattr(result, "provider_error", None),
                })

        final_chunk = next((chunk for chunk in reversed(chunks) if chunk.is_final), chunks[-1])
        final_observation = next(
            (item for item in reversed(observations) if item["is_final"]), None
        )
        if final_observation is None:
            normalized_final = _normalize_asr_hypothesis(final_chunk.text)
            equivalent_observation = next(
                (item for item in reversed(observations)
                 if _normalize_asr_hypothesis(item["query"]) == normalized_final),
                None,
            )
            if equivalent_observation is not None:
                final_observation = equivalent_observation
                predicted = final_observation["predicted_intents"]
                detected = final_observation["is_multi_intent"]
                final_reason = final_observation["reason"]
                final_latency_ms = final_observation["processing_time_ms"]
            elif (
                cached_case
                and cached_case.get("final_provider_status") == "success"
                and _normalize_asr_hypothesis(cached_case.get("final_query", ""))
                == normalized_final
            ):
                predicted = cached_case["predicted_intents"]
                detected = cached_case["is_multi_intent"]
                final_reason = cached_case["decomposition_reason"]
                final_latency_ms = cached_case["decomposition_latency_ms"]
                final_observation = {
                    "provider_status": "success",
                    "provider_error": None,
                }
            else:
                final_result = decomposer.decompose(final_chunk.text)
                predicted = final_result.sub_queries
                detected = final_result.is_multi_intent
                final_reason = final_result.decomposition_reason
                final_latency_ms = final_result.processing_time_ms
                final_observation = {
                    "provider_status": getattr(final_result, "provider_status", "not_applicable"),
                    "provider_error": getattr(final_result, "provider_error", None),
                }
        else:
            predicted = final_observation["predicted_intents"]
            detected = final_observation["is_multi_intent"]
            final_reason = final_observation["reason"]
            final_latency_ms = final_observation["processing_time_ms"]
        matched = _match_intents(expected, predicted, min_intent_f1)
        full_match = (
            category == "multi_intent"
            and len(expected) > 1
            and len(predicted) == len(expected)
            and len(matched) == len(expected)
        )
        row = {
            "case_id": case_id,
            "domain": str(_decomposition_value(case, "domain", "")),
            "split": str(_decomposition_value(case, "split", "")),
            "review_status": str(_decomposition_value(case, "review_status", "legacy_unreviewed")),
            "provenance": _decomposition_value(case, "provenance", {}),
            "category": category,
            "final_query": final_chunk.text,
            "expected_intents": expected,
            "predicted_intents": predicted,
            "expected_count": len(expected),
            "predicted_count": len(predicted),
            "matched_intents": len(matched),
            "is_multi_intent": detected,
            "decomposition_reason": final_reason,
            "decomposition_latency_ms": final_latency_ms,
            "full_compound_match": full_match,
            "retrieval_observations": observations,
            "final_provider_status": (
                final_observation.get("provider_status", "not_applicable")
                if final_observation else "not_applicable"
            ),
            "final_provider_error": (
                final_observation.get("provider_error") if final_observation else None
            ),
        }
        rows.append(row)
        if on_case is not None:
            on_case(row)

    compounds = [row for row in rows if row["category"] == "multi_intent"]
    singles = [row for row in rows if row["category"] == "single_intent"]
    matched_compound = sum(row["matched_intents"] for row in compounds)
    expected_compound = sum(row["expected_count"] for row in compounds)
    predicted_compound = sum(row["predicted_count"] for row in compounds)
    full_compounds = sum(row["full_compound_match"] for row in compounds)
    over_split = sum(row["predicted_count"] > 1 for row in singles)
    missed_detection = sum(not row["is_multi_intent"] for row in compounds)
    provider_observations = [
        observation
        for row in rows
        for observation in row["retrieval_observations"]
        if observation["provider_status"] != "not_applicable"
    ]
    provider_final_rows = [
        row for row in rows if row["final_provider_status"] != "not_applicable"
    ]
    provider_success_rows = [
        row for row in provider_final_rows if row["final_provider_status"] == "success"
    ]
    model_full_compounds = sum(
        row["full_compound_match"] for row in provider_success_rows
        if row["category"] == "multi_intent"
    )
    model_compounds = sum(row["category"] == "multi_intent" for row in provider_final_rows)
    model_success_compounds = sum(row["category"] == "multi_intent" for row in provider_success_rows)
    health = decomposer.get_health_status()
    if health.get("type") == "llm_based":
        calls = []
        for row in rows:
            calls.extend(row["retrieval_observations"])
            final_query = _normalize_asr_hypothesis(row["final_query"])
            final_is_recorded = any(
                _normalize_asr_hypothesis(item["query"]) == final_query
                for item in row["retrieval_observations"]
            )
            if not final_is_recorded:
                calls.append({
                    "provider_status": row["final_provider_status"],
                    "provider_error": row["final_provider_error"],
                    "reason": row["decomposition_reason"],
                    "processing_time_ms": row["decomposition_latency_ms"],
                    "is_multi_intent": row["is_multi_intent"],
                })
        failure_calls = [
            call for call in calls
            if call.get("provider_status") == "fallback"
            and str(call.get("reason", "")).startswith("LLM failed")
        ]
        fallback_calls = [call for call in calls if call.get("provider_status") == "fallback"]
        latencies = [float(call.get("processing_time_ms", 0.0) or 0.0) for call in calls]
        health = {
            "type": "llm_based",
            "total_queries": len(calls),
            "multi_intent_detected": sum(bool(call.get("is_multi_intent")) for call in calls),
            "multi_intent_rate": _ratio(
                sum(bool(call.get("is_multi_intent")) for call in calls), len(calls)
            ),
            "llm_failures": len(failure_calls),
            "error_rate": _ratio(len(failure_calls), len(calls)),
            "fallback_uses": len(fallback_calls),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "circuit_breaker_activations": sum(
                "Circuit breaker active" in str(call.get("reason", ""))
                for call in fallback_calls
            ),
            "circuit_open_at_evaluation_end": None,
            "consecutive_failures_at_evaluation_end": None,
            "health_reconstructed_from_case_outcomes": True,
        }

    return {
        "mode": "decomposition",
        "dataset_note": (
            "Final scores use the final ASR hypothesis; retrieval_observations record "
            "decomposition at streamed controller retrieval points. Benchmark status "
            "and provenance are supplied by the input dataset."
        ),
        "streams": len(rows),
        "resumed_cases": len(resumed_case_ids),
        "compound_queries": len(compounds),
        "single_queries": len(singles),
        "review_status_counts": dict(Counter(row["review_status"] for row in rows)),
        "decomposer_health": health,
        "provider_outcomes": {
            "llm_decomposition_checkpoints": len(provider_observations),
            "successful_llm_decomposition_checkpoints": sum(
                item["provider_status"] == "success" for item in provider_observations
            ),
            "fallback_llm_decomposition_checkpoints": sum(
                item["provider_status"] == "fallback" for item in provider_observations
            ),
            "final_provider_evaluated_cases": len(provider_final_rows),
            "final_provider_success_cases": len(provider_success_rows),
            "final_provider_fallback_cases": len(provider_final_rows) - len(provider_success_rows),
            "provider_statuses_from_latest_checkpoint_results": sum(
                call.get("provider_status") == "success"
                or (call.get("provider_status") == "fallback"
                    and str(call.get("reason", "")).startswith("LLM failed"))
                for call in calls
            ) if health.get("type") == "llm_based" else 0,
            "successful_model_compound_accuracy": _ratio(model_full_compounds, model_success_compounds),
            "model_compound_accuracy_with_failures_counted_incorrect": _ratio(
                model_full_compounds, model_compounds
            ),
            "note": "System compound_success_rate includes fallback output; model accuracy is reported separately and provider failures are not presented as model successes.",
        },
        "compound_success_rate": _ratio(full_compounds, len(compounds)),
        "compound_intent_recall": _ratio(matched_compound, expected_compound),
        "compound_intent_precision": _ratio(matched_compound, predicted_compound),
        "compound_detection_rate": _ratio(
            len(compounds) - missed_detection, len(compounds)
        ),
        "single_overfragmentation_rate": _ratio(over_split, len(singles)),
        "single_correct_rate": _ratio(len(singles) - over_split, len(singles)),
        "matching": {
            "method": "one-to-one content-token F1 with simple plural normalization",
            "minimum_f1": min_intent_f1,
            "compound_success_requires": "all intents matched and predicted count equals expected count",
            "limitation": "Lexical proxy only; human review is required for paraphrased or ambiguous intent equivalence.",
        },
        "cases": rows,
        "failures": [
            row
            for row in rows
            if (row["category"] == "multi_intent" and not row["full_compound_match"])
            or (row["category"] == "single_intent" and row["predicted_count"] != 1)
        ],
    }


def decomposition_request_opportunities(case) -> int:
    """Count distinct Retrieve hypotheses plus a different final ASR hypothesis."""
    chunks = _decomposition_chunks(case)
    controller = RuleBasedRetrievalController()
    retrieved: set[str] = set()
    for chunk in chunks:
        if controller.decide(chunk) == "Retrieve":
            retrieved.add(_normalize_asr_hypothesis(chunk.text))
    if not chunks:
        return 0
    final_chunk = next((chunk for chunk in reversed(chunks) if chunk.is_final), chunks[-1])
    final_query = _normalize_asr_hypothesis(final_chunk.text)
    return len(retrieved) + int(final_query not in retrieved)


def _decomposition_value(case, key: str, default=None):
    if isinstance(case, dict):
        return case.get(key, default)
    return getattr(case, key, default)


def _decomposition_expected(case, category: str) -> list[str]:
    intents = _decomposition_value(case, "expected_intents")
    if intents is None:
        intents = _decomposition_value(case, "sub_intents", ())
    values = [
        str(item.get("text", "")) if isinstance(item, dict) else str(getattr(item, "text", item))
        for item in intents
    ]
    if values:
        return values
    query = str(_decomposition_value(case, "query", ""))
    return [query] if query else []


def _decomposition_chunks(case) -> list[TranscriptChunk]:
    raw = _decomposition_value(case, "chunks", ())
    chunks = [
        chunk if isinstance(chunk, TranscriptChunk) else TranscriptChunk(
            timestamp_s=float(chunk.get("timestamp_s", index)),
            text=str(chunk.get("text", "")),
            is_final=bool(chunk.get("is_final", False)),
            confidence=chunk.get("confidence"),
        )
        for index, chunk in enumerate(raw)
    ]
    if chunks:
        return chunks
    query = str(_decomposition_value(case, "query", ""))
    return simulate_chunks(query) if query else []


def _match_intents(
    expected: list[str], predicted: list[str], threshold: float
) -> list[tuple[int, int]]:
    """Find the maximum one-to-one intent matching with deterministic tie breaks."""
    def intent_tokens(value: str) -> Counter:
        tokens = []
        for token in tokenize(value):
            if token in CONTENT_STOPWORDS or token in {
                "mentioned", "exactly", "confused", "understand", "wondering", "please"
            }:
                continue
            if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            tokens.append(token)
        return Counter(tokens)

    expected_tokens = [intent_tokens(value) for value in expected]
    predicted_tokens = [intent_tokens(value) for value in predicted]
    scores = []
    for gold in expected_tokens:
        score_row = []
        for guess in predicted_tokens:
            overlap = sum((gold & guess).values())
            precision = overlap / max(sum(guess.values()), 1)
            recall = overlap / max(sum(gold.values()), 1)
            score_row.append(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        scores.append(score_row)

    @lru_cache(maxsize=None)
    def best(gold_index: int, used_predictions: int):
        if gold_index == len(expected):
            return 0, 0.0, ()
        best_count, best_score, best_pairs = best(gold_index + 1, used_predictions)
        for prediction_index, score in enumerate(scores[gold_index]):
            if used_predictions & (1 << prediction_index) or score < threshold:
                continue
            count, total_score, pairs = best(
                gold_index + 1, used_predictions | (1 << prediction_index)
            )
            candidate = (
                count + 1,
                total_score + score,
                ((gold_index, prediction_index),) + pairs,
            )
            if (candidate[0], candidate[1]) > (best_count, best_score):
                best_count, best_score, best_pairs = candidate
        return best_count, best_score, best_pairs

    return list(best(0, 0)[2])


def evaluate_curated_streaming(
    streams,
    run_pipeline,
    *,
    limit: int | None = None,
) -> dict:
    """Evaluate the full pipeline on interval-delivered curated ASR streams.

    The final ASR chunk is consumed as a replacement for the partial hypotheses, not
    as text appended to them.  ``behavior_match_proxy`` is deliberately named a
    proxy: citation/qrel overlap and abstention shape do not replace human grounding
    review or claim-level entailment evaluation.
    """
    selected = streams[:limit] if limit is not None else streams
    rows: list[dict] = []
    for stream in selected:
        response = run_pipeline(stream)
        final_timestamp = stream.chunks[-1].timestamp_s if stream.chunks else 0.0
        stability_timestamp = None
        if 0 <= stream.stability_chunk_index < len(stream.chunks):
            stability_timestamp = stream.chunks[stream.stability_chunk_index].timestamp_s
        provisional = [
            event
            for event in response.retrieval_events
            if event.trigger != "final" and event.timestamp_s < final_timestamp
        ]
        first_retrieve = provisional[0].timestamp_s if provisional else None
        citations = set(response.citations)
        qrels = set(stream.qrel_passage_ids)
        qrel_citation_hit = bool(citations & qrels)
        abstained = not citations and response.uncertainty is not None
        clarified = not citations and response.answer.rstrip().endswith("?")
        if stream.expected_behavior in {"answer", "partial_answer"}:
            behavior_match = qrel_citation_hit
        elif stream.expected_behavior == "abstain":
            behavior_match = abstained
        elif stream.expected_behavior == "clarify":
            behavior_match = clarified
        else:
            behavior_match = False
        rows.append(
            {
                "stream_id": stream.stream_id,
                "case_class": stream.case_class,
                "expected_behavior": stream.expected_behavior,
                "first_retrieve_timestamp": first_retrieve,
                "retrieved_pre_final": first_retrieve is not None,
                "retrieved_by_stability": (
                    first_retrieve is not None
                    and stability_timestamp is not None
                    and first_retrieve <= stability_timestamp
                ),
                "qrel_citation_hit": qrel_citation_hit,
                "abstained": abstained,
                "clarified": clarified,
                "behavior_match_proxy": behavior_match,
                "citations": sorted(citations),
            }
        )

    positive = [row for row in rows if row["case_class"] in {"answerable", "partial"}]
    by_class = {}
    for case_class in ("answerable", "partial", "unanswerable", "underspecified"):
        class_rows = [row for row in rows if row["case_class"] == case_class]
        by_class[case_class] = {
            "cases": len(class_rows),
            "behavior_match_proxy_rate": _ratio(
                sum(row["behavior_match_proxy"] for row in class_rows), len(class_rows)
            ),
            "qrel_citation_hit_rate": _ratio(
                sum(row["qrel_citation_hit"] for row in class_rows), len(class_rows)
            ),
        }
    return {
        "mode": "curated_streaming",
        "tasks": len(rows),
        "early_retrieval_rate": _ratio(
            sum(row["retrieved_pre_final"] for row in positive), len(positive)
        ),
        "before_stability_rate": _ratio(
            sum(row["retrieved_by_stability"] for row in positive), len(positive)
        ),
        "qrel_citation_hit_rate": _ratio(
            sum(row["qrel_citation_hit"] for row in positive), len(positive)
        ),
        "behavior_match_proxy_rate": _ratio(
            sum(row["behavior_match_proxy"] for row in rows), len(rows)
        ),
        "by_class": by_class,
        "failures": [row for row in rows if not row["behavior_match_proxy"]],
        "caveat": (
            "Behavior matching is a deterministic proxy, not human review or "
            "claim-level faithfulness evidence."
        ),
    }


def _first_retrieval_is_early(query: str, controller: RuleBasedRetrievalController | ProductionSemanticController) -> bool:
    controller.reset()
    for chunk in simulate_chunks(query):
        if controller.decide(chunk) == "Retrieve":
            return not chunk.is_final
    return False


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
