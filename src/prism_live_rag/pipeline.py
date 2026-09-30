from __future__ import annotations

import time
import uuid
from typing import Callable, Iterable

from .controller import RuleBasedRetrievalController, SemanticRetrievalController, TranscriptChunk, is_presentation_only
from .controller_v2 import ProductionSemanticController
from .decomposer import LLMDecomposer, MultiIntentDecomposer, RuleBasedDecomposer, build_decomposer
from .llm_steps import ChatClient, provider_synthesize_answer, rewrite_query
from .models import ConversationSession, RagResponse, RetrievalEvent
from .providers import ProviderError
from .retrieval import HybridRetriever
from .synthesis import SENTENCE_RE, contextualize_query, focus_refinement_query, needs_clarification, present_previous_answer, refine_answer, route_to_provider, synthesize_answer


class StreamingRagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        controller: RuleBasedRetrievalController | SemanticRetrievalController | ProductionSemanticController | None = None,
        decomposer: MultiIntentDecomposer | None = None,
        *,
        synthesis_mode: str = "deterministic",
        evidence_client: ChatClient | None = None,
        generation_client: ChatClient | None = None,
        enable_query_rewrite: bool = False,
        refine_on_final: bool = True,
        enable_multi_intent: bool = False,
        verify_evidence_relevance: bool = False,
        strict_evidence: bool = False,
    ) -> None:
        self.retriever = retriever
        self.controller = controller or RuleBasedRetrievalController()
        self.decomposer = decomposer
        self.synthesis_mode = synthesis_mode
        self.evidence_client = evidence_client
        self.generation_client = generation_client
        self.enable_query_rewrite = enable_query_rewrite
        self.refine_on_final = refine_on_final
        self.enable_multi_intent = enable_multi_intent
        self.verify_evidence_relevance = verify_evidence_relevance
        self.strict_evidence = strict_evidence

        # Validation: multi-intent requires decomposer
        if self.enable_multi_intent and self.decomposer is None:
            # Default to rule-based decomposer as fallback
            self.decomposer = build_decomposer("rule_based")

    def run(
        self,
        chunks: Iterable[TranscriptChunk],
        domain: str = "cloud",
        context_turns: tuple[tuple[str, str], ...] = (),
        *,
        session: ConversationSession | None = None,
        refinement: bool = False,
        trace: bool = False,
        on_event: Callable[[dict], None] | None = None,
    ) -> RagResponse:
        """Consume a stream of transcript chunks and produce a grounded answer.

        ``chunks`` may be any iterable, including a generator, so text can arrive live
        from ASR rather than being materialised as a list first. The whole stream is
        consumed: a decision is recorded for every chunk, and the controller decides when
        retrieval fires. Because ``RuleBasedRetrievalController`` latches after its first
        stability-triggered fire (``_fired``) and always retrieves on ``is_final``, a
        normal stream yields at most two retrievals -- one provisional for latency, one on
        the final transcript to refine it. That is the "refines rather than restarts"
        behaviour the brief requires, and it is what makes late-constraint G5 refinement
        possible at all: an early break meant the final chunk was never seen.
        """
        started_at = time.perf_counter()
        response = RagResponse()
        if trace or session is not None:
            response.trace_id = uuid.uuid4().hex
        initial_usage = {}
        for name, client in (("evidence", self.evidence_client), ("generation", self.generation_client)):
            usage = getattr(client, "token_usage", None)
            if isinstance(usage, dict):
                initial_usage[name] = dict(usage)
        retrieval_ms = 0.0
        generation_ms = 0.0
        if refinement and (session is None or not session.answer):
            raise ValueError("refinement requires a session with a committed answer")
        if refinement and session.domain not in (None, domain):
            raise ValueError("refinement domain must match the existing session")
        response.previous_version = (
            session.version if session and session.answer else None
        )
        # Traced one-shot runs are version 1; refinements advance the session.
        response.version = session.version + 1 if session is not None else (1 if trace else None)
        retrieved_for = None
        passages = []
        active_query = None
        active_intent_queries: list[str] = []
        intent_passages: dict[str, list] = {}
        last_chunk = None
        self.controller.reset()

        for chunk in chunks:
            last_chunk = chunk
            self._notify(on_event, {
                "type": "chunk_processed",
                "timestamp_s": chunk.timestamp_s,
                "text": chunk.text,
                "is_final": chunk.is_final,
                "confidence": chunk.confidence,
            })
            decision = self.controller.decide(chunk)
            decision_payload = {
                "timestamp_s": chunk.timestamp_s,
                "text": chunk.text,
                "decision": decision,
                "reason": self.controller.last_reason,
            }
            response.decisions.append(decision_payload)
            self._notify(on_event, {"type": "decision", **decision_payload})
            if decision != "Retrieve" or retrieved_for == chunk.text:
                continue

            retrieved_for = chunk.text
            trigger_type = "refinement" if refinement else (
                "provisional" if not chunk.is_final else "final"
            )
            if trigger_type == "final" and not self.refine_on_final:
                # Caller opted out of refining an already-issued provisional answer.
                continue

            # Keep the ASR utterance separate from conversational context. In
            # particular, never ask the decomposer to segment a serialized chat
            # history: question marks in old turns are not new user intents.
            # Refinement input is deliberately a delta (e.g. "for California only").
            # Do not expand it back into the original query or session history.
            retrieval_query = chunk.text if refinement else contextualize_query(chunk.text, context_turns)
            if not refinement and self.enable_query_rewrite and self.evidence_client is not None:
                rewrite_eligible = (
                    self.synthesis_mode == "provider"
                    and chunk.is_final
                    and retrieval_query != chunk.text
                    and not needs_clarification(chunk.text, context_turns)
                )
                self._notify(on_event, {
                    "type": "llm_routing", "stage": "query_rewrite",
                    "eligible": rewrite_eligible,
                    "decision": "call" if rewrite_eligible else "skip",
                    "reason": "Resolve a supported conversational reference." if rewrite_eligible else "A standalone or unresolved query does not need model rewriting.",
                    "query": chunk.text,
                })
                if rewrite_eligible:
                    try:
                        retrieval_query = rewrite_query(self.evidence_client, retrieval_query)
                    except (ProviderError, ValueError):
                        pass

            # Multi-intent decomposition and parallel retrieval
            retrieval_started = time.perf_counter()
            self._notify(on_event, {
                "type": "retrieval_start",
                "timestamp_s": chunk.timestamp_s,
                "query": retrieval_query,
                "requested_trigger": trigger_type,
            })
            sub_query_start = len(response.sub_queries)
            if self.enable_multi_intent and self.decomposer is not None:
                decomposition_context = " ".join(
                    f"{speaker}: {text.strip()}"
                    for speaker, text in context_turns[-4:]
                    if text.strip()
                )
                decomposition_input = chunk.text
                decomposition_context = decomposition_context if not refinement else ""
                if isinstance(self.decomposer, LLMDecomposer):
                    cheap_result = RuleBasedDecomposer().decompose(
                        decomposition_input, context=decomposition_context
                    )
                    use_llm_decomposer = (
                        self.synthesis_mode == "provider" and cheap_result.is_multi_intent
                        and not needs_clarification(chunk.text, context_turns)
                    )
                    self._notify(on_event, {
                        "type": "llm_routing",
                        "stage": "decomposition",
                        "eligible": use_llm_decomposer,
                        "decision": "call" if use_llm_decomposer else "skip",
                        "reason": (
                            "Explicit compound utterance; use LLM to validate/split intents."
                            if use_llm_decomposer else
                            "No explicit compound question or provider mode is disabled; use local rule decomposition."
                        ),
                        "query": chunk.text,
                    })
                    decomposition_result = (
                        self.decomposer.decompose(
                            decomposition_input, context=decomposition_context
                        )
                        if use_llm_decomposer else cheap_result
                    )
                else:
                    self._notify(on_event, {
                        "type": "llm_routing",
                        "stage": "decomposition",
                        "eligible": False,
                        "decision": "skip",
                        "reason": "Rule-based decomposition is local and does not call an LLM.",
                        "query": chunk.text,
                    })
                    decomposition_result = self.decomposer.decompose(
                        decomposition_input, context=decomposition_context
                    )

                if decomposition_result.is_multi_intent:
                    # Multi-intent: parallel retrieval with nested RRF fusion
                    if not refinement:
                        trigger_type = "multi_intent"
                    sub_queries = decomposition_result.sub_queries
                    active_intent_queries = list(sub_queries)
                    retrieval_sub_queries = [
                        contextualize_query(query, context_turns)
                        if not refinement else query
                        for query in sub_queries
                    ]
                    self._notify(on_event, {"type": "search_dispatch", "timestamp_s": chunk.timestamp_s,
                                          "trigger": trigger_type, "queries": retrieval_sub_queries})
                    candidate_passages = self.retriever.search_multi_query(
                        retrieval_sub_queries, domain=domain
                    )
                    groups = getattr(self.retriever, "last_intent_results", [])
                    intent_passages = dict(zip(sub_queries, groups)) if len(groups) == len(sub_queries) else {}
                    # Trace every passage eligible for answer evidence, keeping
                    # fused order first and then per-intent coverage, deduplicated.
                    pool = {item.passage.id: item for item in candidate_passages}
                    for rows in intent_passages.values():
                        for item in rows:
                            pool.setdefault(item.passage.id, item)
                    candidate_passages = list(pool.values())

                    # Log multi-intent retrieval event
                    response.retrieval_events.append(
                        RetrievalEvent(
                            timestamp_s=chunk.timestamp_s,
                            query=chunk.text,
                            trigger=trigger_type,
                        )
                    )

                    # Store all sub-queries for telemetry
                    response.sub_queries.extend(retrieval_sub_queries)

                else:
                    # Single intent: standard retrieval
                    active_intent_queries = []
                    self._notify(on_event, {"type": "search_dispatch", "timestamp_s": chunk.timestamp_s,
                                          "trigger": trigger_type, "queries": [retrieval_query]})
                    candidate_passages = self.retriever.search(retrieval_query, domain=domain)
                    response.sub_queries.append(retrieval_query)
                    response.retrieval_events.append(
                        RetrievalEvent(
                            timestamp_s=chunk.timestamp_s,
                            query=chunk.text,
                            trigger=trigger_type,
                        )
                    )
            else:
                # Standard single-query retrieval (backwards compatible)
                active_intent_queries = []
                self._notify(on_event, {"type": "search_dispatch", "timestamp_s": chunk.timestamp_s,
                                      "trigger": trigger_type, "queries": [retrieval_query]})
                candidate_passages = self.retriever.search(retrieval_query, domain=domain)
                response.sub_queries.append(retrieval_query)
                response.retrieval_events.append(
                    RetrievalEvent(
                        timestamp_s=chunk.timestamp_s,
                        query=chunk.text,
                        trigger=trigger_type,
                    )
                )
            retrieval_ms += (time.perf_counter() - retrieval_started) * 1000
            self._notify(on_event, {
                "type": "retrieval_done",
                "timestamp_s": chunk.timestamp_s,
                "query": retrieval_query,
                "trigger": trigger_type,
                "retrieval_mode": getattr(self.retriever, "retrieval_leg", "unknown"),
                "dense_enabled": bool(getattr(self.retriever, "use_dense", False)),
                "reranker_enabled": bool(getattr(self.retriever, "use_reranker", False)),
                "reranker_status": getattr(self.retriever, "reranker_status", "unknown"),
                "retrieval_method_counts": dict(
                    getattr(self.retriever, "last_search_stats", {})
                ),
                "sub_queries": list(response.sub_queries[sub_query_start:]),
                "passages": [
                    {
                        "id": item.passage.id,
                        "domain": item.passage.domain,
                        "title": item.passage.title,
                        "url": item.passage.url,
                        "text": item.passage.text,
                        "score": item.score,
                        "dense_rank": item.dense_rank,
                        "sparse_rank": item.sparse_rank,
                    }
                    for item in candidate_passages
                ],
            })

            # Update passages if we got results, or keep previous if empty
            if refinement:
                # The final ASR hypothesis supersedes partials, including an empty
                # final search. Never ground the delta answer on stale partial text.
                if chunk.is_final or candidate_passages:
                    passages = candidate_passages
                    active_query = retrieval_query
            elif trigger_type == "final" or candidate_passages or not passages:
                passages = candidate_passages
                active_query = retrieval_query

        # Provisional searches remain in retrieval_events/observer events. The
        # final answer's intent list must not include obsolete ASR prefixes.
        if last_chunk is not None:
            response.sub_queries = list(active_intent_queries) or [last_chunk.text]

        if (
            not refinement and session is not None and session.answer
            and last_chunk is not None and last_chunk.is_final
            and is_presentation_only(last_chunk.text)
        ):
            response.answer = present_previous_answer(session.answer, last_chunk.text)
            response.citations = list(session.citations)
            response.claim_citations = {claim: list(ids) for claim, ids in session.claim_citations.items()}
            response.uncertainty = session.uncertainty
            response.stage_latency_ms = {
                "retrieval": round(retrieval_ms, 3),
                "generation": 0.0,
                "total": round((time.perf_counter() - started_at) * 1000, 3),
            }
            _record_cost(response, initial_usage, self.evidence_client, self.generation_client)
            session.commit(response, domain)
            self._notify_answer(on_event, response)
            return response

        if refinement and session is not None:
            delta = last_chunk.text.strip() if last_chunk is not None else ""
            response.applied_delta = delta or None
            if not passages:
                response.answer = session.answer
                response.citations = list(session.citations)
                response.claim_citations = {claim: list(ids) for claim, ids in session.claim_citations.items()}
                response.uncertainty = (
                    "The new constraint could not be grounded; the prior answer is preserved unchanged."
                )
                response.stage_latency_ms = {
                    "retrieval": round(retrieval_ms, 3),
                    "generation": 0.0,
                    "total": round((time.perf_counter() - started_at) * 1000, 3),
                }
                _record_cost(response, initial_usage, self.evidence_client, self.generation_client)
                session.commit(response, domain)
                self._notify_answer(on_event, response)
                return response
            delta_query = active_query or delta
            evidence_query, required_terms = focus_refinement_query(delta_query, session.answer)
            generation_started = time.perf_counter()
            use_provider, route_reason = route_to_provider(
                evidence_query, passages, multi_intent=False
            )
            self._notify(on_event, {
                "type": "llm_routing",
                "stage": "refinement_evidence",
                "eligible": self.synthesis_mode == "provider" and use_provider,
                "decision": "call" if self.synthesis_mode == "provider" and use_provider else "skip",
                "reason": route_reason if self.synthesis_mode == "provider" else "Provider mode is disabled.",
                "query": delta,
            })
            if self.synthesis_mode == "provider" and use_provider:
                delta_answer, delta_citations, delta_uncertainty = provider_synthesize_answer(
                    self.evidence_client, self.generation_client, evidence_query, passages,
                    verify_relevance=self.verify_evidence_relevance,
                )
            else:
                delta_answer, delta_citations, delta_uncertainty = synthesize_answer(
                    evidence_query,
                    passages,
                    require_direct_support=self.synthesis_mode == "provider" or self.strict_evidence,
                    required_terms=required_terms,
                    min_relevance_score=(
                        0.0 if getattr(self.retriever, "use_reranker", False) else None
                    ),
                )
            if delta_uncertainty or not delta_citations:
                response.answer = session.answer
                response.citations = list(session.citations)
                response.claim_citations = {claim: list(ids) for claim, ids in session.claim_citations.items()}
                response.uncertainty = (
                    "The new constraint could not be grounded; the prior answer is preserved unchanged."
                )
            else:
                response.answer, response.citations, response.uncertainty = refine_answer(
                    session.answer, session.citations, delta_answer, delta_citations, delta_query
                )
                response.claim_citations = {
                    claim: list(ids)
                    for claim, ids in session.claim_citations.items()
                    if claim in response.answer
                }
                response.claim_citations.update(_claim_citation_map(delta_answer, passages))
                final_sentences = [
                    sentence.strip() for sentence in SENTENCE_RE.split(response.answer)
                    if sentence.strip()
                ]
                if final_sentences and all(
                    sentence in response.claim_citations for sentence in final_sentences
                ):
                    response.citations = list(dict.fromkeys(
                        passage_id
                        for sentence in final_sentences
                        for passage_id in response.claim_citations[sentence]
                    ))
            generation_ms += (time.perf_counter() - generation_started) * 1000
            response.stage_latency_ms = {
                "retrieval": round(retrieval_ms, 3),
                "generation": round(generation_ms, 3),
                "total": round((time.perf_counter() - started_at) * 1000, 3),
            }
            _record_cost(response, initial_usage, self.evidence_client, self.generation_client)
            session.commit(response, domain)
            self._notify_answer(on_event, response)
            return response

        # Fallback: ensure we have at least one sub-query for synthesis
        if not response.sub_queries and last_chunk is not None:
            response.sub_queries = [last_chunk.text]
            active_query = last_chunk.text

        # Answer synthesis
        query_for_answer = active_query or response.sub_queries[-1]
        final_query = last_chunk.text if last_chunk is not None else query_for_answer
        presentation_request = is_presentation_only(final_query)
        # A retrieval rewrite is a search aid, not permission to change which
        # question evidence must answer. Validate against the real final turn.
        if last_chunk is not None and not presentation_request:
            query_for_answer = contextualize_query(final_query, context_turns)
        if not presentation_request and needs_clarification(final_query, context_turns):
            self._notify(on_event, {
                "type": "llm_routing", "stage": "answer_evidence",
                "eligible": False, "decision": "skip",
                "reason": "An unresolved reference requires clarification before answering.",
                "query": final_query,
            })
            response.answer = "Could you clarify what specific item or subject you mean?"
            response.citations = []
            response.uncertainty = "The request contains an unresolved conversational reference."
            response.stage_latency_ms = {
                "retrieval": round(retrieval_ms, 3),
                "generation": 0.0,
                "total": round((time.perf_counter() - started_at) * 1000, 3),
            }
            _record_cost(response, initial_usage, self.evidence_client, self.generation_client)
            self._notify_answer(on_event, response)
            return response
        generation_started = time.perf_counter()
        claim_citations = None
        if len(active_intent_queries) > 1:
            # Retrieval fusion must not collapse the answer back into one broad
            # query. Synthesize and cite each detected intent independently, and
            # make any unsupported intent explicit instead of silently dropping it.
            intent_answers: list[str] = []
            intent_citations: list[str] = []
            intent_claim_citations: dict[str, list[str]] = {}
            missing_intents: list[str] = []
            for intent_query in active_intent_queries:
                evidence_passages = intent_passages.get(intent_query, passages)
                intent_evidence_query = (
                    contextualize_query(intent_query, context_turns)
                    if not refinement else intent_query
                )
                use_provider, route_reason = route_to_provider(
                    intent_evidence_query, evidence_passages, multi_intent=True
                ) if not presentation_request else (
                    False, "Presentation request reuses available evidence without an LLM call."
                )
                self._notify(on_event, {
                    "type": "llm_routing",
                    "stage": "answer_evidence",
                    "eligible": self.synthesis_mode == "provider" and use_provider,
                    "decision": "call" if self.synthesis_mode == "provider" and use_provider else "skip",
                    "reason": route_reason if self.synthesis_mode == "provider" else "Provider mode is disabled.",
                    "query": intent_query,
                })
                if self.synthesis_mode == "provider" and use_provider:
                    intent_answer, intent_ids, intent_uncertainty = provider_synthesize_answer(
                        self.evidence_client,
                        self.generation_client,
                        intent_evidence_query,
                        evidence_passages,
                        verify_relevance=self.verify_evidence_relevance,
                    )
                else:
                    intent_answer, intent_ids, intent_uncertainty = synthesize_answer(
                        intent_evidence_query,
                        evidence_passages,
                        min_relevance_score=(
                            0.0 if getattr(self.retriever, "use_reranker", False) else None
                        ),
                        require_direct_support=self.synthesis_mode == "provider" or self.strict_evidence,
                    )
                if intent_uncertainty or not intent_ids:
                    missing_intents.append(intent_query)
                    intent_answers.append(
                        f"{intent_query}: I could not find sufficient cited evidence to answer this part."
                    )
                else:
                    intent_answers.append(f"{intent_query}: {intent_answer}")
                    intent_citations.extend(intent_ids)
                    # Do not assign every intent citation to every claim: only
                    # exact containing source passages support each sentence.
                    intent_claim_citations.update(_claim_citation_map(
                        intent_answer,
                        [item for item in passages if item.passage.id in intent_ids],
                    ))
            answer = "\n\n".join(intent_answers)
            citations = list(dict.fromkeys(intent_citations))
            uncertainty = (
                "One or more requested intents could not be grounded: "
                + "; ".join(missing_intents)
                if missing_intents
                else None
            )
            claim_citations = intent_claim_citations
        elif self.synthesis_mode == "provider":
            use_provider, route_reason = route_to_provider(
                final_query, passages, multi_intent=False
            ) if not presentation_request else (
                False, "Presentation request reuses available evidence without an LLM call."
            )
            self._notify(on_event, {
                "type": "llm_routing",
                "stage": "answer_evidence",
                "eligible": use_provider,
                "decision": "call" if use_provider else "skip",
                "reason": route_reason,
                "query": final_query,
            })
            if use_provider:
                answer, citations, uncertainty = provider_synthesize_answer(
                    self.evidence_client,
                    self.generation_client,
                    query_for_answer,
                    passages,
                    verify_relevance=self.verify_evidence_relevance,
                )
            else:
                answer, citations, uncertainty = synthesize_answer(
                    query_for_answer,
                    passages,
                    require_direct_support=not presentation_request,
                    min_relevance_score=(
                        0.0 if getattr(self.retriever, "use_reranker", False) else None
                    ),
                )
        else:
            answer, citations, uncertainty = synthesize_answer(
                query_for_answer,
                passages,
                require_direct_support=self.strict_evidence and not presentation_request,
                min_relevance_score=(
                    0.0 if getattr(self.retriever, "use_reranker", False) else None
                ),
            )

        response.answer = answer
        response.citations = citations
        response.claim_citations = (
            claim_citations if claim_citations is not None
            else _claim_citation_map(answer, passages)
        )
        response.uncertainty = uncertainty
        generation_ms += (time.perf_counter() - generation_started) * 1000
        response.stage_latency_ms = {
            "retrieval": round(retrieval_ms, 3),
            "generation": round(generation_ms, 3),
            "total": round((time.perf_counter() - started_at) * 1000, 3),
        }
        _record_cost(response, initial_usage, self.evidence_client, self.generation_client)
        if session is not None:
            session.commit(response, domain)
        self._notify_answer(on_event, response)
        return response

    @classmethod
    def _notify_answer(cls, on_event: Callable[[dict], None] | None, response: RagResponse) -> None:
        cls._notify(on_event, {
            "type": "answer_ready",
            "answer": response.answer,
            "citations": list(response.citations),
            "claim_citations": dict(response.claim_citations),
            "uncertainty": response.uncertainty,
        })

    @staticmethod
    def _notify(on_event: Callable[[dict], None] | None, event: dict) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:
            # Observability must not change the retrieval/generation outcome.
            return


def _record_cost(response: RagResponse, initial, evidence_client, generation_client) -> None:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    estimated_cost: float | None = 0.0
    pricing_sources = []
    seen: set[int] = set()
    for name, client in (("evidence", evidence_client), ("generation", generation_client)):
        if client is None or id(client) in seen:
            continue
        seen.add(id(client))
        usage = getattr(client, "token_usage", None)
        before = initial.get(name, {})
        if isinstance(usage, dict):
            delta = {}
            for key in totals:
                delta[key] = max(0, int(usage.get(key, 0)) - int(before.get(key, 0)))
                totals[key] += delta[key]
            if delta["total_tokens"]:
                prices = getattr(client, "price_per_million", None)
                if prices is None:
                    estimated_cost = None
                elif estimated_cost is not None:
                    estimated_cost += (
                        delta["prompt_tokens"] * prices[0]
                        + delta["completion_tokens"] * prices[1]
                    ) / 1_000_000
                pricing_sources.append(getattr(client, "pricing_basis", None) or "unpriced provider")
    response.token_usage = totals
    response.estimated_cost_usd = None if estimated_cost is None else round(estimated_cost, 8)
    response.cost_estimate_basis = "; ".join(dict.fromkeys(pricing_sources)) if pricing_sources else "no provider tokens"


def _claim_citation_map(answer: str, passages: list) -> dict[str, list[str]]:
    """Map exact extractive sentences to their source IDs; leave paraphrases unmapped."""
    mapping = {}
    for sentence in SENTENCE_RE.split(answer.strip()):
        sentence = sentence.strip()
        if sentence:
            ids = list(dict.fromkeys(
                item.passage.id for item in passages
                if " ".join(sentence.split()) in " ".join(item.passage.text.split())
            ))
            if ids:
                mapping[sentence] = ids
    return mapping
