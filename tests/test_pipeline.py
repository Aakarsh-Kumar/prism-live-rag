from pathlib import Path
from types import SimpleNamespace

import pytest

from prism_live_rag.config import Settings
from prism_live_rag.controller import TranscriptChunk, simulate_chunks
from prism_live_rag.data import corpus_path
from prism_live_rag.embeddings import build_encoder
from prism_live_rag.models import ConversationSession, Passage, RagResponse, RetrievedPassage
from prism_live_rag.pipeline import StreamingRagPipeline, _record_cost
from prism_live_rag.retrieval import HybridRetriever
from prism_live_rag.synthesis import extract_evidence, focus_refinement_query, needs_clarification


def corpus_available(domain: str = "cloud") -> bool:
    return corpus_path(Path("data"), domain).exists()


@pytest.mark.skipif(not corpus_available(), reason="cloud corpus is gitignored and absent")
def test_pipeline_returns_required_shape() -> None:
    settings = Settings(data_dir=Path("data"), lancedb_dir=Path(".cache/test-lancedb"))
    retriever = HybridRetriever(
        settings.data_dir,
        settings.lancedb_dir,
        settings.table_name,
        build_encoder("hash"),
        settings.rrf_k,
        settings.sparse_weight,
        corpus_limit=300,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
    )
    pipeline = StreamingRagPipeline(retriever)
    response = pipeline.run(simulate_chunks("I heard the toolchain is not available in South America."), "cloud")
    payload = response.to_dict()
    assert set(payload) == {"retrieval_events", "sub_queries", "decisions", "answer", "citations", "uncertainty"}
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
    assert response.sub_queries == ["I heard the toolchain is not available in South America."]
    assert response.citations == ["cloud-final"]


def test_pipeline_observer_reports_controller_retrieval_evidence_and_answer() -> None:
    events = []
    pipeline = StreamingRagPipeline(FakeRetriever())
    pipeline.run(
        [TranscriptChunk(1.0, "What is the toolchain service?", True, 0.98)],
        "cloud",
        on_event=events.append,
    )

    types = [event["type"] for event in events]
    assert types == [
        "chunk_processed", "decision", "retrieval_start", "search_dispatch", "retrieval_done", "answer_ready"
    ]
    retrieval = next(event for event in events if event["type"] == "retrieval_done")
    assert retrieval["passages"][0]["id"] == "cloud-provisional"
    assert retrieval["passages"][0]["text"] == "The toolchain service is available."
    assert retrieval["passages"][0]["score"] == 1.0
    assert events[-1]["citations"] == ["cloud-provisional"]


def test_pipeline_observer_reports_early_answer_paths() -> None:
    events = []
    session = ConversationSession(
        answer="A prior answer.", citations=["cloud-provisional"], domain="cloud", version=1
    )
    response = StreamingRagPipeline(FakeRetriever()).run(
        [TranscriptChunk(1.0, "repeat that in two bullets", True, 0.98)],
        "cloud",
        session=session,
        on_event=events.append,
    )
    assert events[-1]["type"] == "answer_ready"
    assert events[-1]["answer"] == response.answer


def test_pipeline_discards_stale_provisional_passages_when_final_refresh_is_empty() -> None:
    retriever = FakeRetriever(final_has_passages=False)
    pipeline = StreamingRagPipeline(retriever, refine_on_final=True)
    response = pipeline.run(simulate_chunks("I heard the toolchain is not available in South America."), "cloud")
    assert retriever.queries == [
        "I heard the toolchain is not available in",
        "I heard the toolchain is not available in South America.",
    ]
    assert response.citations == []
    assert response.uncertainty is not None


def test_pipeline_deduplicates_identical_retrieval_chunks() -> None:
    retriever = FakeRetriever()
    chunks = [
        TranscriptChunk(0.8, "I heard the toolchain is not available in", False, 0.8),
        TranscriptChunk(1.6, "I heard the toolchain is not available in", True, 0.95),
    ]
    pipeline = StreamingRagPipeline(retriever, refine_on_final=True)
    pipeline.run(chunks, "cloud")
    assert retriever.queries == ["I heard the toolchain is not available in"]


def test_pipeline_consumes_every_chunk_of_the_stream() -> None:
    """The pipeline must see the whole stream, not stop at the first retrieval.

    An early break meant the final chunk was never observed, which made late-constraint
    G5 refinement structurally impossible.
    """
    retriever = FakeRetriever()
    chunks = [
        TranscriptChunk(0.4, "I", False, 0.51),
        TranscriptChunk(0.9, "I heard", False, 0.57),
        TranscriptChunk(1.4, "I heard the toolchain", False, 0.70),
        TranscriptChunk(2.0, "I heard the toolchain is not available", False, 0.80),
        TranscriptChunk(2.6, "I heard the toolchain is not available in South America.", True, 0.96),
    ]
    response = StreamingRagPipeline(retriever).run(chunks, "cloud")
    assert len(response.decisions) == len(chunks)
    assert [d["decision"] for d in response.decisions] == [
        "Wait", "Wait", "Wait", "Retrieve", "Retrieve",
    ]
    assert [event.trigger for event in response.retrieval_events] == ["provisional", "final"]


def test_pipeline_accepts_a_generator_of_chunks() -> None:
    """Text must be able to arrive lazily, not be materialised as a list first."""
    retriever = FakeRetriever()
    chunks = (
        TranscriptChunk(0.8, "I heard the toolchain is not available in", False, 0.8),
        TranscriptChunk(1.6, "I heard the toolchain is not available in South America.", True, 0.95),
    )
    response = StreamingRagPipeline(retriever).run(iter(list(chunks)), "cloud")
    assert len(response.decisions) == 2
    assert response.answer


def test_pipeline_synthesises_from_the_final_retrieval() -> None:
    """With full-stream consumption the answer must come from the refined final query."""
    retriever = FakeRetriever()
    chunks = [
        TranscriptChunk(0.8, "I heard the toolchain is not available in", False, 0.8),
        TranscriptChunk(1.6, "I heard the toolchain is not available in South America.", True, 0.95),
    ]
    response = StreamingRagPipeline(retriever).run(chunks, "cloud")
    assert retriever.queries[-1].endswith("South America.")
    assert response.citations == ["cloud-final"]


def test_pipeline_suppresses_retrieval_for_presentation_commands() -> None:
    retriever = FakeRetriever()
    chunks = [TranscriptChunk(0.8, "summarize what you just said", True, 0.95)]
    response = StreamingRagPipeline(retriever).run(chunks, "cloud")
    assert retriever.queries == []
    assert response.retrieval_events == []
    assert response.decisions[0]["decision"] == "No-Retrieval"


def test_multi_intent_preserves_evidence_displaced_by_global_fusion():
    contact = RetrievedPassage(Passage(id="contact", domain="govt", text="Republic Services handles missed garbage collection."), score=1.0)
    cart = RetrievedPassage(Passage(id="cart", domain="govt", text="Report someone taking your cart to the recycling company."), score=1.0)

    class Retriever(FakeRetriever):
        def search_multi_query(self, queries, domain):
            self.last_intent_results = [[contact], [cart]]
            return [contact]

    events = []
    response = StreamingRagPipeline(Retriever(), enable_multi_intent=True).run(
        [TranscriptChunk(1.0, "Who handles missed garbage collection and what should I do if someone takes my cart?", True, 0.99)],
        "govt", on_event=events.append,
    )
    assert set(response.citations) == {"contact", "cart"}
    assert response.uncertainty is None
    assert response.claim_citations[cart.passage.text] == ["cart"]
    assert {p["id"] for e in events if e["type"] == "retrieval_done" for p in e["passages"]} == {"contact", "cart"}


def test_pipeline_uses_history_for_context_dependent_retrieval_query() -> None:
    retriever = FakeRetriever()
    chunks = [TranscriptChunk(1.0, "What about South America?", True, 0.98)]
    context = (
        ("user", "Is the toolchain service available?"),
        ("agent", "It is available in several regions."),
    )
    StreamingRagPipeline(retriever).run(chunks, "cloud", context_turns=context)
    assert retriever.queries == [
        "Conversation context: user: Is the toolchain service available? "
        "agent: It is available in several regions. Current question: What about South America?"
    ]


def test_multi_intent_decomposer_receives_only_latest_utterance_and_context_separately() -> None:
    class Decomposer:
        def __init__(self):
            self.calls = []

        def decompose(self, query, context=""):
            self.calls.append((query, context))
            from prism_live_rag.decomposer import MultiIntentResult
            return MultiIntentResult(
                sub_queries=["Who handles missed garbage collection", "What if someone takes my cart"],
                decomposition_reason="two questions",
                is_multi_intent=True,
                processing_time_ms=0,
                original_query=query,
            )

    class MultiRetriever(FakeRetriever):
        def __init__(self):
            super().__init__()
            self.multi_queries = []

        def search_multi_query(self, queries, domain):
            self.multi_queries.append((queries, domain))
            return []

    decomposer = Decomposer()
    retriever = MultiRetriever()
    context = (("user", "I recently moved to town."), ("agent", "Welcome."))
    StreamingRagPipeline(
        retriever, decomposer=decomposer, enable_multi_intent=True
    ).run(
        [TranscriptChunk(
            1.0,
            "Who do I call about missed collection of garbage service? And what should I do if someone takes my cart?",
            True,
            0.98,
        )],
        "govt",
        context_turns=context,
    )

    assert decomposer.calls == [(
        "Who do I call about missed collection of garbage service? And what should I do if someone takes my cart?",
        "user: I recently moved to town. agent: Welcome.",
    )]
    assert retriever.multi_queries[-1][0] == [
        "Who handles missed garbage collection",
        "What if someone takes my cart",
    ]


def test_pipeline_clarifies_unresolved_reference_without_citations() -> None:
    retriever = FakeRetriever()
    chunks = [TranscriptChunk(1.0, "What is the purpose of that mission?", True, 0.98)]
    response = StreamingRagPipeline(retriever).run(chunks, "cloud")
    assert response.answer.endswith("?")
    assert response.citations == []
    assert response.uncertainty is not None


@pytest.mark.parametrize("query,empty", [
    ("Is the toolchain service available?", False),
    ("Is the toolchain service available?", True),
    ("How can I improve its performance?", False),
])
def test_provider_routing_skips_supported_lookup_empty_evidence_and_clarification(query, empty):
    class Client:
        def chat(self, *args, **kwargs):
            pytest.fail("This query should not invoke a provider")

    retriever = FakeRetriever()
    if empty:
        retriever.search = lambda query, domain: []
    events = []
    client = Client()
    result = StreamingRagPipeline(
        retriever, synthesis_mode="provider", evidence_client=client, generation_client=client
    ).run([TranscriptChunk(1.0, query, True, 0.98)], on_event=events.append)
    route = [event for event in events if event["type"] == "llm_routing"][-1]
    assert route["decision"] == "skip"
    assert result.citations == ([] if empty or "its" in query else ["cloud-provisional"])


def test_provider_routing_calls_for_causal_question_and_returns_verified_evidence():
    sentence = "The toolchain service is available because regional infrastructure supports it."

    class Client:
        def __init__(self):
            self.calls = 0

        def chat(self, *args, **kwargs):
            import json
            self.calls += 1
            return json.dumps({"extracted_spans": [{"passage_id": 1, "sentence": sentence}]})

    retriever = SimpleNamespace(search=lambda query, domain: [
        RetrievedPassage(Passage("causal-evidence", domain, sentence), score=1.0)
    ])
    client = Client()
    events = []
    result = StreamingRagPipeline(
        retriever, synthesis_mode="provider", evidence_client=client, generation_client=client
    ).run(
        [TranscriptChunk(1.0, "Why is the toolchain service available?", True, 0.98)],
        on_event=events.append,
    )
    assert client.calls == 1
    assert result.answer == sentence
    assert result.citations == ["causal-evidence"]
    assert next(event for event in events if event["type"] == "llm_routing")["decision"] == "call"


@pytest.mark.parametrize(
    "query",
    [
        "Does this Superior Court publish a judges roster?",
        "What services does the clinic provide?",
        "What is the purpose of that NASA mission?",
    ],
)
def test_clarification_detector_catches_missing_generic_referents(query) -> None:
    assert needs_clarification(query)


def test_clarification_detector_requires_referent_terms_in_context() -> None:
    query = "What is the purpose of that NASA mission?"
    assert needs_clarification(query, (("user", "Tell me about the mission."),))
    assert not needs_clarification(query, (("user", "Tell me about the NASA mission."),))


@pytest.mark.parametrize(
    "query",
    [
        "How can we enhance its performance?",
        "What are the best practices for SDK?",
    ],
)
def test_clarification_detector_catches_unresolved_pronoun_or_bare_sdk(query):
    assert needs_clarification(query)


class RefinementRetriever:
    def __init__(self, has_delta_evidence: bool = True) -> None:
        self.queries = []
        self.has_delta_evidence = has_delta_evidence

    def search(self, query: str, domain: str):
        self.queries.append(query)
        if "maybe" in query:
            return [RetrievedPassage(Passage(
                id="stale-partial", domain=domain,
                text="Maybe California permits the service for enterprise accounts.",
            ), score=0.9)]
        if self.has_delta_evidence and "California" in query:
            return [RetrievedPassage(Passage(
                id="california-rule", domain=domain,
                text="California permits the service for enterprise accounts.",
            ), score=0.9)]
        return []


def test_session_refinement_searches_only_delta_and_preserves_prior_answer_and_citations() -> None:
    retriever = RefinementRetriever()
    pipeline = StreamingRagPipeline(retriever)
    session = ConversationSession(
        answer="The service is available in several regions.",
        citations=["regional-summary"], version=2, domain="cloud",
    )
    response = pipeline.run(
        [TranscriptChunk(1.0, "for California only", True, 0.99)],
        "cloud", session=session, refinement=True,
    )
    assert retriever.queries == ["for California only"]
    assert response.retrieval_events[0].trigger == "refinement"
    assert "several regions" not in response.answer
    assert "California permits" in response.answer
    assert response.citations == ["california-rule"]
    assert response.previous_version == 2
    assert response.version == session.version == 3
    assert response.applied_delta == "for California only"


def test_refinement_preserves_unrelated_claim_and_its_citation() -> None:
    retriever = RefinementRetriever()
    session = ConversationSession(
        answer="The service is available in several regions. Billing help is available every day.",
        citations=["regional-summary", "billing-guide"], version=1, domain="cloud",
    )
    response = StreamingRagPipeline(retriever).run(
        [TranscriptChunk(1.0, "for California only", True, 0.99)],
        "cloud", session=session, refinement=True,
    )
    assert "several regions" not in response.answer
    assert "Billing help is available every day." in response.answer
    assert "California permits" in response.answer
    assert response.citations == ["regional-summary", "billing-guide", "california-rule"]


def test_presentation_followup_reuses_session_without_search() -> None:
    retriever = RefinementRetriever()
    session = ConversationSession(
        answer="The first rule applies. The second rule also applies.",
        citations=["rule-a", "rule-b"], version=2, domain="cloud",
    )
    response = StreamingRagPipeline(retriever).run(
        [TranscriptChunk(1.0, "repeat that in two bullets", True, 0.99)],
        "cloud", session=session,
    )
    assert retriever.queries == []
    assert response.answer == "- The first rule applies.\n- The second rule also applies."
    assert response.citations == ["rule-a", "rule-b"]
    assert response.previous_version == 2 and session.version == response.version == 3


def test_session_refinement_without_delta_evidence_keeps_state_unchanged() -> None:
    retriever = RefinementRetriever(has_delta_evidence=False)
    pipeline = StreamingRagPipeline(retriever)
    session = ConversationSession(
        answer="Existing grounded answer.", citations=["existing-id"], version=4, domain="cloud"
    )
    response = pipeline.run(
        [TranscriptChunk(0.8, "for California only", False, 0.8),
         TranscriptChunk(1.4, "for California only please", True, 0.99)],
        "cloud", session=session, refinement=True,
    )
    assert retriever.queries == ["for California only please"]
    assert response.answer == "Existing grounded answer."
    assert response.citations == ["existing-id"]
    assert response.uncertainty and "prior answer is preserved" in response.uncertainty
    assert response.previous_version == 4 and response.version == 5


def test_final_asr_revision_supersedes_partial_delta_evidence() -> None:
    retriever = RefinementRetriever(has_delta_evidence=False)
    session = ConversationSession(
        answer="Existing grounded answer.", citations=["existing-id"], version=1, domain="cloud"
    )
    response = StreamingRagPipeline(retriever).run(
        [TranscriptChunk(0.4, "maybe California service available", False, 0.8),
         TranscriptChunk(0.8, "maybe California service available", False, 0.82),
         TranscriptChunk(1.5, "for California only", True, 0.99)],
        "cloud", session=session, refinement=True,
    )
    assert retriever.queries == ["maybe California service available", "for California only"]
    assert response.answer == "Existing grounded answer."
    assert response.citations == ["existing-id"]
    assert "stale-partial" not in response.citations


def test_refinement_requires_committed_session() -> None:
    with pytest.raises(ValueError, match="committed answer"):
        StreamingRagPipeline(FakeRetriever()).run(
            [TranscriptChunk(1.0, "for California", True, 0.99)],
            "cloud", session=ConversationSession(), refinement=True,
        )


def test_opt_in_trace_records_stage_timings_and_trace_id() -> None:
    response = StreamingRagPipeline(RefinementRetriever()).run(
        [TranscriptChunk(1.0, "for California only", True, 0.99)],
        "cloud", trace=True,
    )
    payload = response.to_dict()
    assert payload["trace_id"]
    assert set(payload["stage_latency_ms"]) == {"retrieval", "generation", "total"}
    assert payload["stage_latency_ms"]["total"] >= payload["stage_latency_ms"]["retrieval"]
    assert payload["estimated_cost_usd"] == 0.0


def test_shared_cerebras_client_usage_and_cost_counted_once() -> None:
    client = SimpleNamespace(
        token_usage={"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
        price_per_million=(0.35, 0.75), pricing_basis="Cerebras published price",
    )
    before = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    response = RagResponse()
    _record_cost(response, {"evidence": before, "generation": before}, client, client)
    assert response.token_usage == {
        "prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
    }
    assert response.estimated_cost_usd == 0.0000575
    assert response.cost_estimate_basis == "Cerebras published price"


def test_refinement_evidence_focus_skips_old_topic_and_question_headings() -> None:
    focused, anchors = focus_refinement_query(
        "for transit gateway DDoS protection only",
        "IBM Cloud Transit Gateway does not perform encryption; it only provides connectivity.",
    )
    assert focused == "DDoS protection"
    assert anchors == {"ddos"}
    passages = [RetrievedPassage(Passage(
        id="ddos-policy", domain="cloud",
        text=("How do you prevent DDoS attacks? "
              "DDoS attacks cannot bring down the network."),
    ), score=0.9)]
    assert extract_evidence(focused, passages, required_terms=anchors) == [
        ("ddos-policy", "DDoS attacks cannot bring down the network.")
    ]


def test_initial_session_response_starts_answer_lineage_at_version_one() -> None:
    session = ConversationSession()
    response = StreamingRagPipeline(RefinementRetriever()).run(
        [TranscriptChunk(1.0, "The service supports California accounts", True, 0.99)],
        "cloud", session=session,
    )
    payload = response.to_dict()
    assert payload["version"] == session.version == 1
    assert payload["previous_version"] is None
