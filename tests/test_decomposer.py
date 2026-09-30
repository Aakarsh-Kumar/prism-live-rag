from types import SimpleNamespace

import pytest

from prism_live_rag.controller import TranscriptChunk
from prism_live_rag.decomposer import LLMDecomposer, RuleBasedDecomposer
from prism_live_rag.evaluation import evaluate_decomposition
from prism_live_rag.models import Passage, RetrievedPassage
from prism_live_rag.pipeline import StreamingRagPipeline
from prism_live_rag.stream import SubIntent


def test_consecutive_independent_questions_are_decomposed():
    result = RuleBasedDecomposer().decompose("What are Benefit Charges? Can I protest Benefit Charges as well?")
    assert result.sub_queries == ["What are Benefit Charges", "Can I protest Benefit Charges as well"]


def test_comparison_intent_inherits_explicitly_defined_subject():
    result = RuleBasedDecomposer().decompose("What is HAART therapy and how does it differ from AZT?")
    assert result.sub_queries == ["What is HAART therapy", "how does HAART therapy differ from AZT"]


def test_ambiguous_definition_does_not_choose_one_coordinated_subject():
    result = RuleBasedDecomposer().decompose("What is alpha or beta and how does it work?")
    assert result.sub_queries[-1] == "how does it work"


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=512):
        self.calls.append(messages)
        return self.response


def test_rule_decomposer_splits_explicit_composite_stream_boundary() -> None:
    query = "What is cloud storage? . How do I configure it?"

    result = RuleBasedDecomposer().decompose(query)

    assert result.is_multi_intent
    assert result.sub_queries == [
        "What is cloud storage",
        "How do I configure cloud storage",
    ]


def test_rule_decomposer_splits_independent_question_clause() -> None:
    result = RuleBasedDecomposer().decompose(
        "What is cloud storage and how do I configure it?"
    )

    assert result.is_multi_intent
    assert result.sub_queries == [
        "What is cloud storage",
        "how do I configure cloud storage",
    ]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "What is cloud storage? Also, how do I configure it?",
            ["What is cloud storage", "how do I configure cloud storage"],
        ),
        (
            "What is cloud storage, and could you explain the security options?",
            ["What is cloud storage", "could you explain the security options"],
        ),
        (
            "Explain how storage works and whether I need encryption.",
            ["Explain how storage works and whether I need encryption."],
        ),
    ],
)
def test_rule_decomposer_handles_natural_connectors_without_splitting_dependents(query, expected):
    result = RuleBasedDecomposer().decompose(query)
    assert result.sub_queries == expected
    assert result.is_multi_intent is (len(expected) > 1)


def test_rule_decomposer_preserves_single_intent_noun_lists() -> None:
    result = RuleBasedDecomposer().decompose(
        "What kind of sports program is available for children, adults and seniors?"
    )

    assert not result.is_multi_intent
    assert result.sub_queries == [
        "What kind of sports program is available for children, adults and seniors?"
    ]


def test_llm_decomposer_uses_chat_client_and_accepts_fenced_json() -> None:
    client = FakeChatClient(
        '```json\n{"is_multi_intent": true, "sub_queries": '
        '["Cloud storage definition", "Cloud storage configuration"]}\n```'
    )

    result = LLMDecomposer(client).decompose("What is cloud storage and how do I configure it?")

    assert result.is_multi_intent
    assert result.sub_queries == [
        "Cloud storage definition",
        "Cloud storage configuration",
    ]
    assert len(client.calls) == 1
    assert result.provider_status == "success"


def test_llm_decomposer_falls_back_to_one_query_on_malformed_json() -> None:
    client = FakeChatClient("not JSON")
    query = "What is cloud storage and how do I configure it?"
    decomposer = LLMDecomposer(client)

    result = decomposer.decompose(query)

    assert not result.is_multi_intent
    assert result.sub_queries == [query]
    assert decomposer.get_health_status()["llm_failures"] == 1
    assert result.provider_status == "fallback"


def test_llm_decomposer_rejects_multi_intent_flag_with_one_sub_query() -> None:
    client = FakeChatClient(
        '{"is_multi_intent": true, "sub_queries": ["only one search intent"]}'
    )
    query = "What is cloud storage and how do I configure it?"

    result = LLMDecomposer(client).decompose(query)

    assert not result.is_multi_intent
    assert result.sub_queries == [query]


def test_pipeline_routes_decomposed_final_stream_to_multi_query_retrieval() -> None:
    class Retriever:
        def __init__(self):
            self.multi_queries = []

        def search(self, query, domain, **kwargs):
            return []

        def search_multi_query(self, queries, domain, **kwargs):
            self.multi_queries.append((queries, domain))
            return []

    retriever = Retriever()
    query = "What is cloud storage? . How do I configure it?"
    response = StreamingRagPipeline(
        retriever, enable_multi_intent=True
    ).run(
        [
            TranscriptChunk(0.5, "What is cloud storage?", False, 0.8),
            TranscriptChunk(1.0, "What is cloud storage?", False, 0.9),
            TranscriptChunk(1.5, query, True, 0.98),
        ],
        "cloud",
    )

    assert retriever.multi_queries[-1] == (
        ["What is cloud storage", "How do I configure cloud storage"],
        "cloud",
    )
    assert response.retrieval_events[-1].trigger == "multi_intent"


def test_pipeline_synthesizes_and_cites_each_decomposed_intent() -> None:
    first = RetrievedPassage(
        Passage("storage-overview", "cloud", "Cloud storage stores data on remote servers."),
        1.0,
    )
    second = RetrievedPassage(
        Passage("storage-setup", "cloud", "To configure cloud storage, create a bucket and choose a region."),
        0.9,
    )

    class Retriever:
        def search(self, query, domain, **kwargs):
            return []

        def search_multi_query(self, queries, domain, **kwargs):
            return [first, second]

    response = StreamingRagPipeline(
        Retriever(), enable_multi_intent=True
    ).run(
        [TranscriptChunk(1.0, "What is cloud storage and how do I configure it?", True, 0.99)],
        "cloud",
    )

    assert "What is cloud storage:" in response.answer
    assert "how do I configure cloud storage:" in response.answer
    assert set(response.citations) == {"storage-overview", "storage-setup"}
    assert response.uncertainty is None


def test_pipeline_calls_llm_at_each_distinct_streamed_retrieval_checkpoint() -> None:
    client = FakeChatClient(
        '{"is_multi_intent": false, "sub_queries": ["How does cloud storage work?"]}'
    )
    retriever = SimpleNamespace(search=lambda query, domain, **kwargs: [])
    pipeline = StreamingRagPipeline(
        retriever,
        decomposer=LLMDecomposer(client),
        enable_multi_intent=True,
        refine_on_final=True,
        synthesis_mode="provider",
    )

    pipeline.run(
        [
            TranscriptChunk(0.2, "What is cloud storage and how do I configure it?", False, 0.8),
            TranscriptChunk(0.8, "What is cloud storage and how do I configure it?", False, 0.9),
            TranscriptChunk(1.4, "What is cloud storage and how do I configure it in California?", True, 0.99),
        ],
        "cloud",
    )

    assert len(client.calls) == 2  # stable provisional checkpoint plus revised final


def test_pipeline_skips_llm_decomposer_for_single_intent_even_when_enabled():
    client = FakeChatClient("unused")
    events = []
    StreamingRagPipeline(
        SimpleNamespace(search=lambda query, domain: []),
        decomposer=LLMDecomposer(client),
        synthesis_mode="provider",
        enable_multi_intent=True,
    ).run(
        [TranscriptChunk(1.0, "What is cloud storage?", True, 0.98)],
        on_event=events.append,
    )
    assert client.calls == []
    assert next(event for event in events if event.get("stage") == "decomposition")["decision"] == "skip"


def test_decomposition_eval_reports_compound_success_and_overfragmentation() -> None:
    compound = SimpleNamespace(
        stream_id="compound-1",
        domain="cloud",
        category="multi_intent",
        chunks=(TranscriptChunk(1.0, "What is cloud storage? . How do I configure it?", True),),
        sub_intents=(SubIntent("What is cloud storage?", 1), SubIntent("How do I configure it?", 2)),
    )
    single = {
        "case_id": "single-1",
        "domain": "cloud",
        "query": "How can I list storage and network resources?",
    }

    result = evaluate_decomposition(
        [compound], RuleBasedDecomposer(), single_queries=[single]
    )

    assert result["compound_success_rate"] == 1.0
    assert result["compound_intent_precision"] == 1.0
    assert result["compound_intent_recall"] == 1.0
    assert result["single_overfragmentation_rate"] == 0.0


def test_decomposition_eval_scores_final_asr_revision_and_records_retrieval_chunks() -> None:
    final_query = "What is cloud storage? Also, how do I configure it?"
    case = {
        "case_id": "streamed-compound",
        "domain": "cloud",
        "category": "compound",
        "review_status": "pending_g3_review",
        "expected_intents": ["What is cloud storage?", "how do I configure it?"],
        "chunks": [
            {"timestamp_s": 0.4, "text": "What is cloud storage? Also, how do I", "is_final": False, "confidence": 0.8},
            {"timestamp_s": 0.9, "text": "What is cloud storage? Also, how can I set it up?", "is_final": False, "confidence": 0.86},
            {"timestamp_s": 1.4, "text": final_query, "is_final": True, "confidence": 0.98},
        ],
    }

    result = evaluate_decomposition([case], RuleBasedDecomposer())

    row = result["cases"][0]
    assert result["compound_success_rate"] == 1.0
    assert result["review_status_counts"] == {"pending_g3_review": 1}
    assert len(result["failures"]) == 0
    assert result["single_queries"] == 0
    assert row["final_query"] == final_query
    assert row["full_compound_match"]
    assert len(row["retrieval_observations"]) >= 1
    assert row["retrieval_observations"][-1]["is_final"]


def test_decomposition_eval_separates_provider_fallback_from_model_quality():
    class OutcomeDecomposer:
        def get_health_status(self):
            return {"calls": 1}

        def decompose(self, query, context=""):
            return SimpleNamespace(
                sub_queries=[query], is_multi_intent=False,
                decomposition_reason="provider unavailable; fallback",
                processing_time_ms=1.0, provider_status="fallback",
                provider_error="ProviderError",
            )

    case = {
        "case_id": "fallback-compound", "category": "compound",
        "expected_intents": ["What is cloud storage?", "How do I configure cloud storage?"],
        "chunks": [
            {"timestamp_s": 0.2, "text": "What is cloud storage and how do I configure it?", "is_final": True}
        ],
    }
    report = evaluate_decomposition([case], OutcomeDecomposer())

    assert report["compound_success_rate"] == 0.0
    assert report["provider_outcomes"]["final_provider_fallback_cases"] == 1
    assert report["provider_outcomes"]["model_compound_accuracy_with_failures_counted_incorrect"] == 0.0
    assert report["cases"][0]["final_provider_error"] == "ProviderError"


def test_decomposition_eval_resume_reuses_completed_case_without_provider_call():
    class CountingDecomposer:
        def __init__(self):
            self.calls = 0

        def get_health_status(self):
            return {"calls": self.calls}

        def decompose(self, query, context=""):
            self.calls += 1
            return SimpleNamespace(
                sub_queries=["What is cloud storage", "How do I configure it"],
                is_multi_intent=True, decomposition_reason="ok",
                processing_time_ms=1.0, provider_status="success", provider_error=None,
            )

    case = {
        "case_id": "resume-case", "category": "compound",
        "expected_intents": ["What is cloud storage", "How do I configure it"],
        "chunks": [{
            "timestamp_s": 1.0,
            "text": "What is cloud storage and how do I configure it?",
            "is_final": True,
        }],
    }
    decomposer = CountingDecomposer()
    first = evaluate_decomposition([case], decomposer)
    resumed = evaluate_decomposition(
        [case], decomposer, resume_rows={"resume-case": first["cases"][0]}
    )

    assert decomposer.calls == 1
    assert resumed["resumed_cases"] == 1
    assert resumed["compound_success_rate"] == 1.0


def test_decomposition_eval_retries_only_fallback_checkpoints_on_resume():
    class SequenceDecomposer:
        def __init__(self, outcomes):
            self.outcomes = iter(outcomes)
            self.calls = 0

        def get_health_status(self):
            return {"type": "llm_based", "total_queries": self.calls}

        def decompose(self, query, context=""):
            self.calls += 1
            status = next(self.outcomes)
            return SimpleNamespace(
                sub_queries=[query], is_multi_intent=False,
                decomposition_reason="Single intent",
                processing_time_ms=1.0, provider_status=status,
                provider_error="LLM failed (ProviderError_HTTP_429)" if status == "fallback" else None,
            )

    case = {
        "case_id": "retry-one-checkpoint", "category": "single_intent",
        "expected_intents": ["How does cloud storage work in California?"],
        "chunks": [
            {"timestamp_s": 0.2, "text": "How does cloud storage work?", "is_final": False},
            {"timestamp_s": 0.8, "text": "How does cloud storage work?", "is_final": False},
            {"timestamp_s": 1.4, "text": "How does cloud storage work in California?", "is_final": True},
        ],
    }
    initial = SequenceDecomposer(["success", "fallback"])
    first = evaluate_decomposition([case], initial)
    retry = SequenceDecomposer(["success"])
    resumed = evaluate_decomposition(
        [case], retry,
        resume_rows={"retry-one-checkpoint": first["cases"][0]},
        retry_fallbacks=True,
    )

    assert initial.calls == 2
    assert retry.calls == 1
    assert resumed["cases"][0]["final_provider_status"] == "success"
    assert resumed["provider_outcomes"]["final_provider_success_cases"] == 1


def test_llm_decomposer_can_wait_for_circuit_breaker_recovery(monkeypatch):
    client = FakeChatClient('{"is_multi_intent": false, "sub_queries": ["one query"]}')
    sleeps = []
    monkeypatch.setattr("prism_live_rag.decomposer.time.sleep", lambda delay: sleeps.append(delay))
    decomposer = LLMDecomposer(
        client, circuit_breaker_cooldown_s=0.25, wait_for_circuit_recovery=True
    )
    decomposer.circuit_open = True
    decomposer.consecutive_failures = 3
    decomposer.last_failure_time = __import__("time").time()

    result = decomposer.decompose("one query")

    assert result.provider_status == "success"
    assert len(client.calls) == 1
    assert sleeps and 0 < sleeps[0] <= 0.25


def test_decomposition_matching_accepts_searchable_paraphrase_without_wording_match():
    from prism_live_rag.evaluation import _match_intents

    expected = ["I keep seeing bare metal servers for VPC mentioned. What is that, exactly?"]
    predicted = ["What is a bare metal server for VPC"]
    assert _match_intents(expected, predicted, 0.5) == [(0, 0)]
    assert _match_intents(["Explain bare metal server for VPC"], ["County birth record renewal"], 0.5) == []
def test_shared_date_is_retained_for_both_upgrade_intents():
    query = "What version was Git Repos upgraded to, and what version was Delivery Pipeline upgraded to on 31 October 2021?"
    result = RuleBasedDecomposer().decompose(query)
    assert result.is_multi_intent
    assert all("on 31 October 2021" in part for part in result.sub_queries)


def test_trailing_date_is_not_copied_to_unrelated_event():
    result = RuleBasedDecomposer().decompose("Who founded IBM and what profits were reported in 2025?")
    assert "2025" not in result.sub_queries[0]


def test_coordinated_interrogatives_inherit_shared_predicate():
    result = RuleBasedDecomposer().decompose("How and why do I configure index rate alerts?")
    assert result.sub_queries == [
        "How do I configure index rate alerts",
        "why do I configure index rate alerts",
    ]
    assert result.is_multi_intent


def test_noun_coordination_is_not_a_shared_interrogative():
    result = RuleBasedDecomposer().decompose("How do I configure alerts and dashboards?")
    assert result.sub_queries == ["How do I configure alerts and dashboards?"]
