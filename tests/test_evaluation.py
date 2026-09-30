from prism_live_rag.controller import TranscriptChunk
from prism_live_rag.evaluation import (
    evaluate_curated_streaming,
    evaluate_provider,
    evaluate_retrieval,
)
from prism_live_rag.models import QueryTask, RagResponse, RetrievalEvent
from prism_live_rag.stream import SimulatedStream


def test_evaluate_retrieval_reports_recall_and_misses() -> None:
    tasks = [
        QueryTask("t1", "cloud", "toolchain availability in south america region for code engine deployment", qrel_passage_ids=("p2",)),
        QueryTask("t2", "cloud", "missing answer in a sufficiently long query about cloud configuration", qrel_passage_ids=("p9",)),
    ]

    def search(query: str) -> list[str]:
        if "toolchain" in query:
            return ["p1", "p2"]
        return ["p3"]

    result = evaluate_retrieval(tasks, search, limit=2)
    assert result["tasks"] == 2
    assert result["success_at_k"] == 0.5
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] == 0.5
    assert result["early_retrieval_rate"] == 1.0
    assert result["misses"][0]["task_id"] == "t2"


def test_evaluate_retrieval_computes_true_recall() -> None:
    tasks = [
        QueryTask("t1", "cloud", "multi-gold query", qrel_passage_ids=("p1", "p2", "p3")),
        QueryTask("t2", "cloud", "single-gold query", qrel_passage_ids=("p9",)),
    ]

    def search(query: str) -> list[str]:
        if "multi-gold" in query:
            return ["p1", "p2"]
        return ["p3"]

    result = evaluate_retrieval(tasks, search, limit=2)
    assert result["recall_at_k"] == round((2 / 3 + 0 / 1) / 2, 4)
    assert result["success_at_k"] == 0.5


def test_evaluate_provider_reports_citation_validity_and_abstention() -> None:
    tasks = [
        QueryTask("t1", "cloud", "answerable", qrel_passage_ids=("p2",)),
        QueryTask("t2", "cloud", "abstain", qrel_passage_ids=("p9",)),
        QueryTask("t3", "cloud", "empty-without-uncertainty", qrel_passage_ids=("p7",)),
    ]

    def run_pipeline(query: str) -> RagResponse:
        if query == "answerable":
            return RagResponse(
                retrieval_events=[RetrievalEvent(0.1, query, "provisional")],
                citations=["p2"],
                answer="answer",
            )
        if query == "abstain":
            return RagResponse(
                retrieval_events=[RetrievalEvent(0.1, query, "final")],
                citations=[],
                answer="I do not have enough information.",
                uncertainty="No evidence.",
            )
        return RagResponse(
            retrieval_events=[RetrievalEvent(0.1, query, "final")],
            citations=[],
            answer="",
        )

    result = evaluate_provider(tasks, run_pipeline, corpus_ids={"p2", "p7", "p9"}, limit=3)
    assert result["tasks"] == 3
    assert result["citation_valid_rate"] == 1.0
    assert result["qrel_citation_hit_rate"] == round(1 / 3, 4)
    assert result["abstention_rate"] == round(1 / 3, 4)
    assert result["early_retrieval_rate"] == round(1 / 3, 4)
    assert result["failures"][0]["task_id"] == "t2"


def test_evaluate_curated_streaming_uses_partial_timing_and_behavior_proxies() -> None:
    def stream(case_class: str, expected: str, task_id: str) -> SimulatedStream:
        return SimulatedStream(
            stream_id=f"{task_id}-stream",
            task_id=task_id,
            category="early_retrieval",
            domain="cloud",
            qrel_passage_ids=("gold",) if case_class in {"answerable", "partial"} else (),
            stability_chunk_index=1,
            settling_ms=500,
            sub_intents=(),
            chunks=(
                TranscriptChunk(0.5, "how", False, 0.6),
                TranscriptChunk(1.0, "how does this", False, 0.8),
                TranscriptChunk(1.5, "how does this work", True, 0.98),
            ),
            case_class=case_class,
            expected_behavior=expected,
            asr_supersedes=(0, 1),
        )

    streams = [
        stream("answerable", "answer", "answer"),
        stream("unanswerable", "abstain", "abstain"),
        stream("underspecified", "clarify", "clarify"),
    ]

    def run_pipeline(item: SimulatedStream) -> RagResponse:
        if item.task_id == "answer":
            return RagResponse(
                retrieval_events=[RetrievalEvent(1.0, "how does this", "provisional")],
                answer="Supported.",
                citations=["gold"],
            )
        if item.task_id == "abstain":
            return RagResponse(answer="Insufficient evidence.", uncertainty="No evidence.")
        return RagResponse(answer="Which service do you mean?")

    result = evaluate_curated_streaming(streams, run_pipeline)
    assert result["tasks"] == 3
    assert result["early_retrieval_rate"] == 1.0
    assert result["before_stability_rate"] == 1.0
    assert result["qrel_citation_hit_rate"] == 1.0
    assert result["behavior_match_proxy_rate"] == 1.0


def test_curated_streaming_does_not_count_final_retrieval_as_early() -> None:
    item = SimulatedStream(
        stream_id="late-stream",
        task_id="late",
        category="early_retrieval",
        domain="cloud",
        qrel_passage_ids=("gold",),
        stability_chunk_index=0,
        settling_ms=0,
        sub_intents=(),
        chunks=(
            TranscriptChunk(0.5, "partial query", False, 0.7),
            TranscriptChunk(1.0, "partial query complete", True, 0.98),
        ),
        case_class="answerable",
        expected_behavior="answer",
        asr_supersedes=(0,),
    )
    result = evaluate_curated_streaming(
        [item],
        lambda _: RagResponse(
            retrieval_events=[RetrievalEvent(1.0, "partial query complete", "final")],
            citations=["gold"],
        ),
    )
    assert result["early_retrieval_rate"] == 0.0
    assert result["before_stability_rate"] == 0.0
