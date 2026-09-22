from prism_live_rag.evaluation import evaluate_provider, evaluate_retrieval
from prism_live_rag.models import QueryTask, RagResponse, RetrievalEvent


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
