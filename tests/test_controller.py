from prism_live_rag.controller import RuleBasedRetrievalController, TranscriptChunk


def test_controller_waits_on_short_partial() -> None:
    controller = RuleBasedRetrievalController()
    assert controller.decide(TranscriptChunk(0.0, "how do", False, 0.8)) == "Wait"


def test_controller_retrieves_on_final() -> None:
    controller = RuleBasedRetrievalController()
    assert controller.decide(TranscriptChunk(1.2, "how do I deploy code engine", True, 0.9)) == "Retrieve"

