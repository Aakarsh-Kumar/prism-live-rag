from prism_live_rag.controller import RuleBasedRetrievalController, TranscriptChunk


def test_controller_waits_on_short_partial() -> None:
    controller = RuleBasedRetrievalController()
    assert controller.decide(TranscriptChunk(0.0, "how do", False, 0.8)) == "Wait"


def test_controller_retrieves_on_final() -> None:
    controller = RuleBasedRetrievalController()
    assert controller.decide(TranscriptChunk(1.2, "how do I deploy code engine", True, 0.9)) == "Retrieve"


def test_controller_retrieves_on_stable_partial() -> None:
    controller = RuleBasedRetrievalController()
    controller.decide(TranscriptChunk(0.5, "how do I deploy", False, 0.8))
    assert controller.decide(TranscriptChunk(1.0, "how do I deploy code", False, 0.82)) == "Retrieve"
    assert controller.decide(TranscriptChunk(1.5, "how do I deploy code engine", False, 0.85)) == "Wait"


def test_controller_agreement_is_primary_even_before_length_thresholds() -> None:
    controller = RuleBasedRetrievalController()
    assert controller.decide(TranscriptChunk(0.5, "how do I", False, 0.60)) == "Wait"
    assert controller.decide(TranscriptChunk(1.0, "how do I deploy", False, 0.65)) == "Retrieve"


def test_controller_does_not_thrash_after_firing() -> None:
    controller = RuleBasedRetrievalController()
    controller.decide(TranscriptChunk(0.5, "how do I deploy", False, 0.8))
    assert controller.decide(TranscriptChunk(1.0, "how do I deploy code", False, 0.82)) == "Retrieve"
    assert controller.decide(TranscriptChunk(1.5, "how do I deploy code engine", False, 0.85)) == "Wait"
    assert controller.decide(TranscriptChunk(2.0, "how do I deploy code engine now", False, 0.88)) == "Wait"


def test_controller_suppresses_presentation_only() -> None:
    controller = RuleBasedRetrievalController()
    for text in ("repeat that in two bullets", "that's all thank you", "can you rephrase that"):
        controller.reset()
        assert controller.decide(TranscriptChunk(1.0, text, False, 0.8)) == "No-Retrieval"


def test_controller_resets_state() -> None:
    controller = RuleBasedRetrievalController()
    controller.decide(TranscriptChunk(0.5, "how do I deploy code", False, 0.8))
    controller.decide(TranscriptChunk(1.0, "how do I deploy code engine", False, 0.85))
    controller.reset()
    assert controller.decide(TranscriptChunk(1.5, "how do I deploy code", False, 0.85)) == "Wait"
