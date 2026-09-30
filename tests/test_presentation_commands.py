"""Tests for presentation-command detection in the live streaming path.

Regression guard for a symmetric defect: plain ``startswith`` matching both missed the
common polite phrasings and silently suppressed genuine questions that merely began with
the same letters. Under-coverage causes spurious retrieval; over-coverage is a recall bug
in the live path, so both directions are asserted here.
"""

import pytest

from prism_live_rag.controller import (
    RuleBasedRetrievalController,
    SemanticRetrievalController,
    TranscriptChunk,
    is_presentation_only,
)

#: Requests about delivery. Retrieval must be suppressed; prior citations are reused.
SUPPRESS = [
    "repeat that",
    "can you repeat that",
    "could you repeat the question",
    "say that again",
    "read that back to me",
    "what did you just say",
    "go over that again",
    "rephrase the answer",
    "can you rephrase that",
    "please summarize that",
    "I want you to rephrase the answer",
    "go back",
    "go back to the previous point",
    "start over",
    "never mind",
    "hold on",
    "stop",
    "pause",
    "thanks",
    "got it",
    "sounds good",
    "that's all",
    "what did you mean",
    "what did you mean by that",
]

#: Real content. These must retrieve, including ones that start with the same letters as
#: a command word.
RETRIEVE = [
    "stoplight timing in Denver Colorado",
    "pausing mid-call is not permitted in this plan",
    "hold on to the waiver requirement",
    "speak up about the enrollment deadline",
    "start over with the new form",
    "repeat interval in the waiver is 30 days",
    "never mind the deadline is Friday instead",
    "thanks for the update, when is the deadline",
    "the toolchain is not available in South America",
    "I heard the toolchain is not available in South America",
]


@pytest.mark.parametrize("text", SUPPRESS)
def test_presentation_commands_are_suppressed(text: str) -> None:
    assert is_presentation_only(text), text


@pytest.mark.parametrize("text", RETRIEVE)
def test_real_content_is_not_suppressed(text: str) -> None:
    assert not is_presentation_only(text), text


@pytest.mark.parametrize("text", SUPPRESS)
def test_controllers_agree_on_suppression(text: str) -> None:
    """Both controllers must not diverge, or the demo depends on which one is wired."""
    chunk = TranscriptChunk(1.0, text, True, 0.95)
    assert RuleBasedRetrievalController().decide(chunk) == "No-Retrieval"
    assert SemanticRetrievalController().decide(chunk) == "No-Retrieval"


@pytest.mark.parametrize("text", RETRIEVE)
def test_controllers_agree_on_retrieval(text: str) -> None:
    chunk = TranscriptChunk(1.0, text, True, 0.95)
    assert RuleBasedRetrievalController().decide(chunk) == "Retrieve"
    assert SemanticRetrievalController().decide(chunk) == "Retrieve"


def test_presentation_command_as_final_chunk_does_not_re_retrieve() -> None:
    """The live demo beat: provisional retrieval, then "repeat that" as the final chunk.

    Prior citations must be reused with no new query.
    """
    from prism_live_rag.models import Passage, RetrievedPassage
    from prism_live_rag.pipeline import StreamingRagPipeline

    class Retriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, domain: str):
            self.queries.append(query)
            return [
                RetrievedPassage(
                    Passage(
                        id="p1",
                        domain=domain,
                        text="The toolchain is not available in South America.",
                    ),
                    score=1.0,
                )
            ]

    question = "I heard the toolchain is not available in South America"
    retriever = Retriever()
    response = StreamingRagPipeline(retriever).run(
        [
            TranscriptChunk(0.8, question, False, 0.90),
            TranscriptChunk(1.6, question, False, 0.92),
            TranscriptChunk(2.4, "repeat that", True, 0.97),
        ],
        "cloud",
    )
    assert retriever.queries == [question]
    assert [d["decision"] for d in response.decisions] == ["Wait", "Retrieve", "No-Retrieval"]
    assert response.citations == ["p1"]
