from prism_live_rag.llm_steps import (
    extract_json_object,
    extract_spans_with_provider,
    provider_synthesize_answer,
    rewrite_query,
)
from prism_live_rag.models import Passage, RetrievedPassage


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages = []

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=512):
        self.messages.append(messages)
        return self.responses.pop(0)


def passage(pid: str, text: str) -> RetrievedPassage:
    return RetrievedPassage(Passage(id=pid, domain="cloud", text=text), score=1.0)


def test_extract_json_object_accepts_fenced_json() -> None:
    assert extract_json_object('```json\n{"ok": true}\n```') == {"ok": True}


def test_extract_json_object_uses_first_balanced_object() -> None:
    assert extract_json_object('note { "text": "literal { brace }" } trailing {"bad": true}') == {
        "text": "literal { brace }"
    }


def test_rewrite_query_uses_rewritten_version() -> None:
    client = FakeClient(['{"class":"standalone","rewritten_version":"azure cli availability south america"}'])
    assert rewrite_query(client, "is it available there?") == "azure cli availability south america"


def test_rewrite_query_falls_back_on_null_rewrite() -> None:
    client = FakeClient(['{"class":"non-standalone","rewritten_version":null}'])
    assert rewrite_query(client, "I heard the toolchain is not available in") == "I heard the toolchain is not available in"


def test_extract_spans_maps_numeric_passage_ids_to_chunk_ids() -> None:
    client = FakeClient(
        [
            '{"extracted_spans":[{"passage_id":1,"sentence":"Azure CLI is available for supported clouds."},'
            '{"passage_id":99,"sentence":"Ignore invalid passage."}]}'
        ]
    )
    spans = extract_spans_with_provider(
        client,
        "Azure CLI availability",
        [passage("cloud-7", "Azure CLI is available for supported clouds.")],
    )
    assert [(span.passage_id, span.sentence) for span in spans] == [
        ("cloud-7", "Azure CLI is available for supported clouds.")
    ]


def test_provider_synthesis_uses_allowlisted_citations() -> None:
    evidence = FakeClient(['{"extracted_spans":[{"passage_id":1,"sentence":"Use the Cloud SDK installer."}]}'])
    generation = FakeClient(["Use the Cloud SDK installer."])
    answer, citations, uncertainty = provider_synthesize_answer(
        evidence,
        generation,
        "How do I install it?",
        [passage("cloud-1", "Use the Cloud SDK installer.")],
    )
    assert answer == "Use the Cloud SDK installer."
    assert citations == ["cloud-1"]
    assert uncertainty is None


def test_provider_synthesis_falls_back_on_bad_json() -> None:
    evidence = FakeClient(["not json", "still not json"])
    generation = FakeClient(["should not be used"])
    answer, citations, uncertainty = provider_synthesize_answer(
        evidence,
        generation,
        "Cloud SDK installer",
        [passage("cloud-1", "Use the Cloud SDK installer.")],
    )
    assert "Cloud SDK installer" in answer
    assert citations == ["cloud-1"]
    assert "Provider synthesis failed" in uncertainty


def test_extract_spans_retries_once_on_bad_json() -> None:
    evidence = FakeClient(
        [
            '{"extracted_spans" [{"passage_id":1,"sentence":"Use the Cloud SDK installer."}]}',
            '{"extracted_spans":[{"passage_id":1,"sentence":"Use the Cloud SDK installer."}]}',
        ]
    )
    spans = extract_spans_with_provider(
        evidence,
        "Cloud SDK installer",
        [passage("cloud-1", "Use the Cloud SDK installer.")],
    )
    assert [(span.passage_id, span.sentence) for span in spans] == [
        ("cloud-1", "Use the Cloud SDK installer.")
    ]


def test_provider_synthesis_retries_empty_spans_with_rewritten_query() -> None:
    evidence = FakeClient(
        [
            '{"extracted_spans":[]}',
            '{"class":"standalone","rewritten_version":"Cloud SDK installer"}',
            '{"extracted_spans":[{"passage_id":1,"sentence":"Use the Cloud SDK installer."}]}',
        ]
    )
    generation = FakeClient(["Use the Cloud SDK installer."])
    answer, citations, uncertainty = provider_synthesize_answer(
        evidence,
        generation,
        "it?",
        [passage("cloud-1", "Use the Cloud SDK installer.")],
    )
    assert answer == "Use the Cloud SDK installer."
    assert citations == ["cloud-1"]
    assert uncertainty is None
