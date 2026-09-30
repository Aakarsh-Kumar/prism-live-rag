from prism_live_rag.llm_steps import (
    extract_json_object,
    extract_spans_with_provider,
    provider_synthesize_answer,
    rewrite_query,
)
from prism_live_rag.models import Passage, RetrievedPassage
from prism_live_rag.synthesis import route_to_provider


def test_definition_routing_rejects_matching_topic_in_unrelated_sentence():
    hits = [passage("enterprise-example", "The sample application is called enterprise-app.")]
    assert route_to_provider("What is an enterprise?", hits)[0] is True
    hits = [passage("enterprise-definition", "An enterprise is a hierarchy of accounts managed centrally.")]
    assert route_to_provider("What is an enterprise?", hits)[0] is False


def test_provider_rejects_topic_mention_instead_of_tls_definition():
    text = "Hyper Protect Crypto Services provides a way to offload the cryptographic operations that are done during the TLS handshake to establish a secure connection."
    evidence = FakeClient([json_payload(text)])
    spans = extract_spans_with_provider(
        evidence, "I'm not sure I understand what a TLS handshake is.",
        [passage("tls", text)],
    )
    assert spans == []


def test_provider_accepts_explicit_tls_definition():
    text = "A TLS handshake is a process used to establish a secure connection."
    evidence = FakeClient([json_payload(text)])
    spans = extract_spans_with_provider(
        evidence, "I'm not sure I understand what a TLS handshake is.",
        [passage("tls", text)],
    )
    assert [span.sentence for span in spans] == [text]


def json_payload(text):
    import json
    return json.dumps({"extracted_spans": [{"passage_id": 1, "sentence": text}]})


def test_definition_guard_does_not_reject_attribute_lookup():
    from prism_live_rag.synthesis import supports_definition
    assert supports_definition("What is the name of our galaxy?", "Our galaxy is called the Milky Way.")
    assert supports_definition("What is the release date?", "Released on 31 October 2021.")


def test_provider_rejects_unlabeled_multiple_table_counts():
    text = "Included domains 1 2 2"
    spans = extract_spans_with_provider(FakeClient([json_payload(text)]), "How many domains are included in the plan?", [passage("table", text)])
    assert spans == []


def test_provider_accepts_explicit_insight_benefit_for_why_query():
    text = "When you configure index rate alerts, you can gain insight into which applications produce data spikes."
    spans = extract_spans_with_provider(FakeClient([json_payload(text)]), "Why do I configure index rate alerts?", [passage("benefit", text)])
    assert [span.sentence for span in spans] == [text]


def test_procedure_step_can_inherit_topic_from_its_own_source():
    step = "Configure one or more notification channels."
    text = "Configuring index rate alerts. " + step
    spans = extract_spans_with_provider(FakeClient([json_payload(step)]), "How do I configure index rate alerts?", [passage("steps", text)])
    assert [span.sentence for span in spans] == [step]
    wrong = "Configuring calendar reminders. " + step
    spans = extract_spans_with_provider(FakeClient([json_payload(step)]), "How do I configure index rate alerts?", [passage("wrong", wrong)])
    assert spans == []


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


def test_false_premise_correction_accepts_exact_evidence_without_filler_overlap():
    client = FakeClient(['{"extracted_spans":[{"passage_id":1,"sentence":"The application fee expires after 12 months."}]}'])
    spans = extract_spans_with_provider(
        client, "Sorry, please tell me, the application fee never expires, correct?",
        [passage("fee-rule", "The application fee expires after 12 months.")],
    )
    assert len(spans) == 1
    assert spans[0].passage_id == "fee-rule"


def test_false_premise_does_not_allow_unrelated_exact_fact():
    client = FakeClient(['{"extracted_spans":[{"passage_id":1,"sentence":"The library closes at noon."}]}'])
    assert extract_spans_with_provider(
        client, "Sorry, please tell me, the application fee never expires, correct?",
        [passage("library", "The library closes at noon.")],
    ) == []


def test_sentence_id_selection_resolves_only_real_source_text():
    client = FakeClient(['{"span_ids":["1.2","9.1"]}'])
    spans = extract_spans_with_provider(client, "What is HAART therapy?", [
        passage("therapy", "Pregnancy and HIV\nTaking three HIV medicines is called combination or HAART therapy.")
    ])
    assert [(s.passage_id, s.sentence) for s in spans] == [
        ("therapy", "Taking three HIV medicines is called combination or HAART therapy.")
    ]


def test_navigation_heading_cannot_become_answer():
    client = FakeClient(['{"span_ids":["1.1"]}'])
    assert extract_spans_with_provider(client, "Do they provide online services?", [
        passage("court", "Online Services\nContact Us")
    ]) == []


def test_relevance_check_rejects_topical_fact_without_adding_evidence():
    client = FakeClient([
        '{"span_ids":["1.1"]}',
        '{"supported_span_ids":[],"missing_information":"No supported description of transcription direction."}',
    ])
    answer, ids, uncertainty = provider_synthesize_answer(
        client, client, "Is it Speech to Text or the other way?",
        [passage("audio", "The Speech to Text service detects endianness of incoming audio.")],
        verify_relevance=True,
    )
    assert not ids
    assert uncertainty == "No supported description of transcription direction."


def test_relevance_check_can_keep_partial_evidence_but_not_invent_ids():
    client = FakeClient([
        '{"span_ids":["1.1"]}',
        '{"supported_span_ids":[1,99],"missing_information":"Other planet durations are not documented."}',
    ])
    text = "One day on Mars lasts 24.6 hours."
    answer, ids, uncertainty = provider_synthesize_answer(
        client, client, "How long is one day on Mars and other planets?",
        [passage("mars", text)], verify_relevance=True,
    )
    assert answer == text and ids == ["mars"]
    assert uncertainty == "Other planet durations are not documented."


def test_verified_procedure_can_use_steps_without_repeating_topic_words():
    client = FakeClient([
        '{"span_ids":["1.2"]}', '{"supported_span_ids":[1],"missing_information":null}',
    ])
    answer, ids, uncertainty = provider_synthesize_answer(
        client, client, "What should I do after a flood disaster?",
        [passage("flood", "After a flood\nListen for reports on whether the water supply is safe to drink.")],
        verify_relevance=True,
    )
    assert answer == "Listen for reports on whether the water supply is safe to drink."
    assert ids == ["flood"] and uncertainty is None


def test_relevance_json_parse_retry_is_bounded_and_preserves_question():
    client = FakeClient([
        '{"span_ids":["1.1"]}', '{"supported_span_ids":',
        '{"supported_span_ids":[1],"missing_information":null}',
    ])
    answer, ids, uncertainty = provider_synthesize_answer(
        client, client, "What is an enterprise?",
        [passage("enterprise", "An enterprise is a collection of accounts.")],
        verify_relevance=True,
    )
    assert ids == ["enterprise"] and uncertainty is None
    assert client.messages[1] == client.messages[2]


def test_phone_request_cannot_be_satisfied_with_a_postal_address():
    client = FakeClient(['{"span_ids":["1.1"]}'])
    answer, ids, uncertainty = provider_synthesize_answer(
        client, client, "I want the tele number.",
        [passage("address", "Mail the application to PO Box 4444, Janesville, WI 53547.")],
        verify_relevance=True,
    )
    assert not ids and uncertainty is not None
    assert len(client.messages) == 1  # No verification call for a known wrong attribute.


def test_explicitly_wrong_state_source_is_rejected_without_model_call():
    client = FakeClient([])
    hit = RetrievedPassage(Passage(id="ca-park", domain="govt", text="Barbecue rules vary by park and beach.", url="https://www.parks.ca.gov/FAQ"), score=1.0)
    assert extract_spans_with_provider(client, "Conversation context: Camping in New York State Parks. Current question: Can I barbecue there?", [hit]) == []
    assert client.messages == []


def test_state_scope_guard_does_not_mistake_va_for_virginia():
    from prism_live_rag.llm_steps import _jurisdiction_matches_context
    assert _jurisdiction_matches_context("Apply for a housing grant in New York", "https://www.va.gov/housing-assistance/")
    assert _jurisdiction_matches_context("Compare California and New York parks", "https://parks.ca.gov/")


def test_missing_selection_schema_is_not_a_valid_abstention():
    client = FakeClient(['{}', '{"span_ids":["1.1"]}'])
    spans = extract_spans_with_provider(client, "What is an enterprise?", [
        passage("enterprise", "An enterprise is a collection of accounts.")
    ])
    assert spans[0].passage_id == "enterprise"
    assert len(client.messages) == 2


def test_selected_service_heading_includes_its_contiguous_list():
    from prism_live_rag.llm_steps import _expand_selected_structure
    source = "Online Services\n\nCase Inquiry\nPay Fines\nTraffic Payments\n\nCourt Resources\n\nForms"
    text, measurement = _expand_selected_structure("Online Services", source, "Do they provide online services?")
    assert text == "Online Services Case Inquiry Pay Fines Traffic Payments"
    assert text in " ".join(source.split()) and not measurement


def test_selected_measurement_keeps_label_and_cannot_guess_missing_label():
    from prism_live_rag.llm_steps import _expand_selected_structure
    source = "Planet\nDay Length\n\nMercury\n1,408 hours\n\nVenus\n5,832 hours\n\n24 hours\n\nMars\n25 hours"
    assert _expand_selected_structure("Mercury", source, "Day lengths?") == ("Mercury 1,408 hours", True)
    assert _expand_selected_structure("5,832 hours", source, "Day lengths?") == ("Venus 5,832 hours", True)
    assert _expand_selected_structure("24 hours", source, "Day lengths?") == ("24 hours", False)


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


def test_extract_spans_rejects_provider_paraphrases() -> None:
    client = FakeClient(
        ['{"extracted_spans":[{"passage_id":1,"sentence":"The CLI works everywhere."}]}']
    )
    spans = extract_spans_with_provider(
        client,
        "CLI availability",
        [passage("cloud-7", "Azure CLI is available for supported clouds.")],
    )
    assert spans == []


def test_extract_spans_rejects_exact_but_topically_unresponsive_sentence() -> None:
    client = FakeClient(
        ['{"extracted_spans":[{"passage_id":1,"sentence":"Recommended doses vary by age."}]}']
    )
    spans = extract_spans_with_provider(
        client,
        "Why do people need so many doses?",
        [passage("cloud-7", "Recommended doses vary by age.")],
    )
    assert spans == []


def test_extract_spans_requires_comparison_for_same_or_different_question() -> None:
    client = FakeClient(
        ['{"extracted_spans":[{"passage_id":1,"sentence":"IBM Cloud Object Storage stores immutable objects."}]}']
    )
    spans = extract_spans_with_provider(
        client,
        "Are Immutable Object Storage and IBM Cloud the same or different?",
        [passage("cloud-7", "IBM Cloud Object Storage stores immutable objects.")],
    )
    assert spans == []


def test_extract_spans_rejects_comparison_about_different_entities() -> None:
    client = FakeClient(
        ['{"extracted_spans":[{"passage_id":1,"sentence":"IBM Cloud Object Storage is not the same as the S3 API."}]}']
    )
    spans = extract_spans_with_provider(
        client,
        "Trying to understand if Immutable Object Storage and IBM Cloud are the same or different?",
        [passage("cloud-7", "IBM Cloud Object Storage is not the same as the S3 API.")],
    )
    assert spans == []


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
    assert generation.messages == []


def test_provider_synthesis_abstains_on_bad_json_instead_of_weak_fallback() -> None:
    evidence = FakeClient(["not json", "still not json"])
    generation = FakeClient(["should not be used"])
    answer, citations, uncertainty = provider_synthesize_answer(
        evidence,
        generation,
        "Cloud SDK installer",
        [passage("cloud-1", "Use the Cloud SDK installer.")],
    )
    assert "could not verify" in answer
    assert citations == []
    assert "no answer was emitted" in uncertainty
    assert generation.messages == []


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


def test_provider_synthesis_abstains_without_rewriting_a_rejected_question() -> None:
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
    assert "not have enough information" in answer
    assert citations == []
    assert uncertainty is not None
    assert len(evidence.messages) == 1
    assert generation.messages == []
