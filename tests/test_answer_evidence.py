from prism_live_rag.models import Passage, RetrievedPassage
from prism_live_rag.synthesis import SENTENCE_RE, deduplicate_evidence, route_to_provider, select_answer_evidence, synthesize_answer


def test_context_keeps_complete_entity_anchors_and_current_question():
    from prism_live_rag.synthesis import contextualize_query
    context = (("user", "How do I apply for a California instruction permit?"),
               ("agent", "California DMV instructions. " + "Details. " * 150))
    query = "How do I apply for it?"
    resolved = contextualize_query(query, context)
    assert resolved.startswith("Conversation context: user: How do I apply for a California")
    assert resolved.endswith(" Current question: " + query)


def test_facilities_followup_resolves_modifier_but_unrelated_park_does_not():
    from prism_live_rag.synthesis import needs_clarification
    assert not needs_clarification("Can I barbecue on those camp facilities?", (
        ("user", "Do I need vaccines for camping?"),
        ("agent", "Camping in New York State Parks has no listed vaccine requirements."),
    ))
    assert needs_clarification("Can we take pets in the park?", (
        ("user", "What does MMU stand for?"), ("agent", "MMU is a backpack."),
    ))


def test_discourse_fillers_do_not_become_evidence_requirements():
    from prism_live_rag.synthesis import content_terms
    assert content_terms("Sorry, please tell me what you think, is the fee correct?") == {"fee"}
    assert {"never", "expire"} <= content_terms("The application and fee never expire, correct?")


def test_navigation_and_link_stub_are_not_direct_answers():
    from prism_live_rag.synthesis import is_answer_span
    assert not is_answer_span("FEMA: After a Flood Contact Us Site Map Search After a Flood")
    assert not is_answer_span("For the Deep Impact spacecraft record, see:")
    assert is_answer_span("After a flood, listen for reports about whether the water is safe to drink.")
    assert not is_answer_span("Can I have a barbecue in the park?")
    assert not is_answer_span("The following are guidelines for the period following a flood:")


def test_app_reference_cannot_pick_an_unrelated_old_application():
    from prism_live_rag.synthesis import needs_clarification
    assert needs_clarification("What permissions am I giving to this app?", (
        ("user", "Tell me about a Facebook app."), ("agent", "Facebook apps request permissions."),
        ("user", "How do I upgrade my account?"), ("agent", "Upgrade your account in person."),
    ))


def test_short_named_context_can_resolve_policy_followup():
    from prism_live_rag.synthesis import needs_clarification
    assert not needs_clarification("Can you tell me about its policies?", (
        ("user", "What means RBAC?"),
        ("agent", "RBAC means Role-Based Access Control and governs resource access."),
    ))
    assert not needs_clarification("Can you tell me about its policies?", (
        ("user", "What means RBAC?"), ("agent", "RBAC stands for Role-Based Access Control."),
    ))


def test_glued_scraped_sentences_are_separated_without_changing_text():
    text = "The orbit takes 240 million years.This illustration shows the Milky Way."
    assert SENTENCE_RE.split(text) == [
        "The orbit takes 240 million years.",
        "This illustration shows the Milky Way.",
    ]
    assert SENTENCE_RE.split("Version 1.2.3 costs $3.50.") == ["Version 1.2.3 costs $3.50."]
    assert SENTENCE_RE.split("Our home galaxy.NASA/JPL-CaltechFrom our perspective, it is visible.") == [
        "Our home galaxy.", "NASA/JPL-Caltech", "From our perspective, it is visible.",
    ]


def test_deduplicate_preserves_qualifications_and_source_id():
    short = "Pets are allowed in campsites and designated day use areas."
    long = short + " Except the Gorge Trail, where pets are prohibited."
    assert deduplicate_evidence([("short", short), ("long", long)]) == [("long", long)]
    assert deduplicate_evidence([("long", long), ("short", short)]) == [("long", long)]


def test_conflicting_numbers_are_not_deduplicated():
    spans = [("one", "The fee is 25 dollars."), ("two", "The fee is 50 dollars.")]
    assert deduplicate_evidence(spans) == spans


def test_local_answer_selects_topic_sentence_not_glued_orbit_text():
    text = "The orbit takes 240 million years.This illustration shows the Milky Way, our home galaxy."
    hits = [RetrievedPassage(Passage(id="nasa", domain="govt", text=text), score=1.0)]
    answer, citations, _ = synthesize_answer("What is the name of our galaxy?", hits)
    assert answer == "This illustration shows the Milky Way, our home galaxy."
    assert citations == ["nasa"]


def test_name_lookup_keeps_direct_answer_and_its_citation():
    spans = [
        ("extra", "Our galaxy, the Milky Way, contains a black hole."),
        ("direct", "Our home galaxy is called the Milky Way."),
    ]
    assert select_answer_evidence("What is the name of our galaxy?", spans) == [spans[1]]


def test_name_lookup_does_not_hide_disagreement_or_compound_request():
    spans = [("a", "The mission is called Alpha."), ("b", "The mission is called Beta.")]
    assert select_answer_evidence("What is the name of the mission?", spans) == spans
    assert select_answer_evidence("What is the name of the mission and its purpose?", spans) == spans


def test_explicit_name_lookup_routes_locally_only_with_naming_evidence():
    hits = [RetrievedPassage(Passage(id="name", domain="govt", text="Our home galaxy is called the Milky Way."), score=1.0)]
    assert route_to_provider("What is the name of our galaxy?", hits)[0] is False
    answer, ids, uncertainty = synthesize_answer("What is the name of our galaxy?", hits, require_direct_support=True)
    assert answer == hits[0].passage.text
    assert ids == ["name"]
    assert uncertainty is None
    unrelated = [RetrievedPassage(Passage(id="topic", domain="govt", text="Our galaxy has a black hole."), score=1.0)]
    assert route_to_provider("What is the name of our galaxy?", unrelated)[0] is True


def test_descriptive_definitions_support_api_fields_and_rovers():
    from prism_live_rag.synthesis import supports_definition
    assert supports_definition("What is the sounds_like field in an API request?", "The sounds_like field specifies how a word is pronounced by speakers in audio.")
    assert supports_definition("What are the Mars rovers?", "NASA has sent five robotic vehicles, called rovers, to Mars.")
    assert not supports_definition("What is a TLS handshake?", "The server offloads operations during the TLS handshake.")


def test_generic_followup_requires_coherent_immediate_context():
    from prism_live_rag.synthesis import needs_clarification
    context = (("user", "Jumbo Frames"), ("agent", "Jumbo Frames have a payload above the standard MTU."))
    assert not needs_clarification("How is it configure?", context)
    assert needs_clarification("How is it configure?")
    assert needs_clarification("How is it configure?", (("user", "Jumbo Frames"), ("agent", "The weather is sunny.")))
    assert needs_clarification("Does it support Terraform?", context)


def test_named_configuration_uses_exact_sequential_source_steps_without_llm():
    text = "Configure index rate alerts.\nStep 1. Configure rules\nStep 2. Configure channels\nStep 3. Configure frequency"
    hits = [RetrievedPassage(Passage(id="steps", domain="cloud", text=text), score=1.0)]
    assert not route_to_provider("How do I configure index rate alerts?", hits, multi_intent=True)[0]
    answer, ids, uncertainty = synthesize_answer("How do I configure index rate alerts?", hits, require_direct_support=True)
    assert answer == "Step 1. Configure rules Step 2. Configure channels Step 3. Configure frequency"
    assert ids == ["steps"] and uncertainty is None
    assert route_to_provider("How do I configure calendar reminders?", hits)[0]
