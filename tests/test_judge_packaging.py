from pathlib import Path
import json

from prism_live_rag.models import ConversationSession
from prism_live_rag.serve import Dashboard, RunState


def dashboard_with_parent():
    from prism_live_rag.stream import SimulatedStream
    from prism_live_rag.controller import TranscriptChunk
    import threading
    stream = SimulatedStream("example", "task", "early_retrieval", "cloud", (), 0, 0, (),
                             (TranscriptChunk(0, "What is encryption?", True, .99),))
    scenario = {"id": "example", "domain": "cloud", "category": "early_retrieval",
                "stream": stream.to_dict(), "is_evaluated_test": True}
    parent = RunState("parent", scenario, {})
    parent.status = "complete"
    parent.response = {"answer": "Encryption protects traffic.", "citations": ["p1"], "version": 1}
    dashboard = object.__new__(Dashboard)
    dashboard.by_id, dashboard.runs, dashboard.lock = {"example": scenario}, {"parent": parent}, threading.RLock()
    dashboard.executor = type("Executor", (), {"submit": lambda *args: None})()
    return dashboard


def test_followup_copies_state_and_streams_incremental_chunks():
    dashboard = dashboard_with_parent()
    run = dashboard.start_run({"scenario_id": "example", "parent_run_id": "parent",
                               "follow_up": "Actually only include protection against DDoS attacks."})
    assert isinstance(run.session, ConversationSession)
    assert run.session.citations == ["p1"] and run.refinement
    assert len(run.scenario["stream"]["chunks"]) > 1
    assert run.scenario["stream"]["chunks"][-1]["is_final"]
    assert not run.scenario["is_evaluated_test"]
    run.session.citations.append("new")
    assert dashboard.runs["parent"].response["citations"] == ["p1"]


def test_presentation_followup_is_explicit_and_preserves_parent():
    dashboard = dashboard_with_parent()
    run = dashboard.start_run({"scenario_id": "example", "parent_run_id": "parent",
                               "follow_up": "repeat that in two bullets", "follow_up_kind": "presentation"})
    assert not run.refinement and run.session.version == 1


def test_web_assets_are_declared_and_container_excludes_secret_copy():
    root = Path(__file__).parents[1]
    assert 'webui/*.html' in (root / "pyproject.toml").read_text()
    dockerfile = (root / "Dockerfile").read_text()
    assert '"0.0.0.0"' in dockerfile and "EXPOSE 8080" in dockerfile
    assert "COPY .env" not in dockerfile


def test_strict_offline_payment_query_rejects_topic_only_text():
    from prism_live_rag.models import Passage, RetrievedPassage
    from prism_live_rag.synthesis import synthesize_answer
    passage = RetrievedPassage(Passage("p1", "govt", "MoGo rides offer regular bikes and accessible bikes."), 1)
    answer, citations, uncertainty = synthesize_answer("Where do I pay for my MoGo ride?", [passage], require_direct_support=True)
    assert not citations and uncertainty


def test_strict_offline_version_query_requires_the_requested_date():
    from prism_live_rag.models import Passage, RetrievedPassage
    from prism_live_rag.synthesis import synthesize_answer
    passage = RetrievedPassage(Passage("p1", "cloud", "4 November 2021 Git Repos was upgraded to GitLab 14.3.4."), 1)
    _, citations, uncertainty = synthesize_answer("What version was Git Repos upgraded to on 31 October 2021?", [passage], require_direct_support=True)
    assert not citations and uncertainty


def test_additive_followup_preserves_prior_attributes_despite_shared_subject():
    from prism_live_rag.synthesis import refine_answer
    prior = "Replica sets preserve availability. CA replica sets require PostgreSQL."
    delta = "Storage allocation is unchanged because CA replica sets share PostgreSQL."
    answer, citations, uncertainty = refine_answer(prior, ["old"], delta, ["new"], "What about storage allocation for CA replica sets?")
    assert prior in answer and delta in answer
    assert citations == ["old", "new"] and uncertainty is None


def test_provider_date_constraint_cannot_be_dropped_from_changelog_fragment():
    from prism_live_rag.llm_steps import _matches_required_answer_shape
    question = "What version was upgraded on 31 October 2021?"
    assert not _matches_required_answer_shape(question, ": Upgraded to GitLab 14.3.4.")
    assert not _matches_required_answer_shape(question, "4 November 2021 Upgraded to GitLab 14.3.4.")
    assert _matches_required_answer_shape(question, "31 October 2021 Upgraded to version 14.")


def test_run_defaults_to_provider_when_server_key_is_configured():
    dashboard = dashboard_with_parent()
    dashboard.provider_client = object()
    dashboard.provider_available = lambda: True
    run = dashboard.start_run({"scenario_id": "example"})
    assert run.options["mode"] == "provider"


def test_explicit_offline_run_remains_provider_free():
    dashboard = dashboard_with_parent()
    dashboard.provider_client = object()
    run = dashboard.start_run({"scenario_id": "example", "options": {"mode": "deterministic"}})
    assert run.options["mode"] == "deterministic"
