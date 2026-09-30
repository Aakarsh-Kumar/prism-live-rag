from __future__ import annotations

import pytest
from pathlib import Path

from prism_live_rag.models import RagResponse
from prism_live_rag.serve import (
    DashboardHandler,
    ObservedChatClient,
    RunState,
    _append_dashboard_trace,
    evaluate_dashboard_run,
    scenario_uses_decomposition,
)
from prism_live_rag.telemetry import validate_trace_file
from prism_live_rag.providers import ProviderError


def make_request_handler(headers, body=b"{}"):
    import io
    from email.message import Message
    from types import SimpleNamespace

    handler = object.__new__(DashboardHandler)
    handler.headers = Message()
    for name, value in headers.items():
        handler.headers[name] = value
    handler.path = "/api/runs"
    handler.rfile = io.BytesIO(body)
    handler.replies = []
    handler._json = lambda value, status=200: handler.replies.append((status, value))
    handler.dashboard = SimpleNamespace(start_run=lambda payload: SimpleNamespace(run_id="safe-run", status="queued"))
    return handler


@pytest.mark.parametrize("host", ["localhost:8080", "127.0.0.1:8080", "[::1]:8080"])
def test_local_dashboard_accepts_loopback_hosts(host):
    handler = make_request_handler({"Host": host})
    assert handler._local_request()


@pytest.mark.parametrize("host", ["attacker.example:8080", "", "[invalid"])
def test_local_dashboard_rejects_rebinding_hosts(host):
    handler = make_request_handler({"Host": host})
    assert not handler._local_request()
    assert handler.replies[0][0] == 403


@pytest.mark.parametrize("extra_headers,status", [
    ({"Origin": "https://attacker.example"}, 403),
    ({"Origin": "null"}, 403),
    ({"Sec-Fetch-Site": "cross-site"}, 403),
    ({"Content-Type": "text/plain"}, 415),
    ({"Content-Length": "-1"}, 400),
    ({"Content-Length": "64001"}, 400),
    ({"Transfer-Encoding": "chunked"}, 400),
])
def test_dashboard_rejects_unsafe_run_requests_before_reading_body(extra_headers, status):
    handler = make_request_handler({"Host": "localhost:8080", "Content-Type": "application/json",
                                    "Content-Length": "2", **extra_headers})
    handler.do_POST()
    assert handler.replies[0][0] == status
    assert handler.rfile.tell() == 0


def test_dashboard_accepts_same_origin_json_run_request():
    handler = make_request_handler({"Host": "localhost:8080", "Origin": "http://localhost:8080",
                                    "Content-Type": "application/json", "Content-Length": "2"})
    handler.do_POST()
    assert handler.replies == [(202, {"run_id": "safe-run", "status": "queued"})]


class FakeChatClient:
    default_model = "safe-model-name"
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    price_per_million = None
    pricing_basis = None
    api_key = "secret-key-value"

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=512):
        if self.failure:
            raise self.failure
        return "ok"


def test_observed_provider_client_reports_actual_call_lifecycle() -> None:
    events = []
    client = ObservedChatClient(FakeChatClient(), events.append)

    assert client.chat([]) == "ok"
    assert [event["status"] for event in events] == ["started", "completed"]
    assert all(event["model"] == "safe-model-name" for event in events)
    assert client.token_usage["total_tokens"] == 0


def test_observed_provider_client_redacts_key_from_error_event() -> None:
    events = []
    client = ObservedChatClient(
        FakeChatClient(RuntimeError("provider error included secret-key-value")), events.append
    )

    with pytest.raises(RuntimeError):
        client.chat([])
    assert events[-1]["status"] == "failed"
    assert "secret-key-value" not in events[-1]["error"]
    assert "[redacted]" in events[-1]["error"]


def test_offline_quota_retry_reuses_the_same_provider_request():
    class Client(FakeChatClient):
        def __init__(self):
            self.requests = []

        def chat(self, messages, **kwargs):
            self.requests.append(messages)
            if len(self.requests) == 1:
                raise ProviderError("Cerebras request failed: 429 hourly limit exceeded")
            return "supported response"

    client = Client()
    events = []
    observed = ObservedChatClient(client, events.append, retry_rate_limits=True)
    assert observed.chat([{"role": "user", "content": "same query"}]) == "supported response"
    assert client.requests[0] == client.requests[1]
    assert [e["status"] for e in events] == ["started", "failed", "started", "completed"]
    assert events[1]["will_retry"] is True


def test_transient_network_failure_retries_same_messages_with_bounded_attempts():
    import requests

    class Client(FakeChatClient):
        def __init__(self):
            self.requests = []

        def chat(self, messages, **kwargs):
            self.requests.append(messages)
            raise ProviderError("Cerebras network failure") from requests.ConnectionError("DNS failure")

    client = Client()
    events = []
    with pytest.raises(ProviderError):
        ObservedChatClient(client, events.append).chat([{"role": "user", "content": "same query"}])
    assert len(client.requests) == 3
    assert all(item == client.requests[0] for item in client.requests)
    assert [event["will_retry"] for event in events if event["status"] == "failed"] == [True, True, False]


def test_run_state_event_ids_are_monotonic_and_snapshot_catches_up() -> None:
    run = RunState("run-1", {"id": "scenario-1"}, {"mode": "deterministic"})
    run.publish({"type": "chunk", "text": "partial"})
    run.publish({"type": "decision", "decision": "Wait"})

    snapshot = run.snapshot()
    assert [event["id"] for event in snapshot["events"]] == [1, 2]
    assert snapshot["events"][0]["text"] == "partial"
    assert snapshot["status"] == "queued"


@pytest.mark.parametrize(
    ("scenario", "options", "expected"),
    [
        ({"category": "early_retrieval"}, {}, True),
        ({"category": "partial"}, {"multi_intent": False}, False),
        ({"category": "multi_intent"}, {"multi_intent": False}, True),
    ],
)
def test_scenario_decomposition_defaults_on_and_is_forced_for_compound_demo(
    scenario, options, expected
) -> None:
    assert scenario_uses_decomposition(scenario, options) is expected


def test_dashboard_limits_judge_selector_to_evaluated_200_query_set() -> None:
    page = Path(__file__).parents[1] / "src" / "prism_live_rag" / "webui" / "index.html"
    markup = page.read_text(encoding="utf-8")

    assert '<option value="test" selected>Evaluated test set · 200</option>' in markup
    assert 'value="all"' not in markup


def test_dashboard_does_not_overwrite_filtered_initial_query_selection() -> None:
    app = Path(__file__).parents[1] / "src" / "prism_live_rag" / "webui" / "app.js"
    source = app.read_text(encoding="utf-8")

    assert "renderScenarioOptions();\n      if (state.scenarios.length)" not in source
    assert "els.scenario.value = state.scenarios[0].id" not in source
    assert "${shortQuery(row)}" in source


def test_dashboard_scenario_search_uses_the_actual_transcript_query() -> None:
    serve = Path(__file__).parents[1] / "src" / "prism_live_rag" / "serve.py"
    source = serve.read_text(encoding="utf-8")

    assert '"description": stream.base_utterance or' in source


def test_dashboard_run_comparison_uses_expected_label_only_after_response() -> None:
    stream = type("Stream", (), {
        "expected_behavior": "clarify",
        "case_class": "underspecified",
        "qrel_passage_ids": (),
    })()
    result = evaluate_dashboard_run(
        stream,
        RagResponse(answer="Which clinic do you mean?", uncertainty="Referent is missing."),
        is_test=True,
    )

    assert result["actual_behavior_match_proxy"] is True
    assert result["qrel_citation_overlap_proxy"] is None
    assert result["is_evaluated_test"] is True
    assert "not evaluated" in result["note"]


def test_dashboard_does_not_call_gold_overlap_answer_correctness() -> None:
    stream = type("Stream", (), {
        "expected_behavior": "answer",
        "case_class": "answerable",
        "qrel_passage_ids": ("gold-1",),
    })()
    result = evaluate_dashboard_run(
        stream,
        RagResponse(answer="An answer.", citations=["gold-1"]),
        is_test=True,
    )
    assert result["qrel_citation_overlap_proxy"] is True
    assert result["actual_behavior_match_proxy"] is None


def test_dashboard_trace_is_persisted_with_valid_g6_schema(tmp_path) -> None:
    target = tmp_path / "dashboard-traces.jsonl"
    _append_dashboard_trace(target, {
        "trace_id": "trace-1", "status": "complete", "domain": "cloud",
        "retrieval_events": [], "decisions": [], "citations": [], "answer": "ok",
        "stage_latency_ms": {}, "token_usage": {}, "estimated_cost_usd": 0.0,
    })

    result = validate_trace_file(target)
    assert result["valid"] is True
    assert result["traces"] == 1


def test_trace_validation_checks_this_execution_not_unrelated_history(tmp_path):
    target = tmp_path / "traces.jsonl"
    for trace_id in ("older-run", "current-run"):
        _append_dashboard_trace(target, {
            "trace_id": trace_id, "status": "complete", "domain": "cloud",
            "retrieval_events": [], "decisions": [], "citations": [], "answer": "ok",
            "stage_latency_ms": {}, "token_usage": {}, "estimated_cost_usd": 0.0,
        })
    current = validate_trace_file(target, expected_ids={"current-run"})
    assert current["valid"] and current["traces"] == 1 and current["coverage"] == 1
    assert not validate_trace_file(target, expected_ids={"missing-run"})["valid"]
