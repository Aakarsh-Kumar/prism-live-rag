from types import SimpleNamespace

from prism_live_rag.config import Settings
from prism_live_rag.providers import CerebrasChatClient, cerebras_client


def test_cerebras_client_uses_gpt_model_compatible_payload_and_tracks_usage(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"is_multi_intent":false,"sub_queries":["one"]}'}}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26},
            },
        )

    monkeypatch.setattr("prism_live_rag.providers.requests.post", fake_post)
    client = CerebrasChatClient("not-a-real-secret", "gpt-oss-120b")

    content = client.chat(
        [{"role": "user", "content": "test"}], temperature=0, max_tokens=80
    )

    assert "is_multi_intent" in content
    assert captured["url"] == "https://api.cerebras.ai/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-oss-120b"
    assert captured["payload"]["max_completion_tokens"] == 1024
    assert captured["payload"]["reasoning_effort"] == "low"
    assert client.token_usage == {
        "prompt_tokens": 17,
        "completion_tokens": 9,
        "total_tokens": 26,
    }
    assert captured["headers"]["Authorization"] == "Bearer not-a-real-secret"


def test_cerebras_factory_requires_configured_key():
    settings = Settings(cerebras_api_key=None)
    try:
        cerebras_client(settings)
    except RuntimeError as exc:
        assert "CEREBRAS_API_KEY" in str(exc)
    else:
        raise AssertionError("missing Cerebras API key must fail fast")


def test_cerebras_client_paces_requests_at_five_per_minute(monkeypatch):
    clock = {"now": 100.0, "slept": []}
    monkeypatch.setattr("prism_live_rag.providers.time.monotonic", lambda: clock["now"])

    def fake_sleep(delay):
        clock["slept"].append(delay)
        clock["now"] += delay

    monkeypatch.setattr("prism_live_rag.providers.time.sleep", fake_sleep)
    monkeypatch.setattr(
        "prism_live_rag.providers.requests.post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        ),
    )
    client = CerebrasChatClient("key", "gpt-oss-120b")

    client.chat([{"role": "user", "content": "one"}])
    client.chat([{"role": "user", "content": "two"}])

    assert clock["slept"] == [12.1]
    assert client.request_attempts == 2
