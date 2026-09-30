from unittest.mock import Mock, patch

from prism_live_rag.evaluation_framework import _cerebras_rate_limited_completion


def test_cerebras_transport_retry_is_paced_and_preserves_request():
    import requests
    from prism_live_rag.evaluation_framework import _cerebras_post_with_network_retry
    response = Mock(status_code=200)
    with (
        patch("prism_live_rag.evaluation_framework.requests.post", side_effect=[requests.exceptions.ConnectionError("DNS failure"), response]) as post,
        patch("prism_live_rag.evaluation_framework.time.sleep") as sleep,
    ):
        assert _cerebras_post_with_network_retry("endpoint", json={"model": "gpt-oss-120b"}) is response
    assert post.call_args_list[0] == post.call_args_list[1]
    sleep.assert_called_once_with(12.5)


def test_cerebras_transport_retry_exhaustion_raises():
    import pytest
    import requests
    from prism_live_rag.evaluation_framework import _cerebras_post_with_network_retry
    with (
        patch("prism_live_rag.evaluation_framework.requests.post", side_effect=requests.exceptions.Timeout("timeout")) as post,
        patch("prism_live_rag.evaluation_framework.time.sleep") as sleep,
    ):
        with pytest.raises(requests.exceptions.Timeout):
            _cerebras_post_with_network_retry("endpoint")
    assert post.call_count == 3
    assert sleep.call_count == 2


def test_cerebras_callback_paces_calls_and_bounds_completion() -> None:
    response = Mock(status_code=200)
    response.json.return_value = {
        "choices": [{"message": {"content": "supported"}}]
    }
    monotonic = Mock(side_effect=[100.0, 100.0, 105.0, 105.0])
    callback = _cerebras_rate_limited_completion("test-key", "gpt-oss-120b")

    with (
        patch("prism_live_rag.evaluation_framework.requests.post", return_value=response) as post,
        patch("prism_live_rag.evaluation_framework.time.monotonic", monotonic),
        patch("prism_live_rag.evaluation_framework.time.sleep") as sleep,
    ):
        outputs = callback(["prompt one", "prompt two"])

    assert outputs == ["supported", "supported"]
    assert post.call_count == 2
    assert post.call_args.kwargs["json"]["model"] == "gpt-oss-120b"
    assert post.call_args.kwargs["json"]["reasoning_effort"] == "low"
    assert post.call_args.kwargs["json"]["max_completion_tokens"] == 2048
    sleep.assert_called_once_with(7.5)


def test_cerebras_callback_retries_hourly_limit_at_five_rpm() -> None:
    limited = Mock(status_code=429, text='{"message":"Requests per hour limit exceeded"}')
    success = Mock(status_code=200)
    success.json.return_value = {"choices": [{"message": {"content": "supported"}}]}
    monotonic = Mock(side_effect=[100.0, 100.0, 100.0, 112.5])
    callback = _cerebras_rate_limited_completion("test-key", "gpt-oss-120b")

    with (
        patch(
            "prism_live_rag.evaluation_framework.requests.post",
            side_effect=[limited, success],
        ) as post,
        patch("prism_live_rag.evaluation_framework.time.monotonic", monotonic),
        patch("prism_live_rag.evaluation_framework.time.sleep") as sleep,
    ):
        assert callback(["claim check"]) == ["supported"]

    assert post.call_count == 2
    sleep.assert_called_once_with(12.5)
