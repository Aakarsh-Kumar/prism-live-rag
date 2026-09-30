from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import os

import requests

from .config import Settings


@dataclass(frozen=True)
class ProviderStatus:
    groq_configured: bool
    deepseek_configured: bool
    cerebras_configured: bool


def provider_status(settings: Settings) -> ProviderStatus:
    return ProviderStatus(
        groq_configured=settings.has_groq,
        deepseek_configured=settings.has_deepseek,
        cerebras_configured=bool(settings.cerebras_api_key),
    )


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    def __init__(self, api_key: str, base_url: str, default_model: str, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout_s = timeout_s
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.price_per_million: tuple[float, float] | None = None
        self.pricing_basis: str | None = None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        content, _ = self.chat_with_usage(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        return content

    def chat_with_usage(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> tuple[str, dict[str, int]]:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model or self.default_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"Provider request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"Provider request failed: {response.status_code} {response.text[:300]}")
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected provider response shape: {payload}") from exc
        usage = payload.get("usage") or {}
        usage_totals = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        for key, value in usage_totals.items():
            self.token_usage[key] += value
        return content, usage_totals


def _configure_pricing(
    client: OpenAICompatibleChatClient,
    provider: str,
    model: str,
    published: tuple[float, float] | None,
    source: str,
) -> OpenAICompatibleChatClient:
    """Attach an auditable estimate; explicit environment rates override snapshots."""
    prefix = provider.upper()
    input_rate = os.getenv(f"{prefix}_INPUT_USD_PER_MILLION")
    output_rate = os.getenv(f"{prefix}_OUTPUT_USD_PER_MILLION")
    if (input_rate is None) != (output_rate is None):
        raise ValueError(f"Both {prefix} input and output prices must be configured together")
    if input_rate is not None:
        prices = (float(input_rate), float(output_rate))
        if any(price < 0 for price in prices):
            raise ValueError(f"{prefix} token prices must be nonnegative")
        client.price_per_million = prices
        client.pricing_basis = f"configured {provider}/{model} USD per million tokens"
    elif published is not None:
        client.price_per_million = published
        client.pricing_basis = f"published {provider}/{model} USD per million tokens, 2026-09-29: {source}"
    return client


def groq_client(settings: Settings) -> OpenAICompatibleChatClient:
    if not settings.groq_api_key:
        raise ProviderError("GROQ_API_KEY is not configured")
    return OpenAICompatibleChatClient(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        default_model=settings.groq_model,
        timeout_s=settings.provider_timeout_s,
    )


def deepseek_client(settings: Settings) -> OpenAICompatibleChatClient:
    if not settings.deepseek_api_key:
        raise ProviderError("DEEPSEEK_API_KEY is not configured")
    return OpenAICompatibleChatClient(
        api_key=settings.deepseek_api_key,
        base_url="https://api.deepseek.com",
        default_model=settings.deepseek_model,
        timeout_s=settings.provider_timeout_s,
    )


class CerebrasChatClient(OpenAICompatibleChatClient):
    """OpenAI-compatible Cerebras client paced at five requests per minute."""

    def __init__(self, api_key: str, model: str, timeout_s: float = 120.0) -> None:
        super().__init__(api_key, "https://api.cerebras.ai/v1", model, timeout_s)
        self._rate_lock = threading.Lock()
        self._last_request_started = 0.0
        self.request_attempts = 0

    def chat_with_usage(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> tuple[str, dict[str, int]]:
        with self._rate_lock:
            delay = 12.1 - (time.monotonic() - self._last_request_started)
            if self._last_request_started and delay > 0:
                time.sleep(delay)
            self._last_request_started = time.monotonic()
            self.request_attempts += 1
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model or self.default_model,
                    "messages": messages,
                    "temperature": temperature,
                    # GPT-OSS budgets reasoning and visible JSON together. Tiny
                    # schema outputs can otherwise finish before emitting content.
                    "max_completion_tokens": max(max_tokens, 1024),
                    "reasoning_effort": "low",
                },
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"Cerebras request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"Cerebras request failed: {response.status_code} {response.text[:300]}"
            )
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Unexpected Cerebras response shape") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Cerebras returned an empty assistant response")
        usage = payload.get("usage") or {}
        totals = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        for key, value in totals.items():
            self.token_usage[key] += value
        return content, totals


def cerebras_client(settings: Settings) -> CerebrasChatClient:
    if not settings.cerebras_api_key:
        raise ProviderError("CEREBRAS_API_KEY is not configured")
    client = CerebrasChatClient(
        api_key=settings.cerebras_api_key,
        model="gpt-oss-120b",
    )
    return _configure_pricing(
        client, "cerebras", "gpt-oss-120b", (0.35, 0.75),
        "https://inference-docs.cerebras.ai/api-reference/models/public-models",
    )
