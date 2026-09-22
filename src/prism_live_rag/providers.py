from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import Settings


@dataclass(frozen=True)
class ProviderStatus:
    groq_configured: bool
    deepseek_configured: bool


def provider_status(settings: Settings) -> ProviderStatus:
    return ProviderStatus(
        groq_configured=settings.has_groq,
        deepseek_configured=settings.has_deepseek,
    )


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    def __init__(self, api_key: str, base_url: str, default_model: str, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
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
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected provider response shape: {payload}") from exc


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
