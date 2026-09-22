from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .models import RetrievedPassage
from .providers import ProviderError
from .synthesis import synthesize_answer


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        ...


@dataclass(frozen=True)
class EvidenceSpan:
    passage_id: str
    sentence: str


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    elif not stripped.startswith("{"):
        stripped = _first_balanced_json_object(stripped)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ProviderError("Provider JSON response is not an object")
    return payload


def _first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ProviderError("Provider did not return a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for offset, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    raise ProviderError("Provider did not return a balanced JSON object")


def rewrite_query(client: ChatClient, transcript_text: str) -> str:
    payload = extract_json_object(
        client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the latest user utterance into a standalone retrieval query. "
                        "If the utterance is incomplete, copy it into rewritten_version. "
                        "Return only JSON with keys class and rewritten_version."
                    ),
                },
                {"role": "user", "content": transcript_text},
            ],
            temperature=0.0,
            max_tokens=160,
        )
    )
    value = payload.get("rewritten_version", "")
    rewritten = value.strip() if isinstance(value, str) else ""
    if rewritten.lower() in {"null", "none", "n/a"}:
        rewritten = ""
    return rewritten or transcript_text


def extract_spans_with_provider(
    client: ChatClient,
    query: str,
    passages: list[RetrievedPassage],
    max_spans: int = 4,
) -> list[EvidenceSpan]:
    if not passages:
        return []
    try:
        return _extract_spans_once(client, query, passages, max_spans)
    except (ProviderError, json.JSONDecodeError) as exc:
        if not _is_provider_parse_error(exc):
            raise
        return _extract_spans_once(client, query, passages, max_spans)


def _is_provider_parse_error(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    message = str(exc)
    return (
        "JSON" in message
        or "balanced JSON" in message
        or "extracted_spans" in message
        or "did not return" in message
    )


def _extract_spans_once(
    client: ChatClient,
    query: str,
    passages: list[RetrievedPassage],
    max_spans: int,
) -> list[EvidenceSpan]:
    indexed_passages = []
    id_by_index: dict[str, str] = {}
    valid_ids = set()
    for index, item in enumerate(passages, start=1):
        pid = item.passage.id
        id_by_index[str(index)] = pid
        valid_ids.add(pid)
        indexed_passages.append(
            f"Passage {index}\nchunk_id: {pid}\ntitle: {item.passage.title}\ntext: {item.passage.text}"
        )
    payload = extract_json_object(
        client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract exact supporting evidence spans for the query. Use only the provided passages. "
                        "Return only JSON: {\"extracted_spans\":[{\"passage_id\":1,\"sentence\":\"exact text\"}]}."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\n" + "\n\n".join(indexed_passages),
                },
            ],
            temperature=0.0,
            max_tokens=700,
        )
    )
    raw_spans = payload.get("extracted_spans", [])
    if not isinstance(raw_spans, list):
        raise ProviderError("Evidence response missing extracted_spans list")
    spans: list[EvidenceSpan] = []
    for row in raw_spans:
        if not isinstance(row, dict):
            continue
        raw_id = str(row.get("passage_id", "")).strip()
        passage_id = id_by_index.get(raw_id, raw_id)
        sentence = " ".join(str(row.get("sentence", "")).split())
        if passage_id in valid_ids and sentence:
            spans.append(EvidenceSpan(passage_id=passage_id, sentence=sentence))
        if len(spans) >= max_spans:
            break
    return spans


def generate_grounded_answer(client: ChatClient, query: str, spans: list[EvidenceSpan]) -> str:
    facts = "\n".join(f"- [{span.passage_id}] {span.sentence}" for span in spans)
    answer = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied facts. Do not add outside facts. "
                    "If the facts are insufficient, say so briefly."
                ),
            },
            {"role": "user", "content": f"Question: {query}\n\nFacts:\n{facts}"},
        ],
        temperature=0.0,
        max_tokens=260,
    )
    return " ".join(answer.split())


def provider_synthesize_answer(
    evidence_client: ChatClient | None,
    generation_client: ChatClient | None,
    query: str,
    passages: list[RetrievedPassage],
) -> tuple[str, list[str], str | None]:
    if evidence_client is None or generation_client is None:
        answer, citations, uncertainty = synthesize_answer(query, passages)
        return answer, citations, uncertainty or "Provider clients are not configured; used deterministic fallback."
    try:
        spans = extract_spans_with_provider(evidence_client, query, passages)
        if not spans:
            retry_query = rewrite_query(evidence_client, query)
            if retry_query != query:
                spans = extract_spans_with_provider(evidence_client, retry_query, passages)
        if not spans:
            return (
                "I do not have enough information in the retrieved corpus to answer that.",
                [],
                "No provider evidence spans were found.",
            )
        answer = generate_grounded_answer(generation_client, query, spans)
        citations = list(dict.fromkeys(span.passage_id for span in spans))
        return answer, citations, None
    except (ProviderError, json.JSONDecodeError) as exc:
        answer, citations, uncertainty = synthesize_answer(query, passages)
        detail = f"Provider synthesis failed ({exc}); used deterministic fallback."
        return answer, citations, uncertainty or detail
