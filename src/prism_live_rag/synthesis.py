from __future__ import annotations

import re

from .models import RetrievedPassage


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def extract_evidence(query: str, passages: list[RetrievedPassage], max_spans: int = 4) -> list[tuple[str, str]]:
    query_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", query) if len(term) > 2}
    spans: list[tuple[int, str, str]] = []
    for item in passages:
        for sentence in SENTENCE_RE.split(item.passage.text.strip()):
            clean = " ".join(sentence.split())
            if not clean:
                continue
            sentence_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", clean)}
            overlap = len(query_terms & sentence_terms)
            if overlap:
                spans.append((overlap, item.passage.id, clean))
    spans.sort(key=lambda row: row[0], reverse=True)
    return [(pid, sentence) for _, pid, sentence in spans[:max_spans]]


def synthesize_answer(query: str, passages: list[RetrievedPassage]) -> tuple[str, list[str], str | None]:
    evidence = extract_evidence(query, passages)
    if not evidence:
        return (
            "I do not have enough information in the retrieved corpus to answer that.",
            [],
            "No supporting evidence spans were found.",
        )
    citations = []
    answer_parts = []
    for passage_id, sentence in evidence:
        citations.append(passage_id)
        answer_parts.append(sentence)
    return " ".join(answer_parts), list(dict.fromkeys(citations)), None

