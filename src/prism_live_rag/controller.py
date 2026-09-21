from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptChunk:
    timestamp_s: float
    text: str
    is_final: bool = False
    confidence: float | None = None


class RuleBasedRetrievalController:
    def __init__(self, min_chars: int = 24, min_tokens: int = 5, min_confidence: float = 0.72) -> None:
        self.min_chars = min_chars
        self.min_tokens = min_tokens
        self.min_confidence = min_confidence

    def decide(self, chunk: TranscriptChunk) -> str:
        text = chunk.text.strip()
        if not text:
            return "No-Retrieval"
        if chunk.confidence is not None and chunk.confidence < self.min_confidence and not chunk.is_final:
            return "Wait"
        if chunk.is_final:
            return "Retrieve"
        if len(text) >= self.min_chars and len(text.split()) >= self.min_tokens:
            return "Retrieve"
        return "Wait"


def simulate_chunks(text: str, step_words: int = 4, interval_s: float = 0.8) -> list[TranscriptChunk]:
    words = text.split()
    chunks: list[TranscriptChunk] = []
    for index in range(step_words, len(words), step_words):
        chunks.append(
            TranscriptChunk(
                timestamp_s=round(len(chunks) * interval_s, 2),
                text=" ".join(words[:index]),
                is_final=False,
                confidence=0.78,
            )
        )
    chunks.append(
        TranscriptChunk(
            timestamp_s=round(len(chunks) * interval_s, 2),
            text=text,
            is_final=True,
            confidence=0.95,
        )
    )
    return chunks

