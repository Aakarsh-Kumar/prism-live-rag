from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptChunk:
    timestamp_s: float
    text: str
    is_final: bool = False
    confidence: float | None = None


PRESENTATION_COMMANDS = (
    "repeat",
    "summarize",
    "summarise",
    "speak",
    "rephrase",
    "can you rephrase",
    "go back",
    "start over",
    "never mind",
    "hold on",
    "stop",
    "pause",
    "what did you mean",
    "that's all",
    "thats all",
    "thank you",
    "thanks",
    "got it",
    "sounds good",
)


class RuleBasedRetrievalController:
    def __init__(
        self,
        min_chars: int = 24,
        min_tokens: int = 5,
        min_confidence: float = 0.72,
        agreement_n: int = 2,
        min_stable_words: int = 3,
    ) -> None:
        self.min_chars = min_chars
        self.min_tokens = min_tokens
        self.min_confidence = min_confidence
        self.agreement_n = agreement_n
        self.min_stable_words = min_stable_words
        self._window: list[str] = []
        self._fired = False
        self.last_reason = ""

    def reset(self) -> None:
        self._window = []
        self._fired = False
        self.last_reason = ""

    def decide(self, chunk: TranscriptChunk) -> str:
        text = chunk.text.strip()
        if not text:
            self.last_reason = "empty text"
            return "No-Retrieval"
        if self._is_presentation_only(text):
            self.last_reason = "presentation-only command"
            return "No-Retrieval"
        if chunk.is_final:
            self.last_reason = "final transcript"
            return "Retrieve"
        if chunk.confidence is not None and chunk.confidence < self.min_confidence:
            self._window.append(text)
            self.last_reason = f"low confidence {chunk.confidence}"
            return "Wait"
        if len(text) < self.min_chars or len(text.split()) < self.min_tokens:
            self._window.append(text)
            self.last_reason = "too short for retrieval"
            return "Wait"
        self._window.append(text)
        committed = self._committed_prefix_len()
        if committed >= self.min_stable_words:
            if not self._fired:
                self._fired = True
                self.last_reason = f"stable intent ({committed} words agreed)"
                return "Retrieve"
            self.last_reason = "already retrieved; awaiting final"
            return "Wait"
        self.last_reason = f"intent not yet stable ({committed}/{self.min_stable_words} words)"
        return "Wait"

    def _committed_prefix_len(self) -> int:
        recent = self._window[-self.agreement_n :]
        if len(recent) < self.agreement_n:
            return 0
        tokenized = [text.split() for text in recent]
        shortest = min(tokenized, key=len)
        committed = 0
        for index in range(len(shortest)):
            token = shortest[index]
            if all(len(text) > index and text[index] == token for text in tokenized):
                committed += 1
            else:
                break
        return committed

    def _is_presentation_only(self, text: str) -> bool:
        lowered = text.lower().lstrip()
        return any(lowered.startswith(command) for command in PRESENTATION_COMMANDS)


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
