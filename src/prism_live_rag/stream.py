from __future__ import annotations

import random
from dataclasses import dataclass

from .controller import TranscriptChunk
from .models import QueryTask

WORDS_PER_SECOND = 2.5
PUNCTUATION_PAUSE_S = 0.32
MIN_STABLE_WORDS = 3
AGREEMENT_N = 2

CONFUSIONS = {
    "from": "for",
    "for": "fire",
    "available": "available at",
    "not": "now",
    "the": "a",
    "can": "can't",
    "in": "and",
    "is": "isn't",
    "how": "who",
    "what": "that",
}

NO_RETRIEVAL_PHRASES = (
    "repeat that in two bullets",
    "speak more slowly please",
    "summarize what you just said",
    "go back to the previous point",
    "can you rephrase that",
    "that's all thank you",
    "hold on",
    "what did you mean by that",
    "never mind",
    "start over",
)


@dataclass(frozen=True)
class SubIntent:
    text: str
    stable_at: int


@dataclass(frozen=True)
class SimulatedStream:
    stream_id: str
    task_id: str
    category: str
    domain: str
    qrel_passage_ids: tuple[str, ...]
    stability_chunk_index: int
    settling_ms: int
    sub_intents: tuple[SubIntent, ...]
    chunks: tuple[TranscriptChunk, ...]

    def to_dict(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "task_id": self.task_id,
            "category": self.category,
            "domain": self.domain,
            "qrel_passage_ids": list(self.qrel_passage_ids),
            "stability_chunk_index": self.stability_chunk_index,
            "settling_ms": self.settling_ms,
            "sub_intents": [{"text": s.text, "stable_at": s.stable_at} for s in self.sub_intents],
            "chunks": [
                {
                    "timestamp_s": chunk.timestamp_s,
                    "text": chunk.text,
                    "is_final": chunk.is_final,
                    "confidence": chunk.confidence,
                }
                for chunk in self.chunks
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> SimulatedStream:
        return cls(
            stream_id=payload["stream_id"],
            task_id=payload["task_id"],
            category=payload["category"],
            domain=payload["domain"],
            qrel_passage_ids=tuple(payload["qrel_passage_ids"]),
            stability_chunk_index=payload["stability_chunk_index"],
            settling_ms=payload["settling_ms"],
            sub_intents=tuple(SubIntent(s["text"], s["stable_at"]) for s in payload["sub_intents"]),
            chunks=tuple(
                TranscriptChunk(
                    timestamp_s=chunk["timestamp_s"],
                    text=chunk["text"],
                    is_final=chunk["is_final"],
                    confidence=chunk.get("confidence"),
                )
                for chunk in payload["chunks"]
            ),
        )


def _word_durations(words: list[str]) -> list[float]:
    durations: list[float] = []
    elapsed = 0.0
    for word in words:
        duration = (1.0 / WORDS_PER_SECOND) + 0.03 * len(word)
        if word.endswith((".", ",", "?", "!", ";", ":")):
            duration += PUNCTUATION_PAUSE_S
        elapsed += duration
        durations.append(round(elapsed, 2))
    return durations


def _confidence(progress: float, is_final: bool, rng: random.Random) -> float:
    if is_final:
        return round(rng.uniform(0.96, 0.99), 2)
    base = 0.5 + 0.4 * progress
    return round(min(0.93, base + rng.uniform(-0.03, 0.03)), 2)


def _mishear(word: str, rng: random.Random) -> str:
    lowered = word.lower()
    if lowered in CONFUSIONS:
        return CONFUSIONS[lowered]
    if len(word) >= 4:
        index = rng.randrange(len(word) - 1)
        return word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    return word


def _prefix_tokens(partials: list[list[str]]) -> list[int]:
    return [len(_common_prefix(partials[max(0, i - AGREEMENT_N + 1) : i + 1])) for i in range(len(partials))]


def _common_prefix(texts: list[list[str]]) -> list[str]:
    if not texts:
        return []
    shortest = min(texts, key=len)
    prefix: list[str] = []
    for index in range(len(shortest)):
        token = shortest[index]
        if all(len(text) > index and text[index] == token for text in texts):
            prefix.append(token)
        else:
            break
    return prefix


def _build_chunks(
    words: list[str],
    durations: list[float],
    rng: random.Random,
    revision_index: int | None,
) -> list[TranscriptChunk]:
    if not words:
        return []
    chunks: list[TranscriptChunk] = []
    correct_words = list(words)
    misheard_words = list(words)
    correction_index: int | None = None
    if revision_index is not None:
        misheard_words[revision_index] = _mishear(words[revision_index], rng)
        span = min(3, max(1, len(words) - revision_index - 1))
        correction_index = revision_index + rng.randint(1, span)
    total = len(words)
    for index in range(1, total + 1):
        if correction_index is not None and index >= correction_index:
            emitting = correct_words
        elif revision_index is not None and index > revision_index:
            emitting = misheard_words
        else:
            emitting = correct_words
        text = " ".join(emitting[:index])
        chunks.append(
            TranscriptChunk(
                timestamp_s=durations[index - 1],
                text=text,
                is_final=False,
                confidence=_confidence(index / total, False, rng),
            )
        )
    chunks.append(
        TranscriptChunk(
            timestamp_s=round(durations[-1] + rng.uniform(0.2, 0.5), 2),
            text=" ".join(correct_words),
            is_final=True,
            confidence=_confidence(1.0, True, rng),
        )
    )
    return chunks


def _stability_chunk_index(chunks: list[TranscriptChunk], words: list[str]) -> int:
    partials = [chunk.text.split() for chunk in chunks if not chunk.is_final]
    committed = _prefix_tokens(partials)
    for index, count in enumerate(committed):
        if count >= MIN_STABLE_WORDS:
            return index
    if len(words) <= MIN_STABLE_WORDS:
        return len(partials) - 1
    return -1


def _settling_ms(chunks: list[TranscriptChunk], stability_index: int) -> int:
    if stability_index < 0 or not chunks:
        return -1
    first = chunks[0].timestamp_s
    stable = chunks[stability_index].timestamp_s
    return round((stable - first) * 1000)


def _stream(
    *,
    stream_id: str,
    task_id: str,
    category: str,
    domain: str,
    qrels: tuple[str, ...],
    words: list[str],
    sub_intents: tuple[SubIntent, ...],
    rng: random.Random,
    revision_index: int | None = None,
) -> SimulatedStream:
    durations = _word_durations(words)
    chunks = _build_chunks(words, durations, rng, revision_index)
    stability_index = _stability_chunk_index(chunks, words)
    return SimulatedStream(
        stream_id=stream_id,
        task_id=task_id,
        category=category,
        domain=domain,
        qrel_passage_ids=qrels,
        stability_chunk_index=stability_index,
        settling_ms=_settling_ms(chunks, stability_index),
        sub_intents=sub_intents,
        chunks=tuple(chunks),
    )


def early_retrieval_stream(
    task: QueryTask, domain: str, stream_id: str, rng: random.Random, revision: bool = False
) -> SimulatedStream:
    words = task.query.split()
    revision_index = None
    if revision and len(words) >= 6:
        low = max(1, int(len(words) * 0.25))
        high = min(len(words) - 2, int(len(words) * 0.7))
        if low <= high:
            revision_index = rng.randrange(low, high + 1)
    return _stream(
        stream_id=stream_id,
        task_id=task.task_id,
        category="early_retrieval",
        domain=domain,
        qrels=task.qrel_passage_ids,
        words=words,
        sub_intents=(),
        rng=rng,
        revision_index=revision_index,
    )


def multi_intent_stream(
    tasks: list[QueryTask], domain: str, stream_id: str, rng: random.Random
) -> SimulatedStream:
    selected = tasks[:3] if len(tasks) >= 2 else tasks
    joined = " . ".join(task.query for task in selected)
    words = joined.split()
    durations = _word_durations(words)
    chunks = _build_chunks(words, durations, rng, None)
    sub_intents: list[SubIntent] = []
    offset = 0
    for task in selected:
        task_words = task.query.split()
        for index in range(len(chunks)):
            if chunks[index].is_final:
                sub_intents.append(SubIntent(task.query, index))
                break
            if " ".join(words[offset : offset + len(task_words)]) in chunks[index].text:
                sub_intents.append(SubIntent(task.query, index))
                break
        offset += len(task_words) + 1
    stability_index = _stability_chunk_index(chunks, words)
    qrels = tuple(pid for task in selected for pid in task.qrel_passage_ids)
    return SimulatedStream(
        stream_id=stream_id,
        task_id=selected[0].task_id,
        category="multi_intent",
        domain=domain,
        qrel_passage_ids=qrels,
        stability_chunk_index=stability_index,
        settling_ms=_settling_ms(chunks, stability_index),
        sub_intents=tuple(sub_intents),
        chunks=tuple(chunks),
    )


def no_retrieval_stream(domain: str, stream_id: str, rng: random.Random) -> SimulatedStream:
    text = rng.choice(NO_RETRIEVAL_PHRASES)
    words = text.split()
    return _stream(
        stream_id=stream_id,
        task_id="no-retrieval",
        category="no_retrieval",
        domain=domain,
        qrels=(),
        words=words,
        sub_intents=(),
        rng=rng,
    )


def generate_streams(
    tasks: list[QueryTask],
    domain: str,
    *,
    seed: int = 0,
    multi_intent_count: int = 10,
    no_retrieval_count: int = 10,
    revision_probability: float = 0.2,
) -> list[SimulatedStream]:
    rng = random.Random(seed)
    streams: list[SimulatedStream] = []
    for index, task in enumerate(tasks):
        revision = rng.random() < revision_probability
        streams.append(
            early_retrieval_stream(task, domain, f"{domain}-{index:04d}", rng, revision=revision)
        )
    for index in range(multi_intent_count):
        offset = (index * 2) % max(1, len(tasks) - 1)
        group = tasks[offset : offset + 2]
        if len(group) == 2:
            streams.append(
                multi_intent_stream(group, domain, f"{domain}-multi-{index:04d}", rng)
            )
    for index in range(no_retrieval_count):
        streams.append(no_retrieval_stream(domain, f"{domain}-noret-{index:04d}", rng))
    return streams
