from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Passage:
    id: str
    domain: str
    text: str
    title: str = ""
    url: str = ""


@dataclass(frozen=True)
class QueryTask:
    task_id: str
    domain: str
    query: str
    answerability: tuple[str, ...] = ()
    qrel_passage_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalEvent:
    timestamp_s: float
    query: str
    trigger: str


@dataclass(frozen=True)
class RetrievedPassage:
    passage: Passage
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass
class RagResponse:
    retrieval_events: list[RetrievalEvent] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    uncertainty: str | None = None

    def to_dict(self) -> dict:
        return {
            "retrieval_events": [
                {
                    "timestamp_s": event.timestamp_s,
                    "query": event.query,
                    "trigger": event.trigger,
                }
                for event in self.retrieval_events
            ],
            "sub_queries": self.sub_queries,
            "decisions": self.decisions,
            "answer": self.answer,
            "citations": self.citations,
            "uncertainty": self.uncertainty,
        }

