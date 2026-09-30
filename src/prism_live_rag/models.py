from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Passage:
    id: str
    domain: str
    text: str
    title: str = ""
    url: str = ""
    #: URLs lifted out of the prose by loader-side cleaning. See docs/corpus-spec.md §3.5.
    links: tuple[str, ...] = ()
    #: SHA-256 of the raw MTRAG text, so a cleaned passage can be traced back to source.
    text_raw_sha256: str = ""
    #: If this passage was kept because a duplicate group referenced it by qrel, the
    #: non-gold id it displaced is recorded here for audit. Empty otherwise.
    dropped_duplicate_of: str = ""
    #: Flattened cleaning provenance as sorted ``(key, value)`` string pairs, e.g.
    #: ``(("chars_removed", "412"), ("stripped", "nav_chrome"))``. Values are coerced to
    #: ``str`` so the field is genuinely hashable and ``Passage`` stays usable in sets.
    #: Empty when the passage was not cleaned.
    cleaning: tuple[tuple[str, str], ...] = ()


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
    claim_citations: dict[str, list[str]] = field(default_factory=dict)
    uncertainty: str | None = None
    version: int | None = None
    previous_version: int | None = None
    applied_delta: str | None = None
    trace_id: str | None = None
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
    cost_estimate_basis: str | None = None

    def to_dict(self) -> dict:
        payload = {
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
        if self.version is not None:
            payload.update({
                "version": self.version,
                "previous_version": self.previous_version,
                "applied_delta": self.applied_delta,
                "claim_citations": self.claim_citations,
            })
        if self.trace_id is not None:
            payload.update({
                "trace_id": self.trace_id,
                "stage_latency_ms": self.stage_latency_ms,
                "token_usage": self.token_usage,
                "estimated_cost_usd": self.estimated_cost_usd,
                "cost_estimate_basis": self.cost_estimate_basis,
            })
        return payload


@dataclass
class ConversationSession:
    """Small explicit state object for incremental answer refinement."""

    answer: str = ""
    citations: list[str] = field(default_factory=list)
    claim_citations: dict[str, list[str]] = field(default_factory=dict)
    uncertainty: str | None = None
    version: int = 0
    domain: str | None = None

    def commit(self, response: RagResponse, domain: str) -> None:
        self.answer = response.answer
        self.citations = list(response.citations)
        self.claim_citations = {claim: list(ids) for claim, ids in response.claim_citations.items()}
        self.uncertainty = response.uncertainty
        self.version = response.version or self.version + 1
        self.domain = domain
