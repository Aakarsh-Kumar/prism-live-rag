#!/usr/bin/env python3
"""Exercise G5 two-turn continuity with a controlled, cited passage fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_live_rag.cli import _write_trace
from prism_live_rag.controller import TranscriptChunk
from prism_live_rag.models import ConversationSession, Passage, RetrievedPassage
from prism_live_rag.pipeline import StreamingRagPipeline
from prism_live_rag.telemetry import validate_trace_file


class FixtureRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, domain: str):
        self.queries.append(query)
        if "availability" in query and "billing" in query:
            rows = (
                ("base-service", "Service availability: The service is available in several regions."),
                ("base-billing", "Billing help: Billing help is available every day."),
            )
        elif "California" in query:
            rows = (("delta-california", "California permits the service for enterprise accounts."),)
        else:
            rows = ()
        return [
            RetrievedPassage(Passage(id=pid, domain=domain, text=text), score=0.9)
            for pid, text in rows
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    if args.trace.exists():
        raise SystemExit(f"Trace file already exists: {args.trace}")
    retriever = FixtureRetriever()
    pipeline = StreamingRagPipeline(retriever)
    session = ConversationSession()
    first = pipeline.run(
        [
            TranscriptChunk(0.4, "What is service availability", False, 0.8),
            TranscriptChunk(0.9, "What is service availability and billing", False, 0.85),
            TranscriptChunk(1.5, "What is service availability and billing help?", True, 0.99),
        ],
        session=session, trace=True,
    )
    base_queries = list(retriever.queries)
    assert first.version == 1 and first.citations == ["base-service", "base-billing"]
    assert "several regions" in first.answer and "every day" in first.answer

    retriever.queries.clear()
    refined = pipeline.run(
        [
            TranscriptChunk(0.4, "maybe California service", False, 0.8),
            TranscriptChunk(0.8, "maybe California service", False, 0.83),
            TranscriptChunk(1.3, "for California only", True, 0.99),
        ],
        session=session, refinement=True, trace=True,
    )
    refinement_queries = list(retriever.queries)
    assert refined.previous_version == 1 and refined.version == session.version == 2
    assert refined.applied_delta == "for California only"
    assert "every day" in refined.answer and "several regions" not in refined.answer
    assert "California permits" in refined.answer
    assert refined.citations == ["base-billing", "delta-california"]
    assert "base-service" not in refined.citations
    assert refinement_queries[-1] == "for California only"
    assert all("availability and billing" not in query for query in refinement_queries)
    assert all(event.trigger == "refinement" for event in refined.retrieval_events)

    retriever.queries.clear()
    unsupported = pipeline.run(
        [TranscriptChunk(1.0, "for Nevada only", True, 0.99)],
        session=session, refinement=True, trace=True,
    )
    assert retriever.queries == ["for Nevada only"]
    assert unsupported.answer == refined.answer and unsupported.citations == refined.citations
    assert unsupported.version == session.version == 3 and unsupported.uncertainty

    retriever.queries.clear()
    presented = pipeline.run(
        [TranscriptChunk(1.0, "repeat that in two bullets", True, 0.99)],
        session=session, trace=True,
    )
    assert retriever.queries == []
    assert presented.answer.startswith("- ") and "\n- " in presented.answer
    assert presented.citations == unsupported.citations
    assert presented.version == session.version == 4

    for response in (first, refined, unsupported, presented):
        _write_trace(str(args.trace), {**response.to_dict(), "domain": "cloud"})
    trace_validation = validate_trace_file(args.trace, expected=4)
    assert trace_validation["valid"]

    report = {
        "protocol": "controlled two-turn passage fixture; interval-delivered ASR partials and final revision; deterministic synthesis; no production-corpus quality claim",
        "checks_passed": 4,
        "checks_total": 4,
        "initial": {"queries": base_queries, "answer": first.answer, "citations": first.citations, "version": first.version},
        "late_constraint": {"queries": refinement_queries, "answer": refined.answer, "citations": refined.citations, "version": refined.version, "previous_version": refined.previous_version, "applied_delta": refined.applied_delta},
        "unsupported_constraint": {"answer_preserved": unsupported.answer == refined.answer, "version": unsupported.version, "uncertainty": unsupported.uncertainty},
        "presentation_only": {"queries": retriever.queries, "answer": presented.answer, "version": presented.version},
        "trace_validation": trace_validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks_passed": 4, "checks_total": 4}))


if __name__ == "__main__":
    main()
