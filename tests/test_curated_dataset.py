from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from prism_live_rag.curated_dataset import (
    CuratedCase,
    CuratedDatasetError,
    SupportSpan,
    audit_case,
    attach_distractors,
    assigned_split,
    export_review_csv,
    generate_curated_streams,
    generate_cases_for_passage,
    import_reviews,
    normalized_document_group,
    seed_benchmark_negatives,
    seed_benchmark_positives,
    shortlist_cases,
    validate_case,
    validate_curated_streams,
    validate_dataset,
)
from prism_live_rag.models import Passage


class FakeClient:
    default_model = "fake-model"

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def chat(self, messages, **kwargs):
        return json.dumps(self.payload)


def passage(passage_id: str = "cloud-page-0-100", domain: str = "cloud") -> Passage:
    text = (
        "To create an account, open the console and select Create account. "
        "Enter a verified email address, complete the required fields, and confirm "
        "the account through the link sent by email. Account creation itself is free."
    )
    return Passage(
        id=passage_id,
        domain=domain,
        text=text,
        url=f"https://example.test/{passage_id}",
    )


def approved_case(p: Passage, *, split: str, case_id: str = "case-1") -> CuratedCase:
    support = "Account creation itself is free."
    start = p.text.index(support)
    return CuratedCase(
        case_id=case_id,
        split=split,
        domain=p.domain,
        case_class="answerable",
        query="Is account creation free?",
        expected_behavior="answer",
        source_document_group=normalized_document_group(p),
        reference_answer="Account creation itself is free.",
        gold_passage_ids=[p.id],
        support_spans=[SupportSpan(p.id, start, start + len(support), support)],
        provenance={"source": "corpus_generated"},
        generation={"seed": 20260928},
        review={"status": "approved", "reviewer": "reviewer-1"},
    )


def test_generator_requires_exact_verbatim_support() -> None:
    p = passage()
    support = "Account creation itself is free."
    client = FakeClient(
        [
            {
                "case_class": "answerable",
                "query": "Is account creation free?",
                "reference_answer": support,
                "required_clarification": "",
                "support_spans": [support],
                "difficulty": "easy",
                "tags": [],
                "dialogue_history": [],
            }
        ]
    )
    split = assigned_split(normalized_document_group(p), 20260928)
    cases = generate_cases_for_passage(
        client, p, split=split, requested_classes=["answerable"], seed=20260928
    )
    assert cases[0].support_spans[0].text == support
    assert cases[0].support_spans[0].start == p.text.index(support)

    client.payload[0]["support_spans"] = ["This sentence was invented."]
    with pytest.raises(CuratedDatasetError, match="not an exact substring"):
        generate_cases_for_passage(
            client, p, split=split, requested_classes=["answerable"], seed=20260928
        )


def test_validation_detects_changed_span_and_gold_distractor_overlap() -> None:
    p = passage()
    split = assigned_split(normalized_document_group(p), 20260928)
    case = approved_case(p, split=split)
    case.distractor_passage_ids = [p.id]
    case.support_spans[0] = SupportSpan(p.id, 0, 4, "wrong")
    problems = validate_case(case, {p.id: p})
    assert any("also a distractor" in problem for problem in problems)
    assert any("support span does not match" in problem for problem in problems)


def test_dataset_validation_detects_source_document_leakage() -> None:
    p = passage()
    train = approved_case(p, split="train", case_id="train-case")
    dev = approved_case(p, split="dev", case_id="dev-case")
    train.provenance = {"source": "fixture"}
    dev.provenance = {"source": "fixture"}
    report = validate_dataset(
        {"train": [train], "dev": [dev]}, {p.id: p}, require_targets=False
    )
    assert report["valid"] is False
    assert any("source-document leakage" in problem for problem in report["problems"])


def test_review_round_trip_requires_reviewer_for_approval(tmp_path: Path) -> None:
    p = passage()
    split = assigned_split(normalized_document_group(p), 20260928)
    case = approved_case(p, split=split)
    case.review = {"status": "pending"}
    review_path = tmp_path / "review.csv"
    export_review_csv([case], review_path)

    rows = list(csv.DictReader(review_path.open(encoding="utf-8")))
    rows[0]["status"] = "approved"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(CuratedDatasetError, match="requires reviewer"):
        import_reviews([case], review_path)

    rows[0]["reviewer"] = "human-1"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    reviewed = import_reviews([case], review_path)
    assert reviewed[0].review["status"] == "approved"


def test_seed_benchmark_negatives_preserves_history_and_labels(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    rows = []
    for label, count in (("UNANSWERABLE", 2), ("UNDERSPECIFIED", 2)):
        for index in range(count):
            rows.append(
                {
                    "task_id": f"task-{label}-{index}",
                    "conversation_id": f"conversation-{label}-{index}",
                    "Collection": "ibmcloud" if index % 2 == 0 else "govt",
                    "answerability": [label],
                    "input": [
                        {"speaker": "user", "text": "Tell me about accounts."},
                        {"speaker": "agent", "text": "What would you like to know?"},
                        {"speaker": "user", "text": "How much does it cost?"},
                    ],
                    "targets": [{"speaker": "agent", "text": "Which account type?"}],
                }
            )
    reference.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cases = seed_benchmark_negatives(
        reference,
        quotas={"unanswerable": 2, "underspecified": 2},
    )
    assert len(cases) == 4
    assert all(case.query == "How much does it cost?" for case in cases)
    assert all(len(case.dialogue_history) == 2 for case in cases)
    assert all(case.provenance["source"] == "mtragun" for case in cases)
    clarification = [case for case in cases if case.case_class == "underspecified"]
    assert all(case.required_clarification == "Which account type?" for case in clarification)


def test_test_negatives_must_come_from_mtragun() -> None:
    case = CuratedCase(
        case_id="negative-1",
        split="test",
        domain="cloud",
        case_class="unanswerable",
        query="What is the unsupported price?",
        expected_behavior="abstain",
        source_document_group="synthetic:1",
        provenance={"source": "corpus_generated"},
        review={"status": "approved", "reviewer": "human-1"},
    )
    problems = validate_case(case, {})
    assert any("must preserve MTRAG-UN provenance" in problem for problem in problems)


def test_seed_benchmark_positive_uses_existing_gold_and_exact_span(tmp_path: Path) -> None:
    p = passage()
    reference = tmp_path / "reference.jsonl"
    row = {
        "task_id": "task-answerable-1",
        "conversation_id": "conversation-answerable-1",
        "Collection": "ibmcloud",
        "answerability": ["ANSWERABLE"],
        "input": [{"speaker": "user", "text": "Is account creation free?"}],
        "targets": [{"speaker": "agent", "text": "Account creation itself is free."}],
        "contexts": [{"document_id": p.id, "text": p.text}],
    }
    reference.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cases = seed_benchmark_positives(
        reference, {p.id: p}, quotas={"answerable": 1}
    )
    assert cases[0].gold_passage_ids == [p.id]
    assert cases[0].support_spans
    span = cases[0].support_spans[0]
    assert p.text[span.start : span.end] == span.text


def test_attach_distractors_excludes_gold_and_deduplicates() -> None:
    p = passage()
    split = assigned_split(normalized_document_group(p), 20260928)
    case = approved_case(p, split=split)
    attach_distractors(
        [case],
        lambda query, domain, limit: [p.id, "distractor-1", "distractor-1", "distractor-2"],
        limit=2,
    )
    assert case.distractor_passage_ids == ["distractor-1", "distractor-2"]
    assert "hard_distractor" in case.tags


def test_audit_is_advisory_and_shortlist_does_not_approve(monkeypatch) -> None:
    p = passage()
    cases = []
    targets = {"answerable": 2, "partial": 2, "unanswerable": 1, "underspecified": 1}
    monkeypatch.setitem(
        __import__("prism_live_rag.curated_dataset", fromlist=["SPLIT_TARGETS"]).SPLIT_TARGETS,
        "test",
        targets,
    )
    for index in range(2):
        answerable = approved_case(p, split="test", case_id=f"answerable-{index}")
        answerable.review = {"status": "pending"}
        cases.append(answerable)
    for index, domain in enumerate(("cloud", "govt", "govt")):
        partial = approved_case(p, split="test", case_id=f"partial-{index}")
        partial.domain = domain
        partial.case_class = "partial"
        partial.expected_behavior = "partial_answer"
        partial.reference_answer += " The passage does not provide the price."
        partial.review = {"status": "pending"}
        cases.append(partial)
    for case_class in ("unanswerable", "underspecified"):
        cases.append(
            CuratedCase(
                case_id=case_class,
                split="test",
                domain="cloud",
                case_class=case_class,
                query="What does that cost?",
                expected_behavior="abstain" if case_class == "unanswerable" else "clarify",
                source_document_group=f"benchmark:{case_class}",
                required_clarification="Which service?" if case_class == "underspecified" else "",
                provenance={"source": "mtragun"},
            )
        )
    shortlisted, audit_rows = shortlist_cases(cases, "test")
    assert len(shortlisted) == 6
    assert all(case.review.get("status") == "pending" for case in shortlisted)
    chosen_partial_domains = {
        case.domain for case in shortlisted if case.case_class == "partial"
    }
    assert chosen_partial_domains == {"cloud", "govt"}
    assert len(audit_rows) == len(cases)
    assert "score" in audit_case(cases[0])


def test_curated_stream_is_incremental_and_final_supersedes_partials() -> None:
    p = passage()
    case = approved_case(p, split="test")
    case.dialogue_history = [
        {"speaker": "user", "text": "Tell me about account creation."},
        {"speaker": "agent", "text": "What would you like to know?"},
    ]
    streams = generate_curated_streams(
        [case], seed=7, revision_probability=1.0
    )
    stream = streams[0]
    assert len(stream.chunks) > 2
    assert all(not chunk.is_final for chunk in stream.chunks[:-1])
    assert stream.chunks[-1].is_final
    assert stream.chunks[-1].text.split() == case.query.split()
    assert stream.asr_supersedes == tuple(range(len(stream.chunks) - 1))
    assert stream.context_turns[0][0] == "user"
    timestamps = [chunk.timestamp_s for chunk in stream.chunks]
    assert all(after > before for before, after in zip(timestamps, timestamps[1:]))
    report = validate_curated_streams([case], streams)
    assert report["valid"] is True


def test_curated_stream_validation_rejects_flattened_single_chunk() -> None:
    from dataclasses import replace

    p = passage()
    case = approved_case(p, split="test")
    stream = generate_curated_streams([case], seed=3)[0]
    flattened = replace(
        stream,
        chunks=(stream.chunks[-1],),
        asr_supersedes=(),
    )
    report = validate_curated_streams([case], [flattened])
    assert report["valid"] is False


def test_curated_stream_validation_rejects_empty_inputs() -> None:
    report = validate_curated_streams([], [])
    assert report["valid"] is False
    assert "no curated cases were loaded" in report["problems"]
    assert "no curated streams were loaded" in report["problems"]
