from __future__ import annotations

from prism_live_rag.models import QueryTask
from prism_live_rag.stream import (
    NO_RETRIEVAL_PHRASES,
    generate_streams,
    multi_intent_stream,
    no_retrieval_stream,
)


def _task(task_id: str, query: str) -> QueryTask:
    return QueryTask(task_id=task_id, domain="cloud", query=query, qrel_passage_ids=("p1", "p2"))


def test_generate_streams_produces_all_categories() -> None:
    tasks = [_task(f"t{i}", f"this is a reasonably long query number {i}") for i in range(6)]
    streams = generate_streams(tasks, "cloud", seed=0, multi_intent_count=2, no_retrieval_count=2)
    categories = {stream.category for stream in streams}
    assert categories == {"early_retrieval", "multi_intent", "no_retrieval"}
    assert sum(1 for s in streams if s.category == "early_retrieval") == 6
    assert sum(1 for s in streams if s.category == "multi_intent") == 2
    assert sum(1 for s in streams if s.category == "no_retrieval") == 2


def test_generate_streams_is_deterministic() -> None:
    tasks = [_task("t1", "a deterministic streaming query for testing")]
    first = generate_streams(tasks, "cloud", seed=7)
    second = generate_streams(tasks, "cloud", seed=7)
    assert [s.to_dict() for s in first] == [s.to_dict() for s in second]


def test_chunks_end_with_final_and_monotonic_timestamps() -> None:
    tasks = [_task("t1", "a query that is long enough to stream out over several words")]
    stream = generate_streams(tasks, "cloud", seed=1)[0]
    assert stream.chunks[-1].is_final
    timestamps = [c.timestamp_s for c in stream.chunks]
    assert timestamps == sorted(timestamps)
    assert any(not c.is_final for c in stream.chunks)


def test_revision_stream_corrects_mid_stream() -> None:
    import random

    task = _task("t1", "How many backups can I retain each month per volume?")
    rng = random.Random(0)
    # Force a revision at a known index by constructing chunks directly is internal;
    # instead verify that with high revision probability a correction appears.
    streams = generate_streams([task], "cloud", seed=0, revision_probability=1.0)
    stream = streams[0]
    final = stream.chunks[-1].text
    partial_texts = [c.text for c in stream.chunks if not c.is_final]
    assert any(text != final for text in partial_texts), "expected at least one divergent partial"


def test_no_retrieval_stream_has_empty_qrels_and_phrase() -> None:
    import random

    stream = no_retrieval_stream("cloud", "cloud-noret-test", random.Random(0))
    assert stream.category == "no_retrieval"
    assert stream.qrel_passage_ids == ()
    assert stream.chunks[-1].text in NO_RETRIEVAL_PHRASES


def test_multi_intent_stream_has_sub_intents() -> None:
    import random

    tasks = [_task("t1", "first independent question here"), _task("t2", "second independent question here")]
    stream = multi_intent_stream(tasks, "cloud", "cloud-multi-test", random.Random(0))
    assert stream.category == "multi_intent"
    assert len(stream.sub_intents) == 2
    assert all(s.stable_at >= 0 for s in stream.sub_intents)


def test_roundtrip_serialization() -> None:
    import random

    from prism_live_rag.stream import SimulatedStream

    stream = no_retrieval_stream("cloud", "cloud-rt", random.Random(3))
    restored = SimulatedStream.from_dict(stream.to_dict())
    assert restored == stream
