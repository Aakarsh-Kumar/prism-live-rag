import json

from prism_live_rag.cli import _write_trace
from prism_live_rag.telemetry import validate_trace_file


def test_trace_sink_appends_one_json_record_per_execution(tmp_path) -> None:
    path = tmp_path / "nested" / "traces.jsonl"
    _write_trace(str(path), {"answer": "first", "estimated_cost_usd": 0.0,
                             "cost_estimate_basis": "no provider tokens"})
    _write_trace(str(path), {"answer": "second", "trace_id": "stable-id",
                             "estimated_cost_usd": 0.0,
                             "cost_estimate_basis": "no provider tokens"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["answer"] for row in rows] == ["first", "second"]
    assert rows[0]["trace_id"]
    assert rows[1]["trace_id"] == "stable-id"
    assert all(row["recorded_at_utc"] and row["status"] == "complete" for row in rows)
    assert validate_trace_file(path, expected=2)["valid"]


def test_trace_validation_flags_missing_execution_coverage(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    _write_trace(str(path), {"answer": "one"})
    report = validate_trace_file(path, expected=2)
    assert not report["valid"]
    assert report["coverage"] == 0.5
    assert any("expected 2 traces" in problem for problem in report["problems"])
