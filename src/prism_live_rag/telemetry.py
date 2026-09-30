"""Local JSONL trace validation for the G6 execution-coverage check."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_TRACE_FIELDS = {
    "trace_id",
    "recorded_at_utc",
    "status",
    "domain",
    "retrieval_events",
    "decisions",
    "citations",
    "answer",
    "version",
    "previous_version",
    "applied_delta",
    "stage_latency_ms",
    "token_usage",
    "estimated_cost_usd",
    "cost_estimate_basis",
}


def validate_trace_file(path: str | Path, *, expected: int | None = None,
                        expected_ids: set[str] | None = None) -> dict:
    """Check trace schema, uniqueness, and expected execution coverage."""
    target = Path(path)
    if expected_ids is not None:
        expected = len(expected_ids)
    problems: list[str] = []
    rows: list[dict] = []
    if not target.is_file():
        return {
            "valid": False,
            "path": str(target),
            "expected_executions": expected,
            "traces": 0,
            "coverage": 0.0 if expected else None,
            "problems": ["trace file does not exist"],
        }

    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            problems.append(f"line {line_number}: trace must be a JSON object")
            continue
        if expected_ids is not None and row.get("trace_id") not in expected_ids:
            continue
        missing = sorted(REQUIRED_TRACE_FIELDS - row.keys())
        if missing:
            problems.append(f"line {line_number}: missing fields {', '.join(missing)}")
        if row.get("status") not in {"complete", "error"}:
            problems.append(f"line {line_number}: invalid execution status")
        if not isinstance(row.get("retrieval_events"), list) or not isinstance(row.get("decisions"), list):
            problems.append(f"line {line_number}: decisions and retrieval_events must be arrays")
        if not isinstance(row.get("stage_latency_ms"), dict) or not isinstance(row.get("token_usage"), dict):
            problems.append(f"line {line_number}: latency and token_usage must be objects")
        cost = row.get("estimated_cost_usd")
        if cost is not None and (not isinstance(cost, (int, float)) or cost < 0):
            problems.append(f"line {line_number}: estimated_cost_usd must be null or nonnegative")
        if row.get("status") == "complete" and cost is None:
            problems.append(f"line {line_number}: complete trace has no cost estimate")
        if not row.get("cost_estimate_basis"):
            problems.append(f"line {line_number}: cost_estimate_basis is missing")
        rows.append(row)

    trace_ids = [row.get("trace_id") for row in rows]
    if any(not trace_id for trace_id in trace_ids):
        problems.append("one or more traces have an empty trace_id")
    if len(trace_ids) != len(set(trace_ids)):
        problems.append("trace_id values are not unique")
    if expected_ids is not None and expected_ids - set(trace_ids):
        problems.append("missing expected trace IDs: " + ", ".join(sorted(expected_ids - set(trace_ids))))
    coverage = None if expected is None else min(len(rows) / expected, 1.0) if expected > 0 else 1.0
    if expected is not None and len(rows) != expected:
        problems.append(f"expected {expected} traces but found {len(rows)}")
    return {
        "valid": not problems,
        "path": str(target),
        "expected_executions": expected,
        "traces": len(rows),
        "coverage": coverage,
        "problems": problems,
    }
