"""Summarize completed dashboard captures without claiming semantic correctness."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from prism_live_rag.telemetry import validate_trace_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--expected", type=int, default=200)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    ids = [row["query_id"] for row in rows]
    unique = len(ids) == len(set(ids))
    trace_ids = {row["dashboard_response"]["trace_id"] for row in rows}
    classes = {}
    eligible, early, early_simulated = 0, 0, 0
    invalid_ids = []
    latencies = []
    for row in rows:
        bucket = classes.setdefault(row["answerability"], Counter())
        bucket["queries"] += 1
        bucket["cited_answers" if row["citations"] else "uncertain_or_uncited"] += 1
        chunks = [e for e in row["events"] if e["type"] == "chunk"]
        final = next((e for e in reversed(chunks) if e["is_final"]), None)
        if row["answerability"] in {"ANSWERABLE", "PARTIAL"} and final and len(chunks) > 1:
            eligible += 1
            early += any(e["type"] == "search_dispatch" and e.get("observed_elapsed_s", float("inf")) < final.get("observed_elapsed_s", 0) for e in row["events"])
            early_simulated += any(e["timestamp_s"] < final["timestamp_s"] for e in row["dashboard_response"]["retrieval_events"])
        if not set(row["citations"]) <= {p["doc_id"] for p in row["retrieved_context"]}:
            invalid_ids.append(row["query_id"])
        latencies.append(row["dashboard_response"]["stage_latency_ms"]["total"])
    result = {
        "queries": len(rows), "expected_queries": args.expected, "unique_query_ids": unique,
        "completed_coverage": len(rows) == args.expected and unique,
        "classes": classes, "invalid_citation_query_ids": invalid_ids,
        "early_retrieval": {"eligible": eligible, "actual_dispatch_before_final": early,
                            "actual_rate": early / eligible if eligible else None,
                            "simulated_timestamp_rate": early_simulated / eligible if eligible else None},
        "pipeline_latency_ms": {"median": statistics.median(latencies) if latencies else None,
                                "max": max(latencies) if latencies else None},
        "unique_trace_ids": len(trace_ids),
        "trace_validation": validate_trace_file(args.traces, expected_ids=trace_ids) if args.traces else None,
        "semantic_correctness": "not evaluated; citations/answer shape are proxies",
        "source": str(args.input),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
