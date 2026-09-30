"""Run the actual dashboard test scenarios, checkpointing each response for G4."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error


def request(base, path, payload=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60 if payload is None else 30) as response:
                return json.load(response)
        except (TimeoutError, urllib.error.URLError):
            # Retry read-only polling, never POST blindly: a timed-out POST
            # may already have created a run on the server.
            if payload is not None or attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--mode", choices=["provider", "deterministic"], default="provider")
    parser.add_argument("--recheck-flagged-from", type=Path,
                        help="Recheck false abstentions and negative-label answers from a saved capture")
    parser.add_argument("--exclude-completed-from", type=Path,
                        help="Skip already saved cases when a transport-only fix needs a new code snapshot")
    parser.add_argument("--scenario-id", action="append",
                        help="Select specific discovered scenarios for a bounded regression check")
    args = parser.parse_args()
    payload = request(args.url, "/api/scenarios")
    scenarios = [s for s in payload["scenarios"] if s["is_evaluated_test"]]
    if args.scenario_id:
        scenarios = [s for s in scenarios if s["id"] in set(args.scenario_id)]
    if args.recheck_flagged_from:
        previous = [json.loads(line) for line in args.recheck_flagged_from.read_text().splitlines() if line.strip()]
        flagged = {
            row["scenario_id"] for row in previous
            if (row["answerability"] == "ANSWERABLE" and not row["citations"])
            or (row["answerability"] in {"UNANSWERABLE", "UNDERSPECIFIED"} and row["citations"])
        }
        scenarios = [s for s in scenarios if s["id"] in flagged]
    if args.exclude_completed_from:
        saved_rows = [json.loads(line) for line in args.exclude_completed_from.read_text().splitlines() if line.strip()]
        saved = {row["scenario_id"] for row in saved_rows if "Provider evidence check failed" not in (row.get("uncertainty") or "")}
        scenarios = [s for s in scenarios if s["id"] not in saved]
    scenarios = scenarios[:args.limit]
    if len(scenarios) != args.limit:
        raise RuntimeError(f"Expected {args.limit} dashboard scenarios, found {len(scenarios)}")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    streams = {}
    for source in {s["source"] for s in scenarios}:
        for line in (data_dir / source).read_text().splitlines():
            if line.strip():
                stream = json.loads(line)
                streams[stream["stream_id"]] = stream
    metadata = {
        "scenario_hash": hashlib.sha256(json.dumps(scenarios, sort_keys=True).encode()).hexdigest(),
        "mode": args.mode, "model": payload.get("provider_model"),
        "server_code_hash": payload.get("runtime_code_hash", "unavailable: older server"),
        "options": {"speed": 4, "multi_intent": True, "llm_decomposer": False, "retry_rate_limits": True},
        "code_hash": payload.get("runtime_code_hash") or hashlib.sha256(b"".join(
            (Path(__file__).resolve().parents[1] / "src/prism_live_rag" / name).read_bytes()
            for name in ["pipeline.py", "decomposer.py", "synthesis.py", "llm_steps.py", "serve.py", "providers.py", "controller.py", "retrieval.py"]
        )).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(".metadata.json")
    active_path = args.output.with_suffix(".active.json")
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            raise RuntimeError("Checkpoint scenario/model/options mismatch; choose a new output path")
    else:
        if args.output.exists():
            raise RuntimeError("Existing output has no metadata; choose a new output path")
        meta_path.write_text(json.dumps(metadata, indent=2))
    rows = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()] if args.output.exists() else []
    completed = {row["scenario_id"] for row in rows}
    if len(completed) != len(rows) or not completed <= {s["id"] for s in scenarios}:
        raise RuntimeError("Invalid or duplicate checkpoint scenario IDs")
    print(f"Resuming {len(completed)}/{len(scenarios)} dashboard results", flush=True)
    for scenario in scenarios:
        if scenario["id"] in completed:
            continue
        active = json.loads(active_path.read_text()) if active_path.exists() else None
        if active and active["scenario_id"] != scenario["id"]:
            raise RuntimeError("Active run does not match next checkpoint scenario")
        if active:
            try:
                request(args.url, "/api/runs/" + active["run_id"])
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                active_path.unlink()
                active = None
        if not active:
            started = request(args.url, "/api/runs", {
                "scenario_id": scenario["id"],
                "options": {"mode": args.mode, **metadata["options"]},
            })
            active = {"scenario_id": scenario["id"], "run_id": started["run_id"]}
            active_path.write_text(json.dumps(active))
        while True:
            snapshot = request(args.url, "/api/runs/" + active["run_id"])
            if snapshot["status"] in {"complete", "error"}:
                break
            time.sleep(2)
        if snapshot["status"] == "error":
            raise RuntimeError(f"Dashboard run failed; checkpoint preserved: {snapshot['error']}")
        response = snapshot["response"]
        stream = streams[scenario["id"]]
        if (
            "Provider evidence check failed" in (response.get("uncertainty") or "")
            or any(e["type"] == "llm_call" and e.get("status") == "failed" and not e.get("will_retry") for e in snapshot["events"])
        ):
            # Retain redacted diagnostic events before removing the retry marker.
            args.output.with_suffix(".failed.json").write_text(json.dumps(snapshot, indent=2))
            active_path.unlink(missing_ok=True)
            raise RuntimeError("A provider call failed; completed checkpoints retained, failed case can be resumed")
        passages = {
            p["id"]: p for event in snapshot["events"]
            if event["type"] == "retrieval_done" for p in event.get("passages", [])
        }
        if not set(response["citations"]) <= passages.keys():
            raise RuntimeError("Response cites a passage absent from this run")
        row = {
            "query_id": stream["task_id"], "scenario_id": scenario["id"],
            "query": scenario["description"], "domain": scenario["domain"],
            "collection": "ibmcloud" if scenario["domain"] == "cloud" else "govt",
            "answerability": scenario["case_class"].upper(),
            "response": response["answer"], "citations": response["citations"],
            "uncertainty": response.get("uncertainty"),
            "abstained": not response["citations"] and response.get("uncertainty") is not None,
            "gold_passage_ids": stream["qrel_passage_ids"],
            "retrieved_context": [{"doc_id": pid, "text": p["text"], "cited": pid in response["citations"], "score": p["score"]} for pid, p in passages.items()],
            "dashboard_response": response, "events": snapshot["events"], "run_id": active["run_id"],
        }
        with args.output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        active_path.unlink(missing_ok=True)
        completed.add(scenario["id"])
        calls = sum(e["type"] == "llm_call" and e.get("status") == "started" for e in snapshot["events"])
        print(f"{len(completed)}/{len(scenarios)} {scenario['case_class']} calls={calls} {scenario['description'][:65]}", flush=True)
    print(f"Completed {len(completed)} dashboard cases: {args.output}", flush=True)


if __name__ == "__main__":
    main()
