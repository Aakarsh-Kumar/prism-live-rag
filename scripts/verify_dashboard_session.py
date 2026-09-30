"""Verify real dashboard parent/child session state without provider requests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--parent-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-parent", action="store_true")
    args = parser.parse_args()
    parents = [json.loads(line) for line in args.parent_capture.read_text().splitlines() if line.strip()]
    parent = next(row for row in parents if "replica set guarantee" in row["query"] and row["citations"])
    def request(path, payload=None):
        req = urllib.request.Request(args.url + path, data=json.dumps(payload).encode() if payload else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    def execute(payload):
        started = request("/api/runs", payload)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            snapshot = request("/api/runs/" + started["run_id"])
            if snapshot["status"] == "error":
                raise RuntimeError(snapshot["error"])
            if snapshot["status"] == "complete":
                return snapshot
            time.sleep(1)
        raise RuntimeError("Session check timed out")
    def followup(parent_id, text, kind):
        return execute({"scenario_id": parent["scenario_id"], "parent_run_id": parent_id,
                        "follow_up": text, "follow_up_kind": kind,
                        "options": {"mode": "deterministic", "speed": 4}})
    if args.new_parent:
        fresh = execute({"scenario_id": parent["scenario_id"], "options": {"mode": "deterministic", "speed": 4}})
        parent = {**parent, "run_id": fresh["run_id"], "response": fresh["response"]["answer"],
                  "citations": fresh["response"]["citations"], "dashboard_response": fresh["response"]}
        if not parent["citations"]:
            raise RuntimeError("New parent has no grounded answer to refine")
    presented = followup(parent["run_id"], "repeat that in two bullets", "presentation")
    refined = followup(parent["run_id"], "What about storage allocation for CA replica sets?", "refinement")
    current_parent = request("/api/runs/" + parent["run_id"])
    checks = {
        "parent_unchanged": current_parent["response"]["answer"] == parent["response"],
        "presentation_no_retrieval": not presented["response"]["retrieval_events"],
        "presentation_citations_preserved": presented["response"]["citations"] == parent["citations"],
        "presentation_lineage": presented["response"]["previous_version"] == 1 and presented["response"]["version"] == 2,
        "refinement_lineage": refined["response"]["previous_version"] == 1 and refined["response"]["version"] == 2,
        "delta_only_events": all(e["trigger"] == "refinement" for e in refined["response"]["retrieval_events"]),
        "refinement_applied_delta": refined["response"]["applied_delta"] == "What about storage allocation for CA replica sets?",
        "refinement_answer_changed": refined["response"]["answer"] != parent["response"],
        "unaffected_parent_answer_preserved": parent["response"] in refined["response"]["answer"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checks": checks, "parent": parent, "presentation": presented, "refinement": refined}, indent=2))
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("One or more session checks failed; see preserved artifact")


if __name__ == "__main__":
    main()
