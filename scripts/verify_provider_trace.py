#!/usr/bin/env python3
"""One bounded Cerebras GPT-OSS-120B grounding and cost telemetry check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_live_rag.cli import _run_traced
from prism_live_rag.config import load_settings
from prism_live_rag.controller import TranscriptChunk
from prism_live_rag.models import Passage, RetrievedPassage
from prism_live_rag.pipeline import StreamingRagPipeline
from prism_live_rag.providers import cerebras_client
from prism_live_rag.telemetry import validate_trace_file


class FixedRetriever:
    def search(self, query: str, domain: str):
        return [RetrievedPassage(Passage(
            id="provider-smoke-passage", domain=domain,
            text="IBM Cloud Transit Gateway does not perform encryption; it only provides connectivity.",
        ), score=0.9)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    args = parser.parse_args()
    if args.trace.exists():
        raise SystemExit(f"Trace file already exists: {args.trace}")
    client = cerebras_client(load_settings())
    pipeline = StreamingRagPipeline(
        FixedRetriever(), synthesis_mode="provider",
        evidence_client=client, generation_client=client,
    )
    response = _run_traced(
        pipeline,
        [TranscriptChunk(1.0, "Does IBM Cloud Transit Gateway perform encryption?", True, 0.99)],
        "cloud", str(args.trace),
    )
    validation = validate_trace_file(args.trace, expected=1)
    result = {
        "model": client.default_model,
        "response_citations": response.citations,
        "token_usage": response.token_usage,
        "estimated_cost_usd": response.estimated_cost_usd,
        "cost_estimate_basis": response.cost_estimate_basis,
        "trace_validation": validation,
    }
    print(json.dumps(result, indent=2))
    if not validation["valid"] or response.estimated_cost_usd is None or response.token_usage["total_tokens"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
