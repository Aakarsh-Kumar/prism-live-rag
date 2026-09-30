#!/usr/bin/env python3
"""Compare controller trigger timing on identical interval-delivered ASR streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_live_rag.config import load_settings
from prism_live_rag.controller import RuleBasedRetrievalController, SemanticRetrievalController
from prism_live_rag.data import load_streams
from prism_live_rag.embeddings import build_encoder
from prism_live_rag.evaluation import evaluate_streaming
from prism_live_rag.pipeline import StreamingRagPipeline


class EmptyRetriever:
    """The ablation measures decisions, so no corpus search is necessary."""

    def search(self, query: str, domain: str):
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    settings = load_settings()
    encoder = build_encoder(
        "fastembed", model=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir, local_files_only=True,
        device=args.device, batch_size=settings.embedding_batch_size,
        fixed_length=settings.embedding_fixed_length,
    )
    report = {
        "protocol": "same saved timestamped ASR streams, identical no-op retriever, trigger timing only; semantic controller opt-in and not trained",
        "encoder": encoder.name,
        "device": encoder.device,
        "domains": {},
    }
    for domain in ("cloud", "govt"):
        streams = load_streams(settings.data_dir, domain)
        results = {}
        for label, controller in (
            ("rule_based", RuleBasedRetrievalController()),
            ("semantic", SemanticRetrievalController(encoder=encoder)),
        ):
            pipeline = StreamingRagPipeline(EmptyRetriever(), controller=controller)
            results[label] = evaluate_streaming(
                streams, lambda chunks: pipeline.run(chunks, domain=domain)
            )
            print(f"{domain} {label}: {results[label]['early_retrieval_rate']} early", flush=True)
        report["domains"][domain] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": {
        domain: {label: {"eligible": result["eligible"], "early_retrieval_rate": result["early_retrieval_rate"], "false_trigger_rate": result["false_trigger_rate"]}
                 for label, result in values.items()}
        for domain, values in report["domains"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
