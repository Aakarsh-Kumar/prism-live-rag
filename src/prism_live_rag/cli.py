from __future__ import annotations

import argparse
import json

from .config import load_settings
from .controller import simulate_chunks
from .data import load_query_tasks, validate_domain
from .pipeline import StreamingRagPipeline
from .providers import ProviderError, deepseek_client, groq_client, provider_status
from .retrieval import HybridRetriever, LanceDbTimeout, LanceIndex


def cmd_validate(args: argparse.Namespace) -> None:
    settings = load_settings()
    status = provider_status(settings)
    result = {
        "providers": {
            "groq_configured": status.groq_configured,
            "deepseek_configured": status.deepseek_configured,
        },
        "domains": {
            domain: validate_domain(settings.data_dir, domain)
            for domain in args.domains
        },
    }
    print(json.dumps(result, indent=2))


def cmd_index(args: argparse.Namespace) -> None:
    settings = load_settings()
    index = LanceIndex(settings.lancedb_dir, settings.table_name, settings.embedding_dim)
    try:
        count = index.build(settings.data_dir, args.domains, limit=args.limit, timeout_s=args.timeout_s)
    except LanceDbTimeout as exc:
        raise SystemExit(
            f"{exc}. LanceDB is not responding in this Python runtime; run indexing "
            "through Docker (`docker compose run --rm app prism-rag index`) or a "
            "known-good Python 3.11-3.13 environment."
        ) from exc
    print(json.dumps({"indexed_passages": count, "lancedb_dir": str(settings.lancedb_dir)}, indent=2))


def cmd_run_demo(args: argparse.Namespace) -> None:
    settings = load_settings()
    tasks = load_query_tasks(settings.data_dir, args.domain)
    if not tasks:
        raise SystemExit(f"No query tasks found for domain: {args.domain}")
    task = tasks[args.task_index % len(tasks)]
    retriever = HybridRetriever(
        settings.data_dir,
        settings.lancedb_dir,
        settings.table_name,
        settings.embedding_dim,
        settings.rrf_k,
        settings.sparse_weight,
        corpus_limit=args.corpus_limit,
        use_dense=args.use_dense,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
    )
    evidence_client = None
    generation_client = None
    if args.mode == "provider":
        try:
            evidence_client = deepseek_client(settings)
            generation_client = groq_client(settings)
        except ProviderError as exc:
            print(json.dumps({"warning": str(exc), "fallback": "deterministic"}, indent=2))
    pipeline = StreamingRagPipeline(
        retriever,
        synthesis_mode=args.mode,
        evidence_client=evidence_client,
        generation_client=generation_client,
        enable_query_rewrite=args.rewrite_query,
        refine_on_final=args.refine_on_final or args.mode == "provider",
    )
    try:
        response = pipeline.run(simulate_chunks(task.query), domain=args.domain)
    except LanceDbTimeout as exc:
        raise SystemExit(
            f"{exc}. Retry without `--use-dense`, or run dense retrieval through Docker."
        ) from exc
    print(json.dumps(response.to_dict(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prism-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data")
    validate.add_argument("--domains", nargs="+", default=["cloud", "govt"])
    validate.set_defaults(func=cmd_validate)

    index = sub.add_parser("index")
    index.add_argument("--domains", nargs="+", default=["cloud", "govt"])
    index.add_argument("--limit", type=int, default=None)
    index.add_argument("--timeout-s", type=int, default=300)
    index.set_defaults(func=cmd_index)

    demo = sub.add_parser("run-demo")
    demo.add_argument("--domain", choices=["cloud", "govt"], default="cloud")
    demo.add_argument("--task-index", type=int, default=0)
    demo.add_argument("--corpus-limit", type=int, default=5000)
    demo.add_argument("--use-dense", action="store_true", help="Use LanceDB dense search; run `prism-rag index` first.")
    demo.add_argument("--mode", choices=["deterministic", "provider"], default="deterministic")
    demo.add_argument("--rewrite-query", action="store_true", help="Use DeepSeek to rewrite the live utterance before retrieval.")
    demo.add_argument(
        "--refine-on-final",
        action="store_true",
        help="Keep early retrieval, then refresh retrieval when the final transcript arrives.",
    )
    demo.set_defaults(func=cmd_run_demo)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
