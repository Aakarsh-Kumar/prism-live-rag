from __future__ import annotations

import argparse
import json

from .config import load_settings
from .controller import simulate_chunks
from .data import load_query_tasks, load_streams, save_streams, validate_domain
from .embeddings import build_encoder
from .evaluation import corpus_id_set, evaluate_provider, evaluate_retrieval, evaluate_streaming
from .pipeline import StreamingRagPipeline
from .providers import ProviderError, deepseek_client, groq_client, provider_status
from .retrieval import RETRIEVAL_LEGS, HybridRetriever, LanceDbTimeout, LanceIndex
from .stream import generate_streams


def _build_encoder(settings, args: argparse.Namespace):
    return build_encoder(
        getattr(args, "embedding_backend", None) or settings.embedding_backend,
        model=getattr(args, "embedding_model", None) or settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
        hash_dim=settings.embedding_dim,
    )


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
    encoder = _build_encoder(settings, args)
    index = LanceIndex(settings.lancedb_dir, settings.table_name, encoder)
    try:
        count = index.build(settings.data_dir, args.domains, limit=args.limit, timeout_s=args.timeout_s)
    except LanceDbTimeout as exc:
        raise SystemExit(
            f"{exc}. LanceDB is not responding in this Python runtime; run indexing "
            "through Docker (`docker compose run --rm app prism-rag index`) or a "
            "known-good Python 3.11-3.13 environment."
        ) from exc
    print(
        json.dumps(
            {
                "indexed_passages": count,
                "encoder": encoder.name,
                "dim": encoder.dim,
                "table": index.table_name,
                "lancedb_dir": str(settings.lancedb_dir),
            },
            indent=2,
        )
    )


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
        _build_encoder(settings, args),
        settings.rrf_k,
        settings.sparse_weight,
        corpus_limit=args.corpus_limit,
        use_dense=args.use_dense,
        retrieval_leg=args.retrieval_leg,
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


def _build_retriever(settings, args: argparse.Namespace) -> HybridRetriever:
    return HybridRetriever(
        settings.data_dir,
        settings.lancedb_dir,
        settings.table_name,
        _build_encoder(settings, args),
        settings.rrf_k,
        settings.sparse_weight,
        corpus_limit=args.corpus_limit,
        use_dense=args.use_dense,
        retrieval_leg=getattr(args, "retrieval_leg", "hybrid"),
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
    )


def cmd_play_stream(args: argparse.Namespace) -> None:
    settings = load_settings()
    streams = load_streams(settings.data_dir, args.domain)
    if args.stream_id:
        stream = next((s for s in streams if s.stream_id == args.stream_id), None)
        if stream is None:
            raise SystemExit(f"No stream with id {args.stream_id}")
    else:
        pool = [s for s in streams if s.category == args.category] or streams
        stream = pool[args.task_index % len(pool)]
    retriever = _build_retriever(settings, args)
    evidence_client = None
    generation_client = None
    if args.mode == "provider":
        try:
            evidence_client = deepseek_client(settings)
            generation_client = groq_client(settings)
        except ProviderError as exc:
            print(f"[provider unavailable: {exc}] falling back to deterministic")
    pipeline = StreamingRagPipeline(
        retriever,
        synthesis_mode=args.mode,
        evidence_client=evidence_client,
        generation_client=generation_client,
        enable_query_rewrite=args.rewrite_query,
        refine_on_final=True,
    )
    print(f"== stream {stream.stream_id} (category={stream.category}) ==")
    response = pipeline.run(list(stream.chunks), domain=args.domain)
    marker = {"Retrieve": "RETRIEVE", "Wait": "wait    ", "No-Retrieval": "no-retr "}
    for decision in response.decisions:
        label = marker.get(decision["decision"], decision["decision"])
        print(f"[{decision['timestamp_s']:>5.2f}s] {label}  {decision['text']!r}")
    print()
    print(f"answer: {response.answer}")
    print(f"citations: {response.citations}")
    print(f"uncertainty: {response.uncertainty}")
    if args.json:
        print(json.dumps(response.to_dict(), indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    settings = load_settings()
    tasks = load_query_tasks(settings.data_dir, args.domain)
    retriever = _build_retriever(settings, args)
    if args.mode == "retrieval":
        result = evaluate_retrieval(
            tasks,
            lambda query: [
                hit.passage.id
                for hit in retriever.search(query, args.domain, limit=args.top_k, candidate_limit=args.candidate_limit)
            ],
            limit=args.max_tasks,
        )
        result["domain"] = args.domain
        result["top_k"] = args.top_k
        print(json.dumps(result, indent=2))
        return

    if args.mode == "streaming":
        streams = load_streams(settings.data_dir, args.domain)
        pipeline = StreamingRagPipeline(retriever, synthesis_mode="deterministic")
        result = evaluate_streaming(
            streams,
            lambda chunks: pipeline.run(chunks, domain=args.domain),
            limit=args.max_tasks,
        )
        result["domain"] = args.domain
        print(json.dumps(result, indent=2))
        return

    try:
        evidence_client = deepseek_client(settings)
        generation_client = groq_client(settings)
    except ProviderError as exc:
        raise SystemExit(f"Provider eval requires configured providers: {exc}") from exc
    pipeline = StreamingRagPipeline(
        retriever,
        synthesis_mode="provider",
        evidence_client=evidence_client,
        generation_client=generation_client,
        enable_query_rewrite=args.rewrite_query,
        refine_on_final=True,
    )
    result = evaluate_provider(
        tasks,
        lambda query: pipeline.run(simulate_chunks(query), domain=args.domain),
        corpus_ids=corpus_id_set(settings.data_dir, args.domain),
        limit=args.max_tasks,
    )
    result["domain"] = args.domain
    print(json.dumps(result, indent=2))


def cmd_generate_streams(args: argparse.Namespace) -> None:
    settings = load_settings()
    tasks = load_query_tasks(settings.data_dir, args.domain)
    if not tasks:
        raise SystemExit(f"No query tasks found for domain: {args.domain}")
    streams = generate_streams(
        tasks,
        args.domain,
        seed=args.seed,
        multi_intent_count=args.multi_intent,
        no_retrieval_count=args.no_retrieval,
        revision_probability=args.revision_probability,
    )
    path = save_streams(settings.data_dir, args.domain, streams)
    print(json.dumps({"streams": len(streams), "path": str(path)}, indent=2))


def cmd_validate_streams(args: argparse.Namespace) -> None:
    settings = load_settings()
    streams = load_streams(settings.data_dir, args.domain)
    corpus_ids = corpus_id_set(settings.data_dir, args.domain)
    problems: list[str] = []
    by_category: dict[str, int] = {}
    for stream in streams:
        by_category[stream.category] = by_category.get(stream.category, 0) + 1
        if not stream.chunks:
            problems.append(f"{stream.stream_id}: no chunks")
            continue
        if not stream.chunks[-1].is_final:
            problems.append(f"{stream.stream_id}: last chunk is not final")
        timestamps = [chunk.timestamp_s for chunk in stream.chunks]
        if timestamps != sorted(timestamps):
            problems.append(f"{stream.stream_id}: non-monotonic timestamps")
        missing = [pid for pid in stream.qrel_passage_ids if pid not in corpus_ids]
        if missing:
            problems.append(f"{stream.stream_id}: {len(missing)} qrel ids missing from corpus")
    result = {
        "streams": len(streams),
        "categories": by_category,
        "problems": len(problems),
        "samples": problems[:10],
    }
    print(json.dumps(result, indent=2))
    if problems:
        raise SystemExit(1)


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
    index.add_argument("--embedding-backend", choices=["auto", "hash", "fastembed"], default=None)
    index.add_argument("--embedding-model", default=None)
    index.set_defaults(func=cmd_index)

    demo = sub.add_parser("run-demo")
    demo.add_argument("--domain", choices=["cloud", "govt"], default="cloud")
    demo.add_argument("--task-index", type=int, default=0)
    demo.add_argument("--corpus-limit", type=int, default=5000)
    demo.add_argument("--use-dense", action="store_true", help="Use LanceDB dense search; run `prism-rag index` first.")
    demo.add_argument("--retrieval-leg", choices=list(RETRIEVAL_LEGS), default="hybrid")
    demo.add_argument("--embedding-backend", choices=["auto", "hash", "fastembed"], default=None)
    demo.add_argument("--embedding-model", default=None)
    demo.add_argument("--mode", choices=["deterministic", "provider"], default="deterministic")
    demo.add_argument("--rewrite-query", action="store_true", help="Use DeepSeek to rewrite the live utterance before retrieval.")
    demo.add_argument(
        "--refine-on-final",
        action="store_true",
        help="Keep early retrieval, then refresh retrieval when the final transcript arrives.",
    )
    demo.set_defaults(func=cmd_run_demo)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--domain", choices=["cloud", "govt"], default="cloud")
    eval_parser.add_argument("--mode", choices=["retrieval", "provider", "streaming"], default="retrieval")
    eval_parser.add_argument("--max-tasks", type=int, default=10)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--candidate-limit", type=int, default=30)
    eval_parser.add_argument("--corpus-limit", type=int, default=None)
    eval_parser.add_argument("--use-dense", action="store_true", help="Use LanceDB dense search; run `prism-rag index` first.")
    eval_parser.add_argument("--retrieval-leg", choices=list(RETRIEVAL_LEGS), default="hybrid")
    eval_parser.add_argument("--embedding-backend", choices=["auto", "hash", "fastembed"], default=None)
    eval_parser.add_argument("--embedding-model", default=None)
    eval_parser.add_argument("--rewrite-query", action="store_true", help="Use DeepSeek query rewriting during provider eval.")
    eval_parser.set_defaults(func=cmd_eval)

    gen = sub.add_parser("generate-streams")
    gen.add_argument("--domain", choices=["cloud", "govt"], default="cloud")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--multi-intent", type=int, default=10)
    gen.add_argument("--no-retrieval", type=int, default=10)
    gen.add_argument("--revision-probability", type=float, default=0.2)
    gen.set_defaults(func=cmd_generate_streams)

    val_streams = sub.add_parser("validate-streams")
    val_streams.add_argument("--domain", choices=["cloud", "govt"], default="cloud")
    val_streams.set_defaults(func=cmd_validate_streams)

    play = sub.add_parser("play-stream")
    play.add_argument("--domain", choices=["cloud", "govt"], default="cloud")
    play.add_argument("--stream-id", default=None)
    play.add_argument("--category", choices=["early_retrieval", "multi_intent", "no_retrieval"], default="early_retrieval")
    play.add_argument("--task-index", type=int, default=0)
    play.add_argument("--corpus-limit", type=int, default=5000)
    play.add_argument("--use-dense", action="store_true")
    play.add_argument("--retrieval-leg", choices=list(RETRIEVAL_LEGS), default="hybrid")
    play.add_argument("--embedding-backend", choices=["auto", "hash", "fastembed"], default=None)
    play.add_argument("--embedding-model", default=None)
    play.add_argument("--mode", choices=["deterministic", "provider"], default="deterministic")
    play.add_argument("--rewrite-query", action="store_true")
    play.add_argument("--json", action="store_true")
    play.set_defaults(func=cmd_play_stream)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
