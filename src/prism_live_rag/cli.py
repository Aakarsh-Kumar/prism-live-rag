from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import load_settings
from .controller import simulate_chunks
from .curated_dataset import (
    SPLIT_TARGETS,
    build_manifest,
    attach_distractors,
    cases_path,
    choose_accepted_cases,
    curated_dir,
    export_review_csv,
    generate_candidate_pool,
    generate_curated_streams,
    import_reviews,
    load_cases,
    load_curated_streams,
    seed_benchmark_negatives,
    seed_benchmark_positives,
    shortlist_cases,
    select_source_passages,
    validate_dataset,
    validate_curated_streams,
    write_cases,
    write_curated_streams,
)
from .data import iter_passages, load_query_tasks, load_streams, save_streams, validate_domain
from .decomposer import RuleBasedDecomposer, build_decomposer
from .embeddings import DEVICE_CHOICES, build_encoder
from .evaluation import (
    corpus_id_set,
    decomposition_request_opportunities,
    evaluate_curated_streaming,
    evaluate_decomposition,
    evaluate_provider,
    evaluate_retrieval,
    evaluate_streaming,
)
from .pipeline import StreamingRagPipeline
from .providers import ProviderError, cerebras_client, provider_status
from .retrieval import RETRIEVAL_LEGS, HybridRetriever, LanceDbTimeout, LanceIndex
from .stream import generate_streams
from .telemetry import validate_trace_file


DEFAULT_TRACE_PATH = ".cache/telemetry/traces.jsonl"


def _build_encoder(settings, args: argparse.Namespace):
    return build_encoder(
        getattr(args, "embedding_backend", None) or settings.embedding_backend,
        model=getattr(args, "embedding_model", None) or settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
        hash_dim=settings.embedding_dim,
        device=getattr(args, "embedding_device", None) or settings.embedding_device,
        batch_size=getattr(args, "embedding_batch_size", None)
        or settings.embedding_batch_size,
        fixed_length=getattr(args, "embedding_fixed_length", None)
        or settings.embedding_fixed_length,
    )


def cmd_validate(args: argparse.Namespace) -> None:
    settings = load_settings()
    status = provider_status(settings)
    result = {
        "providers": {
            "groq_configured": status.groq_configured,
            "deepseek_configured": status.deepseek_configured,
            "cerebras_configured": status.cerebras_configured,
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
        count = index.build(
            settings.data_dir,
            args.domains,
            limit=args.limit,
            timeout_s=args.timeout_s,
            progress=True,
        )
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
                "device": encoder.device,
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
            evidence_client = generation_client = cerebras_client(settings)
        except ProviderError as exc:
            print(json.dumps({"warning": str(exc), "fallback": "deterministic"}, indent=2))

    # Build multi-intent decomposer if enabled
    decomposer = None
    if args.enable_multi_intent:
        # For LLM-based decomposer, we need a client
        if args.decomposer_type == "llm_based":
            if evidence_client is not None:
                decomposer = build_decomposer("llm_based", client=evidence_client)
            else:
                print(json.dumps({"warning": "LLM decomposer requires provider mode, falling back to rule-based"}, indent=2))
                decomposer = build_decomposer("rule_based")
        else:
            decomposer = build_decomposer("rule_based")

    pipeline = StreamingRagPipeline(
        retriever,
        decomposer=decomposer,
        synthesis_mode=args.mode,
        evidence_client=evidence_client,
        generation_client=generation_client,
        enable_query_rewrite=args.rewrite_query,
        refine_on_final=args.refine_on_final or args.mode == "provider",
        enable_multi_intent=args.enable_multi_intent,
    )
    try:
        response = _run_traced(pipeline, simulate_chunks(task.query), args.domain, args.trace_jsonl)
    except LanceDbTimeout as exc:
        raise SystemExit(
            f"{exc}. Retry without `--use-dense`, or run dense retrieval through Docker."
        ) from exc
    print(json.dumps(response.to_dict(), indent=2))


def _write_trace(path: str | None, payload: dict) -> None:
    trace = dict(payload)
    trace.setdefault("trace_id", uuid.uuid4().hex)
    trace.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat())
    trace.setdefault("status", "complete")
    trace.setdefault("domain", "")
    trace.setdefault("retrieval_events", [])
    trace.setdefault("decisions", [])
    trace.setdefault("citations", [])
    trace.setdefault("answer", "")
    trace.setdefault("version", 1)
    trace.setdefault("previous_version", None)
    trace.setdefault("applied_delta", None)
    trace.setdefault("stage_latency_ms", {})
    trace.setdefault("token_usage", {})
    trace.setdefault("estimated_cost_usd", None)
    trace.setdefault("cost_estimate_basis", "not captured")
    target = Path(path or os.getenv("TRACE_JSONL_PATH") or DEFAULT_TRACE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")


def _run_traced(pipeline, chunks, domain: str, trace_path: str | None = None, **kwargs):
    """Run one stream and persist exactly one trace, including failed attempts."""
    try:
        response = pipeline.run(chunks, domain=domain, trace=True, **kwargs)
    except Exception:
        _write_trace(trace_path, {
            "status": "error", "domain": domain, "retrieval_events": [],
            "decisions": [], "citations": [], "answer": "",
            "version": 1, "previous_version": None, "applied_delta": None,
            "stage_latency_ms": {}, "token_usage": {},
            "estimated_cost_usd": None, "cost_estimate_basis": "request failed before usage was available",
        })
        raise
    payload = response.to_dict()
    payload.update({"domain": domain, "status": "complete"})
    _write_trace(trace_path, payload)
    return response


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
        use_reranker=getattr(args, "use_reranker", False),
        reranker_model=getattr(args, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
        reranker_top_k=getattr(args, "reranker_top_k", 50),
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
            evidence_client = generation_client = cerebras_client(settings)
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
    response = _run_traced(pipeline, iter(stream.chunks), args.domain, args.trace_jsonl)
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
            lambda chunks: _run_traced(pipeline, chunks, args.domain, args.trace_jsonl),
            limit=args.max_tasks,
        )
        result["domain"] = args.domain
        print(json.dumps(result, indent=2))
        return

    try:
        evidence_client = generation_client = cerebras_client(settings)
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
        lambda query: _run_traced(pipeline, simulate_chunks(query), args.domain, args.trace_jsonl),
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


def _curated_passages(settings) -> tuple[dict[str, list], dict[str, object]]:
    by_domain = {
        domain: list(iter_passages(settings.data_dir, domain))
        for domain in ("cloud", "govt")
    }
    by_id = {passage.id: passage for passages in by_domain.values() for passage in passages}
    return by_domain, by_id


def cmd_curated_generate(args: argparse.Namespace) -> None:
    settings = load_settings()
    if (args.case_class is None) != (args.count is None):
        raise SystemExit("--case-class and --count must be supplied together")
    if args.count is not None and args.count <= 0:
        raise SystemExit("--count must be positive")
    try:
        client = cerebras_client(settings)
    except ProviderError as exc:
        raise SystemExit(f"Curated generation requires Cerebras: {exc}") from exc
    passages_by_domain, _ = _curated_passages(settings)
    if args.domain:
        passages_by_domain = {args.domain: passages_by_domain[args.domain]}
    targets = {args.case_class: args.count} if args.case_class else None
    requested_total = sum(targets.values()) if targets else int(
        sum(SPLIT_TARGETS[args.split].values()) * args.multiplier
    )
    needed = max(10, requested_total)
    selected = {
        domain: select_source_passages(
            passages,
            args.split,
            seed=args.seed,
            limit=min(args.source_limit or needed, needed),
        )
        for domain, passages in passages_by_domain.items()
    }
    candidates = generate_candidate_pool(
        client,
        selected,
        split=args.split,
        seed=args.seed,
        multiplier=args.multiplier,
        targets=targets,
        progress=print,
    )
    output = Path(args.output) if args.output else curated_dir(settings.data_dir) / "candidates" / f"{args.split}.jsonl"
    write_cases(output, candidates)
    print(json.dumps({"candidates": len(candidates), "path": str(output)}, indent=2))


def cmd_curated_export_review(args: argparse.Namespace) -> None:
    settings = load_settings()
    source = Path(args.input)
    output = Path(args.output) if args.output else curated_dir(settings.data_dir) / "reviews" / f"{source.stem}.csv"
    cases = load_cases(source)
    export_review_csv(cases, output)
    print(json.dumps({"cases": len(cases), "path": str(output)}, indent=2))


def cmd_curated_seed_benchmark(args: argparse.Namespace) -> None:
    settings = load_settings()
    reference = settings.data_dir / "mtragun-human" / "generation_tasks" / "reference.jsonl"
    if args.kind == "negative":
        cases = seed_benchmark_negatives(reference, split=args.split, seed=args.seed)
    else:
        _, passages = _curated_passages(settings)
        cases = seed_benchmark_positives(
            reference, passages, split=args.split, seed=args.seed
        )
    output = Path(args.output) if args.output else curated_dir(settings.data_dir) / "candidates" / f"{args.split}-mtragun-{args.kind}.jsonl"
    write_cases(output, cases)
    print(json.dumps({"cases": len(cases), "path": str(output)}, indent=2))


def cmd_curated_distractors(args: argparse.Namespace) -> None:
    settings = load_settings()
    retriever = _build_retriever(settings, args)
    cases = load_cases(Path(args.input))
    attach_distractors(
        cases,
        lambda query, domain, limit: [
            hit.passage.id for hit in retriever.search(query, domain, limit=limit)
        ],
        limit=args.limit,
    )
    output = Path(args.output)
    write_cases(output, cases)
    print(json.dumps({"cases": len(cases), "path": str(output)}, indent=2))


def cmd_curated_import_review(args: argparse.Namespace) -> None:
    candidates = load_cases(Path(args.input))
    reviewed = import_reviews(candidates, Path(args.review))
    output = Path(args.output)
    write_cases(output, reviewed)
    print(json.dumps({"cases": len(reviewed), "path": str(output)}, indent=2))


def cmd_curated_merge(args: argparse.Namespace) -> None:
    merged = []
    seen: set[str] = set()
    for source_name in args.inputs:
        for case in load_cases(Path(source_name)):
            if case.case_id in seen:
                raise SystemExit(f"Duplicate case_id while merging: {case.case_id}")
            seen.add(case.case_id)
            merged.append(case)
    output = Path(args.output)
    write_cases(output, sorted(merged, key=lambda case: case.case_id))
    print(json.dumps({"cases": len(merged), "path": str(output)}, indent=2))


def cmd_curated_finalize(args: argparse.Namespace) -> None:
    settings = load_settings()
    reviewed = load_cases(Path(args.input))
    accepted = choose_accepted_cases(reviewed, args.split, seed=args.seed)
    output = cases_path(settings.data_dir, args.split)
    write_cases(output, accepted)
    print(json.dumps({"accepted": len(accepted), "path": str(output)}, indent=2))


def cmd_curated_shortlist(args: argparse.Namespace) -> None:
    cases = load_cases(Path(args.input))
    shortlisted, audit_rows = shortlist_cases(cases, args.split)
    output = Path(args.output)
    write_cases(output, shortlisted)
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"shortlisted": len(shortlisted), "path": str(output), "audit": str(audit_output)},
            indent=2,
        )
    )


def _validate_curated(settings, require_targets: bool = True) -> dict:
    _, passages = _curated_passages(settings)
    cases_by_split = {
        split: load_cases(cases_path(settings.data_dir, split))
        for split in SPLIT_TARGETS
        if cases_path(settings.data_dir, split).exists()
    }
    if not cases_by_split:
        raise SystemExit("No accepted curated split files found")
    return validate_dataset(cases_by_split, passages, require_targets=require_targets)


def cmd_curated_validate(args: argparse.Namespace) -> None:
    settings = load_settings()
    report = _validate_curated(settings, require_targets=not args.allow_incomplete)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def cmd_curated_validate_candidates(args: argparse.Namespace) -> None:
    settings = load_settings()
    _, passages = _curated_passages(settings)
    cases = load_cases(Path(args.input))
    by_split: dict[str, list] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)
    report = validate_dataset(
        by_split,
        passages,
        require_targets=False,
        require_test_review=False,
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def cmd_curated_generate_streams(args: argparse.Namespace) -> None:
    settings = load_settings()
    cases = load_cases(Path(args.input))
    streams = generate_curated_streams(
        cases,
        seed=args.seed,
        revision_probability=args.revision_probability,
    )
    output = (
        Path(args.output)
        if args.output
        else curated_dir(settings.data_dir) / "streams" / f"{args.split}.jsonl"
    )
    write_curated_streams(output, streams)
    report = validate_curated_streams(cases, streams)
    print(json.dumps({**report, "path": str(output)}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def cmd_curated_validate_streams(args: argparse.Namespace) -> None:
    cases = load_cases(Path(args.cases))
    streams = load_curated_streams(Path(args.streams))
    report = validate_curated_streams(cases, streams)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def cmd_curated_validate_traces(args: argparse.Namespace) -> None:
    report = validate_trace_file(args.input, expected=args.expected)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def cmd_curated_evaluate_streams(args: argparse.Namespace) -> None:
    settings = load_settings()
    cases = load_cases(Path(args.cases))
    streams = load_curated_streams(Path(args.streams))
    validation = validate_curated_streams(cases, streams)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, indent=2))
    retriever = _build_retriever(settings, args)
    pipeline = StreamingRagPipeline(
        retriever,
        synthesis_mode="deterministic",
        enable_multi_intent=True,
    )
    result = evaluate_curated_streaming(
        streams,
        lambda stream: _run_traced(
            pipeline,
            iter(stream.chunks),
            stream.domain,
            args.trace_jsonl,
            context_turns=stream.context_turns,
        ),
        limit=args.max_tasks,
    )
    result["review_status"] = "pending_human_review" if any(
        case.review.get("status") != "approved" for case in cases
    ) else "approved"
    result["evaluation_config"] = {
        "corpus_limit": args.corpus_limit,
        "use_dense": args.use_dense,
        "retrieval_leg": args.retrieval_leg,
        "use_reranker": args.use_reranker,
        "embedding_backend": args.embedding_backend or settings.embedding_backend,
        "embedding_device": args.embedding_device or settings.embedding_device,
        "max_tasks": args.max_tasks,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({"path": str(output), **{k: v for k, v in result.items() if k != "failures"}}, indent=2))
    else:
        print(rendered)


def cmd_curated_evaluate_decomposition(args: argparse.Namespace) -> None:
    streams = []
    for stream_path in args.streams or []:
        streams.extend(load_curated_streams(Path(stream_path)))
    for cases_path in args.cases or []:
        streams.extend(
            json.loads(line)
            for line in Path(cases_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    single_queries = []
    if args.single_queries:
        single_queries = [
            json.loads(line)
            for line in Path(args.single_queries).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not streams and not single_queries:
        raise SystemExit("Supply at least one of --cases, --streams, or --single-queries.")
    if any(not isinstance(case.get("query"), str) for case in single_queries):
        raise SystemExit("Each single-query evaluation row must contain a string 'query'.")
    checkpoint_path = Path(args.checkpoint or f"{args.output}.checkpoint.jsonl")
    settings = load_settings() if args.decomposer in {"llm_based", "both"} else None
    checkpoint_metadata = {
        "schema_version": 3,
        "decomposer": args.decomposer,
        "model": settings.cerebras_model if settings is not None else None,
        "provider_limit": args.provider_limit,
        "max_tasks": args.max_tasks,
        "retry_fallbacks": args.retry_fallbacks,
        "wait_for_circuit_recovery": True,
        "input_hashes": {},
    }
    for source in [*(args.streams or []), *(args.cases or [])] + ([args.single_queries] if args.single_queries else []):
        source_path = Path(source)
        checkpoint_metadata["input_hashes"][str(source_path)] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    cached_rows: dict[str, dict] = {}
    if args.decomposer in {"llm_based", "both"} and checkpoint_path.exists():
        if not args.resume:
            raise SystemExit(
                f"Checkpoint already exists at {checkpoint_path}; pass --resume to continue it, "
                "or choose a new --checkpoint path. Existing checkpoint was left unchanged."
            )
        checkpoint_lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
        checkpoint_header = None
        for line_number, line in enumerate(checkpoint_lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted final write is ignored; complete preceding rows remain usable.
                if line_number == len(checkpoint_lines):
                    continue
                raise SystemExit(f"Invalid checkpoint JSON at {checkpoint_path}:{line_number}")
            if isinstance(row, dict) and "_checkpoint_metadata" in row:
                checkpoint_header = row["_checkpoint_metadata"]
                continue
            if isinstance(row, dict) and row.get("case_id"):
                cached_rows[str(row["case_id"])] = row
        if checkpoint_header != checkpoint_metadata:
            legacy_keys = {"decomposer", "model", "provider_limit", "max_tasks", "input_hashes"}
            legacy_matches = (
                checkpoint_header
                and checkpoint_header.get("schema_version") in {1, 2}
                and {key: checkpoint_header.get(key) for key in legacy_keys}
                == {key: checkpoint_metadata[key] for key in legacy_keys}
            )
            if legacy_matches and args.decomposer == "llm_based":
                # G3 trace v1 inverted success/fallback labels. Recover from the
                # stable reason strings in each already-saved case without API calls.
                if checkpoint_header.get("schema_version") == 1:
                    for row in cached_rows.values():
                        for observation in row.get("retrieval_observations", []):
                            reason = str(observation.get("reason", ""))
                            failed = reason.startswith(("LLM failed", "Circuit breaker active"))
                            observation["provider_status"] = "fallback" if failed else "success"
                            observation["provider_error"] = reason if failed else None
                        reason = str(row.get("decomposition_reason", ""))
                        failed = reason.startswith(("LLM failed", "Circuit breaker active"))
                        row["final_provider_status"] = "fallback" if failed else "success"
                        row["final_provider_error"] = reason if failed else None
            else:
                raise SystemExit(
                    "Checkpoint inputs/settings do not match this run; choose a new --checkpoint path."
                )
    strategies = {"rule_based": RuleBasedDecomposer()}
    provider_sampled = False
    llm_client = None
    if args.decomposer in {"llm_based", "both"}:
        if args.decomposer == "llm_based":
            strategies = {}
        provider_cap = args.provider_limit
        if provider_cap < 1:
            raise SystemExit("--provider-limit must be a positive request budget.")
        try:
            llm_client = cerebras_client(settings)
            strategies["llm_based"] = build_decomposer(
                "llm_based", client=llm_client, wait_for_circuit_recovery=True
            )
        except ProviderError as exc:
            raise SystemExit(f"LLM decomposition evaluation requires Cerebras: {exc}") from exc
        candidates = streams[:args.max_tasks] if args.max_tasks is not None else streams
        category_of = lambda case: (
            case.get("category", "") if isinstance(case, dict) else getattr(case, "category", "")
        )
        compounds = [case for case in candidates if category_of(case) in {"compound", "multi_intent"}]
        singles = [case for case in candidates if category_of(case) == "single_intent"]
        # Select whole streams by their actual number of distinct Retrieve checkpoints,
        # balancing compound/single cases without assuming a fixed calls-per-stream.
        ordered = []
        for index in range(max(len(compounds), len(singles))):
            if index < len(compounds):
                ordered.append(compounds[index])
            if index < len(singles):
                ordered.append(singles[index])
        if not ordered:
            ordered = candidates
        provider_cases = []
        estimated_provider_calls = 0
        for case in ordered:
            opportunities = decomposition_request_opportunities(case)
            if opportunities and estimated_provider_calls + opportunities <= provider_cap:
                provider_cases.append(case)
                estimated_provider_calls += opportunities
        streams_for_provider = provider_cases
        provider_sampled = True
    else:
        streams_for_provider = streams

    reports = {}
    if args.decomposer == "both":
        reports["rule_based_sampled"] = evaluate_decomposition(
            streams_for_provider, RuleBasedDecomposer()
        )
    for strategy_name, decomposer in strategies.items():
        selected_streams = streams_for_provider if strategy_name == "llm_based" else streams
        selected_singles = [] if args.cases else single_queries
        selected_limit = None if strategy_name == "llm_based" else args.max_tasks
        checkpoint_handle = None
        on_case = None
        resume_for_strategy = None
        if strategy_name == "llm_based":
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint_path.exists():
                with checkpoint_path.open("x", encoding="utf-8") as header_handle:
                    header_handle.write(json.dumps({"_checkpoint_metadata": checkpoint_metadata}) + "\n")
            checkpoint_handle = checkpoint_path.open("a", encoding="utf-8")
            resume_for_strategy = cached_rows
            updated_counter = [0]

            def save_checkpoint(row, handle=checkpoint_handle):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                updated_counter[0] += 1
                print(json.dumps({
                    "checkpoint": str(checkpoint_path),
                    "updated_case_id": row["case_id"],
                    "updated_cases_this_run": updated_counter[0],
                    "provider_request_attempts": llm_client.request_attempts,
                }), flush=True)

            on_case = save_checkpoint
        report = evaluate_decomposition(
            selected_streams,
            decomposer,
            single_queries=selected_singles,
            limit=selected_limit,
            resume_rows=resume_for_strategy,
            on_case=on_case,
            retry_fallbacks=args.retry_fallbacks,
        )
        if checkpoint_handle is not None:
            checkpoint_handle.close()
        report["case_files"] = args.cases or []
        report["stream_files"] = args.streams or []
        report["single_queries_file"] = args.single_queries
        if strategy_name == "llm_based" and llm_client is not None:
            settings = load_settings()
            report["provider"] = "cerebras"
            report["model"] = settings.cerebras_model
            report["provider_token_usage"] = (
                dict(llm_client.token_usage) if report["resumed_cases"] == 0 else None
            )
            report["provider_token_usage_complete"] = report["resumed_cases"] == 0
            report["provider_token_usage_note"] = (
                "Exact for this run." if report["resumed_cases"] == 0
                else "Not persisted per case; full-run token usage cannot be reconstructed after resume."
            )
            report["provider_request_attempts"] = (
                llm_client.request_attempts if report["resumed_cases"] == 0 else None
            )
            report["provider_request_attempts_scope"] = (
                "Exact for this invocation." if report["resumed_cases"] == 0
                else "Unknown for the full run after resume; this invocation's count is separate."
            )
            report["provider_request_attempts_this_invocation"] = llm_client.request_attempts
            report["provider_request_budget"] = args.provider_limit
            report["estimated_selected_request_opportunities"] = estimated_provider_calls
            report["checkpoint"] = str(checkpoint_path)
            report["resume_enabled"] = bool(args.resume)
            report["retry_fallbacks_enabled"] = bool(args.retry_fallbacks)
        reports[strategy_name] = report
    result = reports.get("rule_based") or reports.get("llm_based")
    if args.decomposer == "both":
        result = {
            "mode": "decomposition_comparison",
            "provider_sampled": provider_sampled,
            "provider_request_limit": args.provider_limit,
            "provider_request_rate_limit_per_minute": 5,
            "estimated_selected_request_opportunities": estimated_provider_calls,
            "strategies": reports,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "path": str(output),
                **{key: value for key, value in result.items() if key not in {"failures", "cases", "strategies"}},
                "failure_count": len(result.get("failures", [])),
                "strategies": {
                    name: {"compound_success_rate": report["compound_success_rate"],
                           "single_overfragmentation_rate": report["single_overfragmentation_rate"],
                           "failure_count": len(report["failures"])}
                    for name, report in result.get("strategies", {}).items()
                },
            },
            indent=2,
        )
    )


def cmd_curated_manifest(args: argparse.Namespace) -> None:
    settings = load_settings()
    report = _validate_curated(settings, require_targets=True)
    if not report["valid"]:
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    manifest = build_manifest(settings.data_dir, report, seed=args.seed)
    output = curated_dir(settings.data_dir) / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


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
    index.add_argument("--embedding-device", choices=list(DEVICE_CHOICES), default=None)
    index.add_argument("--embedding-batch-size", type=int, default=None)
    index.add_argument("--embedding-fixed-length", type=int, default=None, help="Fixed sequence length for GPU memory stability (default: 512)")
    index.set_defaults(func=cmd_index)

    demo = sub.add_parser("run-demo")
    demo.add_argument("--domain", choices=["cloud", "govt"], default="cloud")
    demo.add_argument("--task-index", type=int, default=0)
    demo.add_argument("--corpus-limit", type=int, default=5000)
    demo.add_argument("--use-dense", action="store_true", help="Use LanceDB dense search; run `prism-rag index` first.")
    demo.add_argument("--retrieval-leg", choices=list(RETRIEVAL_LEGS), default="hybrid")
    demo.add_argument("--embedding-backend", choices=["auto", "hash", "fastembed"], default=None)
    demo.add_argument("--embedding-model", default=None)
    demo.add_argument("--embedding-device", choices=list(DEVICE_CHOICES), default=None)
    demo.add_argument("--embedding-batch-size", type=int, default=None)
    demo.add_argument("--mode", choices=["deterministic", "provider"], default="deterministic")
    demo.add_argument("--rewrite-query", action="store_true", help="Use DeepSeek to rewrite the live utterance before retrieval.")
    demo.add_argument("--enable-multi-intent", action="store_true", help="Enable multi-intent query decomposition and parallel retrieval.")
    demo.add_argument("--decomposer-type", choices=["rule_based", "llm_based"], default="rule_based", help="Type of multi-intent decomposer to use.")
    demo.add_argument(
        "--refine-on-final",
        action="store_true",
        help="Keep early retrieval, then refresh retrieval when the final transcript arrives.",
    )
    demo.add_argument("--use-reranker", action="store_true", help="Enable cross-encoder reranking for improved retrieval quality.")
    demo.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Cross-encoder model for reranking.")
    demo.add_argument("--reranker-top-k", type=int, default=50, help="Number of candidates to retrieve before reranking.")
    demo.add_argument("--trace-jsonl", default=None, help="Append one structured execution trace per run to this JSONL file.")
    demo.set_defaults(func=cmd_run_demo)

    serve = sub.add_parser("serve", help="Serve the interactive streaming RAG dashboard.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(func=cmd_serve)

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
    eval_parser.add_argument("--embedding-device", choices=list(DEVICE_CHOICES), default=None)
    eval_parser.add_argument("--embedding-batch-size", type=int, default=None)
    eval_parser.add_argument("--rewrite-query", action="store_true", help="Use DeepSeek query rewriting during provider eval.")
    eval_parser.add_argument("--use-reranker", action="store_true", help="Enable cross-encoder reranking for improved retrieval quality.")
    eval_parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Cross-encoder model for reranking.")
    eval_parser.add_argument("--reranker-top-k", type=int, default=50, help="Number of candidates to retrieve before reranking.")
    eval_parser.add_argument("--trace-jsonl", default=None, help="Trace JSONL destination (default: TRACE_JSONL_PATH or .cache/telemetry/traces.jsonl).")
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

    curated = sub.add_parser("curated-data", help="Generate, review, freeze, and validate the curated grounding suite.")
    curated_sub = curated.add_subparsers(dest="curated_command", required=True)

    curated_gen = curated_sub.add_parser("generate", help="Generate an overcomplete candidate pool with DeepSeek.")
    curated_gen.add_argument("--split", choices=list(SPLIT_TARGETS), required=True)
    curated_gen.add_argument("--domain", choices=["cloud", "govt"], default=None)
    curated_gen.add_argument("--seed", type=int, default=20260928)
    curated_gen.add_argument("--multiplier", type=float, default=2.0)
    curated_gen.add_argument("--case-class", choices=["answerable", "partial", "unanswerable", "underspecified"], default=None)
    curated_gen.add_argument("--count", type=int, default=None, help="Exact candidate count; requires --case-class.")
    curated_gen.add_argument("--source-limit", type=int, default=None)
    curated_gen.add_argument("--output", default=None)
    curated_gen.set_defaults(func=cmd_curated_generate)

    curated_seed = curated_sub.add_parser("seed-benchmark", help="Seed authoritative MTRAG-UN test negatives.")
    curated_seed.add_argument("--split", choices=list(SPLIT_TARGETS), default="test")
    curated_seed.add_argument("--kind", choices=["positive", "negative"], default="negative")
    curated_seed.add_argument("--seed", type=int, default=20260928)
    curated_seed.add_argument("--output", default=None)
    curated_seed.set_defaults(func=cmd_curated_seed_benchmark)

    curated_distractors = curated_sub.add_parser("add-distractors", help="Attach top-ranked non-gold passages.")
    curated_distractors.add_argument("--input", required=True)
    curated_distractors.add_argument("--output", required=True)
    curated_distractors.add_argument("--limit", type=int, default=5)
    curated_distractors.add_argument("--corpus-limit", type=int, default=None)
    curated_distractors.add_argument("--use-dense", action="store_true")
    curated_distractors.add_argument("--retrieval-leg", choices=list(RETRIEVAL_LEGS), default="hybrid")
    curated_distractors.add_argument("--embedding-backend", choices=["auto", "hash", "fastembed"], default=None)
    curated_distractors.add_argument("--embedding-model", default=None)
    curated_distractors.add_argument("--embedding-device", choices=list(DEVICE_CHOICES), default=None)
    curated_distractors.add_argument("--embedding-batch-size", type=int, default=None)
    curated_distractors.add_argument("--use-reranker", action="store_true")
    curated_distractors.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    curated_distractors.add_argument("--reranker-top-k", type=int, default=50)
    curated_distractors.set_defaults(func=cmd_curated_distractors)

    curated_export = curated_sub.add_parser("export-review", help="Export candidates to a human-review CSV.")
    curated_export.add_argument("--input", required=True)
    curated_export.add_argument("--output", default=None)
    curated_export.set_defaults(func=cmd_curated_export_review)

    curated_import = curated_sub.add_parser("import-review", help="Merge completed human-review decisions.")
    curated_import.add_argument("--input", required=True)
    curated_import.add_argument("--review", required=True)
    curated_import.add_argument("--output", required=True)
    curated_import.set_defaults(func=cmd_curated_import_review)

    curated_merge = curated_sub.add_parser("merge", help="Merge candidate or reviewed JSONL files.")
    curated_merge.add_argument("--inputs", nargs="+", required=True)
    curated_merge.add_argument("--output", required=True)
    curated_merge.set_defaults(func=cmd_curated_merge)

    curated_finalize = curated_sub.add_parser("finalize", help="Select approved cases and write an accepted split.")
    curated_finalize.add_argument("--split", choices=list(SPLIT_TARGETS), required=True)
    curated_finalize.add_argument("--input", required=True)
    curated_finalize.add_argument("--seed", type=int, default=20260928)
    curated_finalize.set_defaults(func=cmd_curated_finalize)

    curated_shortlist = curated_sub.add_parser("shortlist", help="Rank candidates into a pending-review target-sized shortlist.")
    curated_shortlist.add_argument("--split", choices=list(SPLIT_TARGETS), required=True)
    curated_shortlist.add_argument("--input", required=True)
    curated_shortlist.add_argument("--output", required=True)
    curated_shortlist.add_argument("--audit-output", required=True)
    curated_shortlist.set_defaults(func=cmd_curated_shortlist)

    curated_validate = curated_sub.add_parser("validate", help="Validate accepted splits and leakage invariants.")
    curated_validate.add_argument("--allow-incomplete", action="store_true")
    curated_validate.set_defaults(func=cmd_curated_validate)

    curated_validate_candidates = curated_sub.add_parser("validate-candidates", help="Validate candidate schemas before human review.")
    curated_validate_candidates.add_argument("--input", required=True)
    curated_validate_candidates.set_defaults(func=cmd_curated_validate_candidates)

    curated_streams = curated_sub.add_parser("generate-streams", help="Convert current-turn queries into timestamped ASR partial streams.")
    curated_streams.add_argument("--split", choices=list(SPLIT_TARGETS), required=True)
    curated_streams.add_argument("--input", required=True)
    curated_streams.add_argument("--output", default=None)
    curated_streams.add_argument("--seed", type=int, default=20260928)
    curated_streams.add_argument("--revision-probability", type=float, default=0.35)
    curated_streams.set_defaults(func=cmd_curated_generate_streams)

    curated_validate_streams = curated_sub.add_parser("validate-streams", help="Validate ASR timing, revisions, final supersession, and case joins.")
    curated_validate_streams.add_argument("--cases", required=True)
    curated_validate_streams.add_argument("--streams", required=True)
    curated_validate_streams.set_defaults(func=cmd_curated_validate_streams)

    curated_eval_streams = curated_sub.add_parser(
        "evaluate-streams",
        help="Run the deterministic pipeline over interval-delivered curated ASR streams.",
    )
    curated_eval_streams.add_argument("--cases", required=True)
    curated_eval_streams.add_argument("--streams", required=True)
    curated_eval_streams.add_argument("--max-tasks", type=int, default=None)
    curated_eval_streams.add_argument("--output", default=None)
    curated_eval_streams.add_argument("--corpus-limit", type=int, default=None)
    curated_eval_streams.add_argument("--use-dense", action="store_true")
    curated_eval_streams.add_argument(
        "--retrieval-leg", choices=list(RETRIEVAL_LEGS), default="hybrid"
    )
    curated_eval_streams.add_argument(
        "--embedding-backend", choices=["auto", "hash", "fastembed"], default=None
    )
    curated_eval_streams.add_argument("--embedding-model", default=None)
    curated_eval_streams.add_argument(
        "--embedding-device", choices=list(DEVICE_CHOICES), default=None
    )
    curated_eval_streams.add_argument("--embedding-batch-size", type=int, default=None)
    curated_eval_streams.add_argument("--use-reranker", action="store_true")
    curated_eval_streams.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2"
    )
    curated_eval_streams.add_argument("--reranker-top-k", type=int, default=50)
    curated_eval_streams.add_argument("--trace-jsonl", default=None, help="Trace JSONL destination (default: TRACE_JSONL_PATH or .cache/telemetry/traces.jsonl).")
    curated_eval_streams.set_defaults(func=cmd_curated_evaluate_streams)

    curated_eval_decomposition = curated_sub.add_parser(
        "evaluate-decomposition",
        help="Measure compound-intent recovery and single-query over-fragmentation.",
    )
    curated_eval_decomposition.add_argument("--streams", nargs="+", default=None)
    curated_eval_decomposition.add_argument("--cases", nargs="+", default=None)
    curated_eval_decomposition.add_argument("--single-queries", default=None)
    curated_eval_decomposition.add_argument(
        "--decomposer", choices=["rule_based", "llm_based", "both"], default="rule_based"
    )
    curated_eval_decomposition.add_argument(
        "--provider-limit", type=int, default=10,
        help="Maximum distinct streamed decomposition calls selected; Cerebras paced at 5 RPM.",
    )
    curated_eval_decomposition.add_argument("--max-tasks", type=int, default=None)
    curated_eval_decomposition.add_argument(
        "--checkpoint", default=None,
        help="Per-case JSONL checkpoint (default: <output>.checkpoint.jsonl).",
    )
    curated_eval_decomposition.add_argument(
        "--resume", action="store_true",
        help="Resume completed cases from the checkpoint; requires the same input and selection settings.",
    )
    curated_eval_decomposition.add_argument(
        "--retry-fallbacks", action="store_true",
        help="On resume, retry only cached provider fallback checkpoints; successful outputs are reused.",
    )
    curated_eval_decomposition.add_argument(
        "--output",
        default="data/curated_dataset/evaluation/decomposition-baseline.json",
    )
    curated_eval_decomposition.set_defaults(func=cmd_curated_evaluate_decomposition)

    curated_validate_traces = curated_sub.add_parser(
        "validate-traces", help="Audit G6 JSONL trace schema and execution coverage."
    )
    curated_validate_traces.add_argument("--input", default=DEFAULT_TRACE_PATH)
    curated_validate_traces.add_argument("--expected", type=int, default=None)
    curated_validate_traces.set_defaults(func=cmd_curated_validate_traces)

    curated_manifest = curated_sub.add_parser("manifest", help="Validate and freeze hashes in manifest.json.")
    curated_manifest.add_argument("--seed", type=int, default=20260928)
    curated_manifest.set_defaults(func=cmd_curated_manifest)

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
    play.add_argument("--embedding-device", choices=list(DEVICE_CHOICES), default=None)
    play.add_argument("--embedding-batch-size", type=int, default=None)
    play.add_argument("--mode", choices=["deterministic", "provider"], default="deterministic")
    play.add_argument("--rewrite-query", action="store_true")
    play.add_argument("--json", action="store_true")
    play.add_argument("--trace-jsonl", default=None, help="Append one structured execution trace per run to this JSONL file.")
    play.set_defaults(func=cmd_play_stream)
    return parser


def cmd_serve(args: argparse.Namespace) -> None:
    from .serve import serve_dashboard

    serve_dashboard(host=args.host, port=args.port)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
