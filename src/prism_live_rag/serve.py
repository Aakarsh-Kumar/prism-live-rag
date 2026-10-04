"""Dependency-free local dashboard server for the streaming RAG demonstration."""

from __future__ import annotations

import json
import hashlib
import mimetypes
import queue
import threading
import time
import uuid
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

from .config import load_settings
from .data import iter_passages
from .decomposer import build_decomposer
from .embeddings import build_encoder
from .models import RagResponse, ConversationSession
from .controller import TranscriptChunk
from .pipeline import StreamingRagPipeline
from .providers import ProviderError, cerebras_client
from .retrieval import HybridRetriever, LanceIndex
from .stream import SimulatedStream
from .telemetry import validate_trace_file

ASSET_DIR = Path(__file__).with_name("webui")


def runtime_code_hash() -> str:
    directory = Path(__file__).parent
    return hashlib.sha256(b"".join((directory / name).read_bytes() for name in (
        "pipeline.py", "decomposer.py", "synthesis.py", "llm_steps.py", "serve.py",
        "providers.py", "controller.py", "retrieval.py",
    ))).hexdigest()


class RunState:
    def __init__(self, run_id: str, scenario: dict, options: dict) -> None:
        self.run_id = run_id
        self.scenario = scenario
        self.options = options
        self.events: list[dict] = []
        self.status = "queued"
        self.error: str | None = None
        self.response: dict | None = None
        self.session: ConversationSession | None = None
        self.refinement = False
        self.created_at = time.monotonic()
        self.condition = threading.Condition()

    def publish(self, event: dict) -> None:
        with self.condition:
            value = {"id": len(self.events) + 1,
                     "observed_elapsed_s": round(time.monotonic() - self.created_at, 6), **event}
            self.events.append(value)
            self.condition.notify_all()

    def snapshot(self) -> dict:
        with self.condition:
            return {
                "run_id": self.run_id,
                "status": self.status,
                "scenario": self.scenario["id"],
                "options": self.options,
                "events": list(self.events),
                "response": self.response,
                "error": self.error,
            }


def evaluate_dashboard_run(stream: SimulatedStream, response: RagResponse, *, is_test: bool) -> dict:
    """Expose label/ID comparisons as proxies, never as semantic faithfulness."""
    expected = stream.expected_behavior
    citations = set(response.citations)
    gold_ids = set(stream.qrel_passage_ids)
    qrel_hit = bool(citations & gold_ids)
    abstained = not citations and response.uncertainty is not None
    clarified = not citations and response.answer.rstrip().endswith("?")
    if expected in {"answer", "partial_answer"}:
        behavior_match = None
    elif expected == "abstain":
        behavior_match = abstained
    elif expected == "clarify":
        behavior_match = clarified
    else:
        behavior_match = None
    return {
        "is_evaluated_test": is_test,
        "case_class": stream.case_class or None,
        "expected_behavior": expected or None,
        "qrel_citation_overlap_proxy": qrel_hit if gold_ids else None,
        "actual_behavior_match_proxy": behavior_match,
        "note": (
            "Gold passage ID overlap is retrieval/citation evidence only; it does not "
            "score answer correctness or semantic support. Interactive-run faithfulness "
            "was not evaluated."
        ),
    }


def scenario_uses_decomposition(scenario: dict, options: dict) -> bool:
    """Keep decomposition on by default and mandatory for labeled demo compounds."""
    return bool(options.get("multi_intent", True)) or scenario.get("category") == "multi_intent"


class ObservedChatClient:
    """Decorate a provider client to report actual request attempts and outcomes."""

    def observe_evidence_selection(self, summary: dict) -> None:
        self._emit({"type": "evidence_selection", **summary})

    def observe_relevance_check(self, query: str) -> None:
        self._emit({"type": "llm_routing", "stage": "evidence_relevance",
                    "eligible": True, "decision": "call", "query": query,
                    "reason": "Check that selected exact evidence answers the requested subject and relation."})

    def __init__(self, client: Any, emit, *, retry_rate_limits: bool = False) -> None:
        self._client = client
        self._emit = emit
        self.retry_rate_limits = retry_rate_limits

    @property
    def token_usage(self):
        return self._client.token_usage

    @property
    def price_per_million(self):
        return self._client.price_per_million

    @property
    def pricing_basis(self):
        return self._client.pricing_basis

    @property
    def default_model(self):
        return self._client.default_model

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=512):
        safe_model = model or self.default_model
        transient_failures = 0
        while True:
            self._emit({"type": "llm_call", "status": "started", "model": safe_model})
            try:
                result = self._client.chat(
                    messages, model=model, temperature=temperature, max_tokens=max_tokens
                )
                break
            except Exception as exc:
                detail = str(exc)[:240]
                secret = getattr(self._client, "api_key", None)
                if secret:
                    detail = detail.replace(secret, "[redacted]")
                network_failure = isinstance(exc.__cause__, (requests.ConnectionError, requests.Timeout))
                if network_failure:
                    transient_failures += 1
                retry = (
                    self.retry_rate_limits and isinstance(exc, ProviderError) and "request failed: 429" in str(exc)
                ) or (network_failure and transient_failures < 3)
                self._emit({
                    "type": "llm_call", "status": "failed", "model": safe_model,
                    "error": detail, "will_retry": retry,
                })
                if not retry:
                    raise
                # The shared Cerebras client paces every attempt at <=5 RPM.
                # DNS/timeouts retry at most three attempts; do not silently
                # classify transport exhaustion as an answerability outcome.
        self._emit({"type": "llm_call", "status": "completed", "model": safe_model})
        return result

    def __getattr__(self, name: str):
        return getattr(self._client, name)


class Dashboard:
    def __init__(self) -> None:
        self.settings = load_settings()
        try:
            # Reuse the rate-limited client across runs so starting a fresh demo
            # cannot reset Cerebras' per-process request pacing.
            self.provider_client = cerebras_client(self.settings) if self.settings.cerebras_api_key else None
        except ProviderError:
            self.provider_client = None
        self.scenarios = self._discover_scenarios()
        self.by_id = {row["id"]: row for row in self.scenarios}
        self.runs: dict[str, RunState] = {}
        self.lock = threading.RLock()
        # One worker serializes runs: each run owns a fresh controller and no two
        # executions mutate the shared retriever simultaneously.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-demo")
        # The dashboard must exercise the same indexed hybrid path as the CLI.
        # Never build an index here: use the configured encoder only when its
        # exact table already exists, otherwise retain full-corpus BM25.
        dense_encoder = None
        dense_index_available = False
        try:
            candidate_encoder = build_encoder(
                self.settings.embedding_backend,
                model=self.settings.embedding_model,
                cache_dir=self.settings.embedding_cache_dir,
                local_files_only=True,
                hash_dim=self.settings.embedding_dim,
                device=self.settings.embedding_device,
                batch_size=self.settings.embedding_batch_size,
                fixed_length=self.settings.embedding_fixed_length,
            )
            candidate_index = LanceIndex(
                self.settings.lancedb_dir, self.settings.table_name, candidate_encoder
            )
            if candidate_index.exists():
                dense_encoder = candidate_encoder
                dense_index_available = True
        except Exception:
            # Dashboard startup remains usable offline with exact full-corpus
            # sparse search when a local neural runtime/model is unavailable.
            pass

        self.retriever = HybridRetriever(
            self.settings.data_dir,
            self.settings.lancedb_dir,
            self.settings.table_name,
            dense_encoder or build_encoder("hash", hash_dim=self.settings.embedding_dim, device="cpu"),
            self.settings.rrf_k,
            self.settings.sparse_weight,
            corpus_limit=None,
            use_dense=dense_index_available,
            retrieval_leg="hybrid" if dense_index_available else "sparse",
            bm25_k1=self.settings.bm25_k1,
            bm25_b=self.settings.bm25_b,
            use_reranker=True,
            reranker_local_files_only=True,
        )
        self.retrieval_mode = "hybrid dense + BM25 + RRF" if dense_index_available else "BM25 sparse only (dense index unavailable)"
        self.indexed_passage_count = len(self.retriever.sparse.passages)
        self.dense_encoder_device = dense_encoder.device if dense_encoder else None
        self._passage_by_id: dict[str, Any] | None = None

    def _discover_scenarios(self) -> list[dict]:
        found: dict[str, dict] = {}
        preview_decomposer = build_decomposer("rule_based")
        # Walk all stream JSONL sources; duplicate stream IDs (e.g. test shortlist)
        # are retained once, with the first canonical source taking precedence.
        candidates = sorted((self.settings.data_dir / "simulated_streams").glob("*.jsonl"))
        curated = list((self.settings.data_dir / "curated_dataset" / "streams").glob("*.jsonl"))
        curated.sort(key=lambda path: (path.name != "test.jsonl", path.name))
        candidates.extend(curated)
        for source in candidates:
            try:
                for line in source.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    stream = SimulatedStream.from_dict(json.loads(line))
                    if stream.stream_id in found:
                        continue
                    preview = preview_decomposer.decompose(
                        stream.base_utterance or (stream.chunks[-1].text if stream.chunks else ""),
                        context=" ".join(f"{speaker}: {text}" for speaker, text in stream.context_turns[-4:]),
                    )
                    found[stream.stream_id] = {
                        "id": stream.stream_id,
                        "source": source.relative_to(self.settings.data_dir).as_posix(),
                        "domain": stream.domain,
                        "category": stream.category,
                        "case_class": stream.case_class,
                        "predicted_intent_count": len(preview.sub_queries),
                        "expected_behavior": stream.expected_behavior,
                        "split": stream.split,
                        "is_evaluated_test": (
                            source.name == "test.jsonl"
                            and source.parent.name == "streams"
                            and stream.split == "test"
                        ),
                        "gold_passage_ids": list(stream.qrel_passage_ids),
                        # Keep the actual utterance searchable and visible to judges;
                        # labels like "answer" are useful metadata but not a query name.
                        "description": stream.base_utterance or stream.expected_behavior or stream.category,
                        "chunk_count": len(stream.chunks),
                        "has_revision": bool(stream.revision_log),
                        "revision_log": [
                            {"chunk_index": i, "from": before, "to": after}
                            for i, before, after in stream.revision_log
                        ],
                        "stream": stream.to_dict(),
                    }
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(found.values(), key=lambda row: (row["domain"], row["category"], row["id"]))

    def provider_available(self) -> bool:
        return self.provider_client is not None

    def start_run(self, payload: dict) -> RunState:
        scenario_id = payload.get("scenario_id")
        if not isinstance(scenario_id, str):
            raise ValueError("scenario_id must be a string")
        scenario = self.by_id.get(scenario_id)
        if scenario is None:
            raise ValueError("Unknown scenario_id")
        session = None
        refinement = False
        if "parent_run_id" in payload:
            parent_id = payload["parent_run_id"]
            if not isinstance(parent_id, str):
                raise ValueError("parent_run_id must be a string")
            parent = self.runs.get(parent_id)
            if parent is None or parent.status != "complete" or not parent.response:
                raise ValueError("Follow-up requires a completed parent run")
            if parent.scenario["domain"] != scenario["domain"]:
                raise ValueError("Follow-up must remain in the same domain")
            text = payload.get("follow_up")
            kind = payload.get("follow_up_kind", "refinement")
            if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                raise ValueError("follow_up must contain 1–2000 characters")
            if kind not in {"refinement", "presentation"}:
                raise ValueError("follow_up_kind must be refinement or presentation")
            from .controller import is_presentation_only
            if kind == "presentation" and not is_presentation_only(text):
                raise ValueError("Use refinement for new facts; presentation is only for reformatting")
            session = ConversationSession(
                answer=parent.response["answer"], citations=list(parent.response["citations"]),
                claim_citations=dict(parent.response.get("claim_citations", {})),
                uncertainty=parent.response.get("uncertainty"), version=parent.response.get("version") or 1,
                domain=parent.scenario["domain"],
            )
            if not session.answer:
                raise ValueError("Parent run has no answer to refine")
            words = text.strip().split()
            ends = list(range(3, len(words), 3)) + [len(words)]
            chunks = tuple(TranscriptChunk(i * 0.5, " ".join(words[:end]), i == len(ends) - 1, 0.95)
                           for i, end in enumerate(ends))
            original = SimulatedStream.from_dict(parent.scenario["stream"])
            followup = replace(original, chunks=chunks, context_turns=(), sub_intents=(),
                               base_utterance=text.strip(), category=kind, qrel_passage_ids=(),
                               case_class="", expected_behavior="", revision_log=(), asr_supersedes=())
            scenario = {**parent.scenario, "stream": followup.to_dict(), "description": text.strip(),
                        "is_evaluated_test": False, "category": kind}
            refinement = kind == "refinement"
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("options must be an object")
        mode = options.get("mode", "provider" if getattr(self, "provider_client", None) is not None else "deterministic")
        if not isinstance(mode, str) or mode not in {"deterministic", "provider"}:
            raise ValueError("mode must be deterministic or provider")
        if mode == "provider" and not self.provider_available():
            raise ValueError("Provider mode is unavailable: no Cerebras API key is configured on the server")
        speed = options.get("speed", 1)
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed not in {0.5, 1, 2, 4}:
            raise ValueError("speed must be one of 0.5, 1, 2, or 4")
        for key in ("multi_intent", "llm_decomposer", "retry_rate_limits"):
            if key in options and not isinstance(options[key], bool):
                raise ValueError(f"{key} must be a boolean")
        safe_options = {
            "mode": mode,
            "speed": speed,
            # Rule decomposition is cheap and active by default; a named
            # multi-intent demo cannot bypass it when the UI option is unchecked.
            "multi_intent": scenario_uses_decomposition(scenario, options),
            "llm_decomposer": bool(options.get("llm_decomposer", False)),
            "retry_rate_limits": bool(options.get("retry_rate_limits", False)),
        }
        if safe_options["llm_decomposer"] and not safe_options["multi_intent"]:
            raise ValueError("llm_decomposer requires multi_intent")
        if safe_options["llm_decomposer"] and mode != "provider":
            raise ValueError("LLM decomposition requires provider mode")
        run = RunState(uuid.uuid4().hex, scenario, safe_options)
        run.session, run.refinement = session, refinement
        with self.lock:
            self.runs[run.run_id] = run
        run.publish({"type": "run", "status": "queued", "scenario_id": scenario_id})
        self.executor.submit(self._execute, run)
        return run

    def _execute(self, run: RunState) -> None:
        run.status = "running"
        run.publish({"type": "run", "status": "running"})
        try:
            stream = SimulatedStream.from_dict(run.scenario["stream"])
            emit = run.publish
            emit({"type": "conversation_context", "turns": [
                {"speaker": speaker, "text": text} for speaker, text in stream.context_turns
            ]})
            evidence_client = generation_client = None
            if run.options["mode"] == "provider":
                evidence_client = generation_client = ObservedChatClient(
                    self.provider_client, emit,
                    retry_rate_limits=run.options["retry_rate_limits"],
                )
            decomposer = None
            if run.options["multi_intent"]:
                if run.options["llm_decomposer"]:
                    decomposer = build_decomposer("llm_based", client=evidence_client)
                else:
                    decomposer = build_decomposer("rule_based")
            pipeline = StreamingRagPipeline(
                self.retriever,
                decomposer=decomposer,
                synthesis_mode=run.options["mode"],
                evidence_client=evidence_client,
                generation_client=generation_client,
                enable_multi_intent=run.options["multi_intent"],
                enable_query_rewrite=run.options["mode"] == "provider",
                verify_evidence_relevance=run.options["mode"] == "provider",
                strict_evidence=True,
                refine_on_final=True,
            )

            chunks_in: queue.Queue = queue.Queue()
            playback_started = time.monotonic()

            def deliver_asr_chunks() -> None:
                for chunk in stream.chunks:
                    deadline = playback_started + max(0.0, chunk.timestamp_s) / float(run.options["speed"])
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    emit({
                        "type": "chunk",
                        "timestamp_s": chunk.timestamp_s,
                        "received_elapsed_s": round(time.monotonic() - playback_started, 3),
                        "text": chunk.text,
                        "is_final": chunk.is_final,
                        "confidence": chunk.confidence,
                    })
                    chunks_in.put(chunk)
                chunks_in.put(None)

            threading.Thread(target=deliver_asr_chunks, name=f"asr-{run.run_id[:8]}", daemon=True).start()

            def incoming_chunks():
                while True:
                    chunk = chunks_in.get()
                    if chunk is None:
                        return
                    yield chunk

            response: RagResponse = pipeline.run(
                incoming_chunks(), domain=stream.domain, context_turns=stream.context_turns,
                trace=True, on_event=emit, session=run.session, refinement=run.refinement,
            )
            response_data = response.to_dict()
            # Claim-level links are exact extractive mappings only; absence is meaningful.
            response_data["claim_citations"] = response.claim_citations
            response_data["domain"] = stream.domain
            response_data["synthesis_mode"] = pipeline.synthesis_mode
            response_data["provider_routing"] = "query-selective" if pipeline.synthesis_mode == "provider" else "disabled"
            response_data["provider_model"] = (
                getattr(evidence_client, "default_model", None) if evidence_client else None
            )
            response_data["trace_id"] = response.trace_id
            response_data["evaluation"] = evaluate_dashboard_run(
                stream, response, is_test=run.scenario["is_evaluated_test"]
            )
            trace_path = self.settings.root_dir / ".cache" / "telemetry" / "dashboard-traces.jsonl"
            _append_dashboard_trace(trace_path, {
                **response.to_dict(), "domain": stream.domain, "status": "complete",
            })
            response_data["trace_validation"] = validate_trace_file(trace_path, expected_ids={response.trace_id})
            run.response = response_data
            run.status = "complete"
            run.publish({"type": "answer", **response_data})
            run.publish({"type": "done", "status": "complete"})
        except Exception as exc:
            detail = str(exc)
            secret = self.settings.cerebras_api_key
            if secret:
                detail = detail.replace(secret, "[redacted]")
            run.error = f"{type(exc).__name__}: {detail[:400]}"
            run.status = "error"
            trace_path = self.settings.root_dir / ".cache" / "telemetry" / "dashboard-traces.jsonl"
            _append_dashboard_trace(trace_path, {
                "trace_id": uuid.uuid4().hex, "domain": run.scenario.get("domain", ""),
                "status": "error", "retrieval_events": [], "decisions": [],
                "citations": [], "answer": "", "version": 1,
                "previous_version": None, "applied_delta": None,
                "stage_latency_ms": {}, "token_usage": {},
                "estimated_cost_usd": None,
                "cost_estimate_basis": "request failed before usage was available",
            })
            run.publish({"type": "error", "message": run.error})

    def evidence(self, passage_id: str) -> dict | None:
        if self._passage_by_id is None:
            self._passage_by_id = {
                passage.id: passage
                for domain in ("cloud", "govt")
                for passage in iter_passages(self.settings.data_dir, domain)
            }
        passage = self._passage_by_id.get(passage_id)
        if passage is None:
            return None
        return {
            "id": passage.id, "domain": passage.domain, "title": passage.title,
            "url": passage.url, "text": passage.text,
        }


def _append_dashboard_trace(path: Path, payload: dict) -> None:
    """Append a schema-complete local trace without recording provider secrets."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")


class DashboardHandler(BaseHTTPRequestHandler):
    dashboard: Dashboard

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        super().end_headers()

    def _local_request(self) -> bool:
        # Loopback-only deployment, including Docker's published localhost port.
        # Reject arbitrary Host names to avoid exposing local traces via DNS rebinding.
        try:
            host = urlparse("//" + self.headers.get("Host", "")).hostname
        except ValueError:
            host = None
        if host not in {"localhost", "127.0.0.1", "::1"}:
            self._json({"error": "Use the localhost dashboard URL"}, 403)
            return False
        return True

    def log_message(self, format: str, *args) -> None:
        # Avoid logging query strings or request bodies that may contain transcript text.
        super().log_message(format, *args)

    def _json(self, value: dict, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if not self._local_request():
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json({"status": "ready", "corpus_passages_loaded": self.dashboard.indexed_passage_count,
                        "retrieval_mode": self.dashboard.retrieval_mode})
        elif path == "/api/scenarios":
            self._json({
                "scenarios": [
                    {k: v for k, v in row.items() if k not in {"stream", "gold_passage_ids"}}
                    for row in self.dashboard.scenarios
                ],
                "runtime_code_hash": runtime_code_hash(),
                "provider_available": self.dashboard.provider_available(),
                "default_mode": "provider" if self.dashboard.provider_available() else "deterministic",
                "provider_model": self.dashboard.settings.cerebras_model if self.dashboard.provider_available() else None,
                "retrieval_mode": self.dashboard.retrieval_mode,
                "dense_enabled": self.dashboard.retriever.use_dense,
                "dense_device": self.dashboard.dense_encoder_device,
                "corpus_passages_loaded": self.dashboard.indexed_passage_count,
                "corpus_limit_per_domain": None,
            })
        elif path.startswith("/api/runs/") and path.endswith("/events"):
            self._sse(unquote(path.split("/")[-2]))
        elif path.startswith("/api/runs/"):
            run_id = unquote(path.rsplit("/", 1)[-1])
            run = self.dashboard.runs.get(run_id)
            self._json(run.snapshot() if run else {"error": "run not found"}, 200 if run else 404)
        elif path.startswith("/api/evidence/"):
            evidence = self.dashboard.evidence(unquote(path.rsplit("/", 1)[-1]))
            self._json(evidence or {"error": "passage not found"}, 200 if evidence else 404)
        elif path in {"/", "/index.html"} or path.startswith("/assets/"):
            relative = "index.html" if path in {"/", "/index.html"} else path.removeprefix("/assets/")
            target = (ASSET_DIR / relative).resolve()
            if ASSET_DIR.resolve() not in target.parents and target != ASSET_DIR.resolve() / "index.html":
                self._json({"error": "not found"}, 404)
                return
            if not target.is_file():
                self._json({"error": "not found"}, 404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._local_request():
            return
        origin = self.headers.get("Origin")
        try:
            parsed_origin = urlparse(origin or "")
            same_origin = (parsed_origin.scheme in {"http", "https"}
                           and parsed_origin.netloc.lower() == self.headers.get("Host", "").lower()
                           and not parsed_origin.path and not parsed_origin.query and not parsed_origin.fragment)
        except ValueError:
            same_origin = False
        if self.headers.get("Sec-Fetch-Site") == "cross-site" or (origin and not same_origin):
            self._json({"error": "Cross-origin run requests are not allowed"}, 403)
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._json({"error": "Content-Type must be application/json"}, 415)
            return
        if urlparse(self.path).path != "/api/runs":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64_000:
                raise ValueError("request body too large")
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("transfer encoding is not supported")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            run = self.dashboard.start_run(payload)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
            return
        self._json({"run_id": run.run_id, "status": run.status}, 202)

    def _sse(self, run_id: str) -> None:
        run = self.dashboard.runs.get(run_id)
        if run is None:
            self._json({"error": "run not found"}, 404)
            return
        try:
            last_id = max(0, int(self.headers.get("Last-Event-ID", "0")))
        except ValueError:
            last_id = 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.end_headers()
        cursor = last_id
        try:
            while True:
                with run.condition:
                    if cursor >= len(run.events) and run.status not in {"complete", "error"}:
                        run.condition.wait(timeout=15)
                    pending = [event for event in run.events if event["id"] > cursor]
                    finished = run.status in {"complete", "error"}
                if not pending:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    if finished:
                        break
                    continue
                for event in pending:
                    payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
                    self.wfile.write(b"id: " + str(event["id"]).encode() + b"\n")
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
                    cursor = event["id"]
                if finished and cursor >= len(run.events):
                    break
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            # A completed SSE response has no Content-Length by design. Close its
            # HTTP connection so clients can observe EOF after the terminal event.
            self.close_connection = True

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()


def serve_dashboard(*, host: str = "127.0.0.1", port: int = 8080) -> None:
    dashboard = Dashboard()
    handler = type("BoundDashboardHandler", (DashboardHandler,), {"dashboard": dashboard})
    display_host = "localhost" if host in {"0.0.0.0", "::"} else host
    print(f"Prism Live RAG dashboard: http://{display_host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        dashboard.executor.shutdown(wait=False, cancel_futures=True)
