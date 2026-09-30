from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from .models import Passage
from .models import QueryTask
from .providers import OpenAICompatibleChatClient, ProviderError
from .stream import SimulatedStream, early_retrieval_stream


SCHEMA_VERSION = "1.0"
CASE_CLASSES = ("answerable", "partial", "unanswerable", "underspecified")
EXPECTED_BEHAVIOURS = {
    "answerable": "answer",
    "partial": "partial_answer",
    "unanswerable": "abstain",
    "underspecified": "clarify",
}
SPLIT_TARGETS: dict[str, dict[str, int]] = {
    "train": {"answerable": 300, "partial": 90, "unanswerable": 120, "underspecified": 90},
    "dev": {"answerable": 60, "partial": 18, "unanswerable": 24, "underspecified": 18},
    "test": {"answerable": 100, "partial": 30, "unanswerable": 40, "underspecified": 30},
}
SPLIT_HASH_RANGES = {
    "train": range(0, 6522),
    "dev": range(6522, 7826),
    "test": range(7826, 10000),
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it of on or that the "
    "their this to was what when where which who why with you your".split()
)


class CuratedDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class SupportSpan:
    passage_id: str
    start: int
    end: int
    text: str

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "SupportSpan":
        return cls(
            passage_id=str(row.get("passage_id", "")),
            start=int(row.get("start", -1)),
            end=int(row.get("end", -1)),
            text=str(row.get("text", "")),
        )


@dataclass
class CuratedCase:
    case_id: str
    split: str
    domain: str
    case_class: str
    query: str
    expected_behavior: str
    source_document_group: str
    dialogue_history: list[dict[str, str]] = field(default_factory=list)
    reference_answer: str = ""
    required_clarification: str = ""
    gold_passage_ids: list[str] = field(default_factory=list)
    support_spans: list[SupportSpan] = field(default_factory=list)
    distractor_passage_ids: list[str] = field(default_factory=list)
    difficulty: str = "standard"
    tags: list[str] = field(default_factory=list)
    provenance: dict[str, object] = field(default_factory=dict)
    generation: dict[str, object] = field(default_factory=dict)
    review: dict[str, object] = field(default_factory=lambda: {"status": "pending"})
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "CuratedCase":
        spans = row.get("support_spans") or []
        return cls(
            case_id=str(row.get("case_id", "")),
            split=str(row.get("split", "")),
            domain=str(row.get("domain", "")),
            case_class=str(row.get("case_class", "")),
            query=str(row.get("query", "")),
            expected_behavior=str(row.get("expected_behavior", "")),
            source_document_group=str(row.get("source_document_group", "")),
            dialogue_history=[dict(turn) for turn in row.get("dialogue_history") or []],
            reference_answer=str(row.get("reference_answer", "")),
            required_clarification=str(row.get("required_clarification", "")),
            gold_passage_ids=[str(item) for item in row.get("gold_passage_ids") or []],
            support_spans=[SupportSpan.from_dict(item) for item in spans],
            distractor_passage_ids=[str(item) for item in row.get("distractor_passage_ids") or []],
            difficulty=str(row.get("difficulty", "standard")),
            tags=[str(item) for item in row.get("tags") or []],
            provenance=dict(row.get("provenance") or {}),
            generation=dict(row.get("generation") or {}),
            review=dict(row.get("review") or {"status": "pending"}),
            schema_version=str(row.get("schema_version", SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def curated_dir(data_dir: Path) -> Path:
    return data_dir / "curated_dataset"


def cases_path(data_dir: Path, split: str) -> Path:
    return curated_dir(data_dir) / f"{split}.jsonl"


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise CuratedDatasetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def load_cases(path: Path) -> list[CuratedCase]:
    return [CuratedCase.from_dict(row) for row in iter_jsonl(path)]


def write_cases(path: Path, cases: Sequence[CuratedCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def normalized_document_group(passage: Passage) -> str:
    if passage.url.strip():
        url = passage.url.strip().lower().split("#", 1)[0]
        return f"{passage.domain}:url:{url}"
    prefix = passage.id.rsplit("-", 2)[0] if passage.id.count("-") >= 2 else passage.id
    return f"{passage.domain}:id:{prefix}"


def assigned_split(document_group: str, seed: int = 20260928) -> str:
    digest = hashlib.sha256(f"{seed}:{document_group}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    return next(name for name, values in SPLIT_HASH_RANGES.items() if bucket in values)


def passage_is_suitable(passage: Passage) -> bool:
    text = passage.text.strip()
    if not 300 <= len(text) <= 5000:
        return False
    tokens = TOKEN_RE.findall(text.lower())
    if len(tokens) < 55 or len(set(tokens)) < 25:
        return False
    if text.count("http") > 8:
        return False
    return True


def select_source_passages(
    passages: Iterable[Passage],
    split: str,
    *,
    seed: int,
    limit: int,
) -> list[Passage]:
    candidates = [
        passage
        for passage in passages
        if passage_is_suitable(passage)
        and assigned_split(normalized_document_group(passage), seed) == split
    ]
    random.Random(seed + sum(ord(ch) for ch in split)).shuffle(candidates)
    seen_groups: set[str] = set()
    selected: list[Passage] = []
    for passage in candidates:
        group = normalized_document_group(passage)
        if group in seen_groups:
            continue
        selected.append(passage)
        seen_groups.add(group)
        if len(selected) >= limit:
            break
    return selected


def _extract_json_payload(raw: str) -> object:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start_candidates = [pos for pos in (raw.find("["), raw.find("{")) if pos >= 0]
    if not start_candidates:
        raise CuratedDatasetError("generator returned no JSON payload")
    start = min(start_candidates)
    end = max(raw.rfind("]"), raw.rfind("}"))
    if end < start:
        raise CuratedDatasetError("generator returned incomplete JSON")
    return json.loads(raw[start : end + 1])


def _generation_prompt(passage: Passage, requested_classes: Sequence[str]) -> str:
    classes = ", ".join(requested_classes)
    return f"""Create grounded RAG evaluation cases from the passage below.

Return only a JSON array with one object for each requested class, in this order: {classes}.
Each object must contain: case_class, query, reference_answer, required_clarification,
support_spans (an array of exact verbatim substrings copied from PASSAGE), difficulty,
tags, and dialogue_history (an array of speaker/text objects).

Rules:
- answerable: every answer claim must be entailed by its support spans.
- partial: ask for two related details; the passage must support exactly one, and the
  answer must state which requested detail is missing.
- unanswerable: ask for a plausible but absent detail and return an empty answer.
- underspecified: omit a necessary referent and provide one targeted clarification.
- Do not use outside knowledge or invent product names, numbers, dates, or procedures.
- At most one third of cases may use dialogue history.

PASSAGE ID: {passage.id}
TITLE: {passage.title}
PASSAGE:
{passage.text}
"""


def _locate_spans(passage: Passage, span_texts: Sequence[str]) -> list[SupportSpan]:
    spans: list[SupportSpan] = []
    for text in span_texts:
        clean = str(text).strip()
        start = passage.text.find(clean)
        if not clean or start < 0:
            raise CuratedDatasetError(
                f"generated support span is not an exact substring of {passage.id}: {clean[:80]!r}"
            )
        spans.append(SupportSpan(passage.id, start, start + len(clean), clean))
    return spans


def generate_cases_for_passage(
    client: OpenAICompatibleChatClient,
    passage: Passage,
    *,
    split: str,
    requested_classes: Sequence[str],
    seed: int,
) -> list[CuratedCase]:
    raw = client.chat(
        [
            {"role": "system", "content": "You build auditable evaluation data. Output valid JSON only."},
            {"role": "user", "content": _generation_prompt(passage, requested_classes)},
        ],
        temperature=0.0,
        max_tokens=1800,
    )
    payload = _extract_json_payload(raw)
    rows = payload.get("cases", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise CuratedDatasetError("generator JSON must be an array or an object with cases[]")
    if len(rows) != len(requested_classes):
        raise CuratedDatasetError(
            f"generator returned {len(rows)} cases for {len(requested_classes)} requested classes"
        )

    generated: list[CuratedCase] = []
    for ordinal, (expected_class, row) in enumerate(zip(requested_classes, rows, strict=True)):
        if not isinstance(row, dict) or row.get("case_class") != expected_class:
            raise CuratedDatasetError(f"expected {expected_class} case at position {ordinal}")
        span_texts = row.get("support_spans") or []
        spans = _locate_spans(passage, span_texts)
        if expected_class in {"answerable", "partial"} and not spans:
            raise CuratedDatasetError(f"{expected_class} case has no exact support span")
        if expected_class in {"unanswerable", "underspecified"} and spans:
            raise CuratedDatasetError(f"{expected_class} case must not declare support spans")
        case_key = f"{seed}:{split}:{passage.id}:{ordinal}:{row.get('query', '')}"
        case_id = "curated-" + hashlib.sha256(case_key.encode("utf-8")).hexdigest()[:16]
        generated.append(
            CuratedCase(
                case_id=case_id,
                split=split,
                domain=passage.domain,
                case_class=expected_class,
                query=str(row.get("query", "")).strip(),
                expected_behavior=EXPECTED_BEHAVIOURS[expected_class],
                source_document_group=normalized_document_group(passage),
                dialogue_history=[dict(turn) for turn in row.get("dialogue_history") or []],
                reference_answer=str(row.get("reference_answer", "")).strip(),
                required_clarification=str(row.get("required_clarification", "")).strip(),
                gold_passage_ids=[passage.id] if spans else [],
                support_spans=spans,
                difficulty=str(row.get("difficulty", "standard")),
                tags=sorted({str(tag) for tag in row.get("tags") or []}),
                provenance={
                    "source": "corpus_generated",
                    "passage_id": passage.id,
                    "url": passage.url,
                },
                generation={
                    "provider": "cerebras" if "cerebras.ai" in getattr(client, "base_url", "") else "unknown",
                    "model": client.default_model,
                    "seed": seed,
                    "temperature": 0.0,
                },
            )
        )
    return generated


def generate_candidate_pool(
    client: OpenAICompatibleChatClient,
    passages_by_domain: Mapping[str, Sequence[Passage]],
    *,
    split: str,
    seed: int = 20260928,
    multiplier: float = 2.0,
    targets: Mapping[str, int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[CuratedCase]:
    if split not in SPLIT_TARGETS:
        raise CuratedDatasetError(f"unknown split: {split}")
    if targets is None:
        target_classes = ("answerable", "partial") if split == "test" else CASE_CLASSES
        wanted = {
            name: max(1, int(SPLIT_TARGETS[split][name] * multiplier))
            for name in target_classes
        }
    else:
        invalid = set(targets) - set(CASE_CLASSES)
        if invalid or any(count <= 0 for count in targets.values()):
            raise CuratedDatasetError(f"invalid generation targets: {dict(targets)}")
        if split == "test" and set(targets) - {"answerable", "partial"}:
            raise CuratedDatasetError("generated test targets may only be answerable or partial")
        target_classes = tuple(name for name in CASE_CLASSES if name in targets)
        wanted = {name: int(targets[name]) for name in target_classes}
    cases: list[CuratedCase] = []
    counts: Counter[str] = Counter()
    ordered: list[Passage] = []
    for domain in sorted(passages_by_domain):
        ordered.extend(passages_by_domain[domain])
    random.Random(seed).shuffle(ordered)

    failures = 0
    for passage in ordered:
        remaining = [name for name in target_classes if counts[name] < wanted[name]]
        if not remaining:
            break
        # One call can create up to four differently labelled cases from one passage.
        requested = remaining[:4]
        try:
            batch = generate_cases_for_passage(
                client, passage, split=split, requested_classes=requested, seed=seed
            )
        except (CuratedDatasetError, ProviderError, ValueError) as exc:
            failures += 1
            if progress:
                progress(f"skip {passage.id}: {exc}")
            continue
        cases.extend(batch)
        counts.update(case.case_class for case in batch)
        if progress:
            progress(f"generated {len(cases)} cases; counts={dict(counts)}")
    short = {
        name: wanted[name] - counts[name]
        for name in target_classes
        if counts[name] < wanted[name]
    }
    if short:
        raise CuratedDatasetError(f"candidate generation exhausted passages; short={short}, failures={failures}")
    return cases


def seed_benchmark_negatives(
    reference_path: Path,
    *,
    split: str = "test",
    seed: int = 20260928,
    quotas: Mapping[str, int] | None = None,
) -> list[CuratedCase]:
    """Create review candidates from MTRAG-UN's authoritative negative labels.

    These are intentionally not synthesized. They preserve the original conversation and
    label so the frozen test set does not rely on a model proving its own negatives.
    """
    wanted = dict(quotas or {"unanswerable": 40, "underspecified": 30})
    collection_domain = {"ibmcloud": "cloud", "govt": "govt"}
    pools: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in iter_jsonl(reference_path):
        domain = collection_domain.get(str(row.get("Collection", "")))
        labels = row.get("answerability") or []
        label = str(labels[0]) if labels else ""
        case_class = label.lower()
        if domain and case_class in wanted:
            row = dict(row)
            row["_domain"] = domain
            pools[case_class].append(row)
    rng = random.Random(seed)
    cases: list[CuratedCase] = []
    for case_class, count in wanted.items():
        pool = pools.get(case_class, [])
        rng.shuffle(pool)
        if len(pool) < count:
            raise CuratedDatasetError(
                f"MTRAG-UN has {len(pool)} {case_class} covered cases; requested {count}"
            )
        selected: list[dict[str, object]] = []
        by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in pool:
            by_domain[str(row["_domain"])].append(row)
        while len(selected) < count:
            made_progress = False
            for domain in sorted(by_domain):
                if by_domain[domain] and len(selected) < count:
                    selected.append(by_domain[domain].pop())
                    made_progress = True
            if not made_progress:
                break
        for row in selected:
            inputs = [dict(turn) for turn in row.get("input") or []]
            last_user_index = max(
                (index for index, turn in enumerate(inputs) if turn.get("speaker") == "user"),
                default=-1,
            )
            if last_user_index < 0:
                continue
            query = str(inputs[last_user_index].get("text", "")).strip()
            history = [
                {"speaker": str(turn.get("speaker", "")), "text": str(turn.get("text", ""))}
                for turn in inputs[:last_user_index]
            ]
            targets = row.get("targets") or []
            target_text = str(targets[0].get("text", "")) if targets else ""
            task_id = str(row.get("task_id", ""))
            conversation_id = str(row.get("conversation_id", task_id.split("<::>", 1)[0]))
            cases.append(
                CuratedCase(
                    case_id="mtragun-" + hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16],
                    split=split,
                    domain=str(row["_domain"]),
                    case_class=case_class,
                    query=query,
                    expected_behavior=EXPECTED_BEHAVIOURS[case_class],
                    source_document_group=f"benchmark:conversation:{conversation_id}",
                    dialogue_history=history,
                    required_clarification=target_text if case_class == "underspecified" else "",
                    tags=["benchmark", "multi_turn"] if history else ["benchmark"],
                    provenance={
                        "source": "mtragun",
                        "task_id": task_id,
                        "conversation_id": conversation_id,
                        "original_answerability": str((row.get("answerability") or [""])[0]),
                    },
                    generation={"provider": None, "seed": seed},
                )
            )
    return sorted(cases, key=lambda case: case.case_id)


def _best_support_spans(
    passages: Sequence[Passage], reference_answer: str, *, max_spans: int = 3
) -> list[SupportSpan]:
    answer_tokens = set(TOKEN_RE.findall(reference_answer.lower()))
    ranked: list[tuple[float, SupportSpan]] = []
    for passage in passages:
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]|$)", passage.text):
            text = match.group(0).strip()
            tokens = set(TOKEN_RE.findall(text.lower()))
            if len(tokens) < 4:
                continue
            overlap = len(tokens & answer_tokens) / max(len(tokens | answer_tokens), 1)
            if overlap <= 0:
                continue
            start = passage.text.find(text, match.start(), match.end() + 1)
            ranked.append(
                (overlap, SupportSpan(passage.id, start, start + len(text), text))
            )
    ranked.sort(key=lambda item: (-item[0], item[1].passage_id, item[1].start))
    return [span for _, span in ranked[:max_spans]]


def seed_benchmark_positives(
    reference_path: Path,
    passages: Mapping[str, Passage],
    *,
    split: str = "test",
    seed: int = 20260928,
    quotas: Mapping[str, int] | None = None,
) -> list[CuratedCase]:
    """Seed answerable/partial candidates with MTRAG-UN answers and gold passages."""
    wanted = dict(quotas or {"answerable": 100, "partial": 22})
    collection_domain = {"ibmcloud": "cloud", "govt": "govt"}
    pools: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in iter_jsonl(reference_path):
        domain = collection_domain.get(str(row.get("Collection", "")))
        labels = row.get("answerability") or []
        case_class = str(labels[0]).lower() if labels else ""
        if not domain or case_class not in wanted:
            continue
        context_ids = [str(context.get("document_id", "")) for context in row.get("contexts") or []]
        if context_ids and all(passage_id in passages for passage_id in context_ids):
            copied = dict(row)
            copied["_domain"] = domain
            copied["_context_ids"] = context_ids
            pools[case_class].append(copied)
    rng = random.Random(seed)
    cases: list[CuratedCase] = []
    for case_class, count in wanted.items():
        pool = pools.get(case_class, [])
        rng.shuffle(pool)
        if len(pool) < count:
            raise CuratedDatasetError(
                f"MTRAG-UN has {len(pool)} usable {case_class} positives; requested {count}"
            )
        class_start = len(cases)
        for row in pool:
            if len(cases) - class_start >= count:
                break
            inputs = [dict(turn) for turn in row.get("input") or []]
            last_user_index = max(
                (index for index, turn in enumerate(inputs) if turn.get("speaker") == "user"),
                default=-1,
            )
            targets = row.get("targets") or []
            reference_answer = str(targets[0].get("text", "")) if targets else ""
            context_passages = [passages[passage_id] for passage_id in row["_context_ids"]]
            spans = _best_support_spans(context_passages, reference_answer)
            if last_user_index < 0 or not reference_answer or not spans:
                continue
            task_id = str(row.get("task_id", ""))
            history = [
                {"speaker": str(turn.get("speaker", "")), "text": str(turn.get("text", ""))}
                for turn in inputs[:last_user_index]
            ]
            cases.append(
                CuratedCase(
                    case_id="mtragun-" + hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16],
                    split=split,
                    domain=str(row["_domain"]),
                    case_class=case_class,
                    query=str(inputs[last_user_index].get("text", "")).strip(),
                    expected_behavior=EXPECTED_BEHAVIOURS[case_class],
                    source_document_group=normalized_document_group(context_passages[0]),
                    dialogue_history=history,
                    reference_answer=reference_answer,
                    gold_passage_ids=list(dict.fromkeys(row["_context_ids"])),
                    support_spans=spans,
                    tags=["benchmark", "multi_turn"] if history else ["benchmark"],
                    provenance={
                        "source": "mtragun",
                        "task_id": task_id,
                        "conversation_id": str(row.get("conversation_id", "")),
                        "original_answerability": str((row.get("answerability") or [""])[0]),
                        "support_spans": "lexical_candidates_pending_human_review",
                    },
                    generation={"provider": None, "seed": seed},
                )
            )
        created = len(cases) - class_start
        if created < count:
            raise CuratedDatasetError(
                f"MTRAG-UN yielded only {created}/{count} valid {case_class} positive cases"
            )
    return sorted(cases, key=lambda case: case.case_id)


def attach_distractors(
    cases: Sequence[CuratedCase],
    search: Callable[[str, str, int], Sequence[str]],
    *,
    limit: int = 5,
) -> list[CuratedCase]:
    """Attach high-ranking non-gold passages without changing the corpus or gold set."""
    for case in cases:
        candidates = search(case.query, case.domain, max(limit * 4, 20))
        case.distractor_passage_ids = [
            passage_id
            for passage_id in dict.fromkeys(candidates)
            if passage_id not in case.gold_passage_ids
        ][:limit]
        if case.distractor_passage_ids and "hard_distractor" not in case.tags:
            case.tags.append("hard_distractor")
            case.tags.sort()
    return list(cases)


def audit_case(case: CuratedCase) -> dict[str, object]:
    """Return deterministic quality signals without pretending to be human review."""
    flags: list[str] = []
    query_tokens = set(TOKEN_RE.findall(case.query.lower())) - STOPWORDS
    answer_tokens = set(TOKEN_RE.findall(case.reference_answer.lower())) - STOPWORDS
    evidence_text = " ".join(span.text for span in case.support_spans).lower()
    evidence_tokens = set(TOKEN_RE.findall(evidence_text)) - STOPWORDS
    answer_coverage = (
        len(answer_tokens & evidence_tokens) / len(answer_tokens) if answer_tokens else None
    )
    query_evidence_overlap = (
        len(query_tokens & evidence_tokens) / len(query_tokens) if query_tokens else 0.0
    )

    if case.case_class in {"answerable", "partial"}:
        score = 0.15 + 0.70 * float(answer_coverage or 0.0) + 0.15 * query_evidence_overlap
        if answer_coverage is not None and answer_coverage < 0.35:
            flags.append("low_answer_span_overlap")
        if not case.support_spans:
            flags.append("missing_support_span")
    else:
        score = 0.90 if case.provenance.get("source") == "mtragun" else 0.55
    if not 4 <= len(TOKEN_RE.findall(case.query)) <= 80:
        score -= 0.10
        flags.append("unusual_query_length")
    if case.case_class == "partial":
        lowered = case.reference_answer.lower()
        missing_markers = (
            "not provide", "does not provide", "do not have", "no information",
            "cannot", "however", "but ", "not specify", "not state",
        )
        if any(marker in lowered for marker in missing_markers):
            score += 0.10
        else:
            flags.append("partial_answer_missing_explicit_gap")
    if case.case_class == "underspecified" and not case.required_clarification.strip():
        score -= 0.25
        flags.append("missing_clarification")
    if case.provenance.get("support_spans") == "lexical_candidates_pending_human_review":
        flags.append("lexical_spans_need_review")
    return {
        "score": round(max(0.0, min(score, 1.0)), 4),
        "answer_span_token_coverage": (
            round(answer_coverage, 4) if answer_coverage is not None else None
        ),
        "query_evidence_overlap": round(query_evidence_overlap, 4),
        "flags": flags,
    }


def shortlist_cases(
    cases: Sequence[CuratedCase],
    split: str,
    *,
    balance_partial_domains: bool = True,
) -> tuple[list[CuratedCase], list[dict[str, object]]]:
    """Choose a review shortlist by deterministic signals; never approve cases."""
    targets = SPLIT_TARGETS[split]
    selected: list[CuratedCase] = []
    audit_rows: list[dict[str, object]] = []
    for case_class, target in targets.items():
        pool = [case for case in cases if case.case_class == case_class]
        ranked = sorted(
            pool,
            key=lambda case: (
                -float(audit_case(case)["score"]),
                case.case_id,
            ),
        )
        if len(ranked) < target:
            raise CuratedDatasetError(
                f"{split}/{case_class}: shortlist needs {target} cases, found {len(ranked)}"
            )
        if case_class == "partial" and balance_partial_domains:
            per_domain = target // 2
            cloud = [case for case in ranked if case.domain == "cloud"][:per_domain]
            govt = [case for case in ranked if case.domain == "govt"][:per_domain]
            picked_ids = {case.case_id for case in cloud + govt}
            picked = cloud + govt
            picked.extend(case for case in ranked if case.case_id not in picked_ids)
            chosen = picked[:target]
        else:
            chosen = ranked[:target]
        chosen_ids = {case.case_id for case in chosen}
        selected.extend(chosen)
        for case in ranked:
            audit = audit_case(case)
            audit_rows.append(
                {
                    "case_id": case.case_id,
                    "case_class": case.case_class,
                    "domain": case.domain,
                    "selected": case.case_id in chosen_ids,
                    **audit,
                }
            )
    return sorted(selected, key=lambda case: case.case_id), audit_rows


def _revision_log(stream: SimulatedStream) -> tuple[tuple[int, str, str], ...]:
    revisions: list[tuple[int, str, str]] = []
    partials = [chunk for chunk in stream.chunks if not chunk.is_final]
    for index in range(1, len(partials)):
        before = partials[index - 1].text
        after = partials[index].text
        if not after.startswith(before + " ") and after != before:
            revisions.append((index, before, after))
    return tuple(revisions)


def curated_case_stream(
    case: CuratedCase,
    *,
    seed: int = 20260928,
    revision_probability: float = 0.35,
) -> SimulatedStream:
    """Convert one current-turn query into an interval-based ASR partial stream."""
    digest = hashlib.sha256(f"{seed}:{case.case_id}".encode("utf-8")).digest()
    case_seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(case_seed)
    task = QueryTask(
        task_id=case.case_id,
        domain=case.domain,
        query=case.query,
        answerability=(case.case_class.upper(),),
        qrel_passage_ids=tuple(case.gold_passage_ids),
    )
    base = early_retrieval_stream(
        task,
        case.domain,
        f"{case.case_id}-stream",
        rng,
        revision=rng.random() < revision_probability,
    )
    final_index = len(base.chunks) - 1
    return replace(
        base,
        category=(
            "early_retrieval"
            if case.case_class in {"answerable", "partial"}
            else case.case_class
        ),
        split=case.split,
        case_class=case.case_class,
        expected_behavior=case.expected_behavior,
        base_utterance=case.query,
        context_turns=tuple(
            (str(turn.get("speaker", "")), str(turn.get("text", "")))
            for turn in case.dialogue_history
        ),
        asr_supersedes=tuple(range(final_index)),
        revision_log=_revision_log(base),
        provenance=str(case.provenance.get("source", "")),
    )


def generate_curated_streams(
    cases: Sequence[CuratedCase],
    *,
    seed: int = 20260928,
    revision_probability: float = 0.35,
) -> list[SimulatedStream]:
    return [
        curated_case_stream(
            case, seed=seed, revision_probability=revision_probability
        )
        for case in sorted(cases, key=lambda item: item.case_id)
    ]


def write_curated_streams(path: Path, streams: Sequence[SimulatedStream]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for stream in streams:
            handle.write(json.dumps(stream.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def load_curated_streams(path: Path) -> list[SimulatedStream]:
    return [SimulatedStream.from_dict(row) for row in iter_jsonl(path)]


def validate_curated_streams(
    cases: Sequence[CuratedCase], streams: Sequence[SimulatedStream]
) -> dict[str, object]:
    cases_by_id = {case.case_id: case for case in cases}
    problems: list[str] = []
    if not cases:
        problems.append("no curated cases were loaded")
    if not streams:
        problems.append("no curated streams were loaded")
    revision_count = 0
    for stream in streams:
        case = cases_by_id.get(stream.task_id)
        prefix = stream.stream_id
        if case is None:
            problems.append(f"{prefix}: no matching curated case")
            continue
        finals = [index for index, chunk in enumerate(stream.chunks) if chunk.is_final]
        if finals != [len(stream.chunks) - 1]:
            problems.append(f"{prefix}: must contain exactly one final chunk at the end")
            continue
        if len(case.query.split()) > 1 and len(stream.chunks) < 3:
            problems.append(f"{prefix}: utterance was flattened instead of streamed")
        timestamps = [chunk.timestamp_s for chunk in stream.chunks]
        if any(after <= before for before, after in zip(timestamps, timestamps[1:])):
            problems.append(f"{prefix}: timestamps are not strictly increasing")
        if stream.chunks[-1].text.split() != case.query.split():
            problems.append(f"{prefix}: final ASR text differs from the case query")
        expected_supersedes = tuple(range(len(stream.chunks) - 1))
        if stream.asr_supersedes != expected_supersedes:
            problems.append(f"{prefix}: final chunk does not supersede every partial")
        if stream.context_turns != tuple(
            (str(turn.get("speaker", "")), str(turn.get("text", "")))
            for turn in case.dialogue_history
        ):
            problems.append(f"{prefix}: conversation context was not preserved")
        if stream.qrel_passage_ids != tuple(case.gold_passage_ids):
            problems.append(f"{prefix}: gold passage IDs changed during streaming conversion")
        revision_count += len(stream.revision_log)
        for chunk_index, before, after in stream.revision_log:
            if not 0 < chunk_index < len(stream.chunks) - 1 or before == after:
                problems.append(f"{prefix}: invalid revision log entry")
    if len(streams) != len(cases):
        problems.append(f"stream count {len(streams)} != case count {len(cases)}")
    return {
        "valid": not problems,
        "cases": len(cases),
        "streams": len(streams),
        "streams_with_revisions": sum(bool(stream.revision_log) for stream in streams),
        "revision_events": revision_count,
        "problems": problems,
    }


def export_review_csv(cases: Sequence[CuratedCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "split", "domain", "case_class", "query", "reference_answer",
        "required_clarification", "gold_passage_ids", "support_spans", "status",
        "audit_score", "audit_flags", "reviewer", "rejection_reason", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            audit = audit_case(case)
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "split": case.split,
                    "domain": case.domain,
                    "case_class": case.case_class,
                    "query": case.query,
                    "reference_answer": case.reference_answer,
                    "required_clarification": case.required_clarification,
                    "gold_passage_ids": json.dumps(case.gold_passage_ids),
                    "support_spans": json.dumps([asdict(span) for span in case.support_spans], ensure_ascii=False),
                    "status": case.review.get("status", "pending"),
                    "audit_score": audit["score"],
                    "audit_flags": ";".join(audit["flags"]),
                    "reviewer": case.review.get("reviewer", ""),
                    "rejection_reason": case.review.get("rejection_reason", ""),
                    "notes": case.review.get("notes", ""),
                }
            )


def import_reviews(cases: Sequence[CuratedCase], review_csv: Path) -> list[CuratedCase]:
    by_id = {case.case_id: case for case in cases}
    seen: set[str] = set()
    with review_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = row.get("case_id", "")
            if case_id not in by_id:
                raise CuratedDatasetError(f"review references unknown case: {case_id}")
            status = row.get("status", "").strip().lower()
            if status not in {"pending", "approved", "rejected"}:
                raise CuratedDatasetError(f"{case_id}: invalid review status {status!r}")
            case = by_id[case_id]
            case.review = {
                "status": status,
                "reviewer": row.get("reviewer", "").strip(),
                "rejection_reason": row.get("rejection_reason", "").strip(),
                "notes": row.get("notes", "").strip(),
            }
            if status == "approved" and not case.review["reviewer"]:
                raise CuratedDatasetError(f"{case_id}: approved review requires reviewer")
            seen.add(case_id)
    missing = set(by_id) - seen
    if missing:
        raise CuratedDatasetError(f"review CSV omits {len(missing)} candidate cases")
    return list(cases)


def choose_accepted_cases(cases: Sequence[CuratedCase], split: str, *, seed: int) -> list[CuratedCase]:
    targets = SPLIT_TARGETS[split]
    approved = [case for case in cases if case.review.get("status") == "approved"]
    random.Random(seed).shuffle(approved)
    accepted: list[CuratedCase] = []
    for case_class, target in targets.items():
        pool = [case for case in approved if case.case_class == case_class]
        # Alternate domains where possible, while retaining deterministic order.
        domains: dict[str, list[CuratedCase]] = defaultdict(list)
        for case in pool:
            domains[case.domain].append(case)
        picked: list[CuratedCase] = []
        while len(picked) < target and any(domains.values()):
            for domain in sorted(domains):
                if domains[domain] and len(picked) < target:
                    picked.append(domains[domain].pop())
        if len(picked) < target:
            raise CuratedDatasetError(
                f"{split}/{case_class}: need {target} approved cases, found {len(pool)}"
            )
        accepted.extend(picked)
    return sorted(accepted, key=lambda case: case.case_id)


def validate_case(
    case: CuratedCase,
    passages: Mapping[str, Passage],
    *,
    require_test_review: bool = True,
) -> list[str]:
    problems: list[str] = []
    prefix = case.case_id or "<missing-case-id>"
    if case.schema_version != SCHEMA_VERSION:
        problems.append(f"{prefix}: unsupported schema_version {case.schema_version!r}")
    if case.split not in SPLIT_TARGETS:
        problems.append(f"{prefix}: invalid split {case.split!r}")
    if case.domain not in {"cloud", "govt"}:
        problems.append(f"{prefix}: invalid domain {case.domain!r}")
    if case.case_class not in CASE_CLASSES:
        problems.append(f"{prefix}: invalid case_class {case.case_class!r}")
    elif case.expected_behavior != EXPECTED_BEHAVIOURS[case.case_class]:
        problems.append(f"{prefix}: expected_behavior does not match case_class")
    if not case.query.strip():
        problems.append(f"{prefix}: empty query")
    split_seed = int(case.generation.get("seed", 20260928))
    if assigned_split(case.source_document_group, split_seed) != case.split and case.provenance.get("source") == "corpus_generated":
        problems.append(f"{prefix}: source document hashes to another split")
    if set(case.gold_passage_ids) & set(case.distractor_passage_ids):
        problems.append(f"{prefix}: a gold passage is also a distractor")
    for passage_id in case.gold_passage_ids + case.distractor_passage_ids:
        if passage_id not in passages:
            problems.append(f"{prefix}: unknown passage id {passage_id}")
    for span in case.support_spans:
        passage = passages.get(span.passage_id)
        if passage is None:
            problems.append(f"{prefix}: span references unknown passage {span.passage_id}")
            continue
        if not (0 <= span.start <= span.end <= len(passage.text)):
            problems.append(f"{prefix}: invalid offsets for {span.passage_id}")
        elif passage.text[span.start : span.end] != span.text:
            problems.append(f"{prefix}: support span does not match {span.passage_id}")
        if span.passage_id not in case.gold_passage_ids:
            problems.append(f"{prefix}: support span passage is not in gold_passage_ids")
    if case.case_class in {"answerable", "partial"}:
        if not case.reference_answer.strip() or not case.support_spans:
            problems.append(f"{prefix}: grounded case requires answer and support spans")
    if case.case_class == "underspecified" and not case.required_clarification.strip():
        problems.append(f"{prefix}: underspecified case requires a clarification")
    if case.case_class == "unanswerable" and case.reference_answer.strip():
        problems.append(f"{prefix}: unanswerable case must not contain a reference answer")
    if require_test_review and case.split == "test" and case.review.get("status") != "approved":
        problems.append(f"{prefix}: test case lacks approved human review")
    if (
        case.split == "test"
        and case.case_class in {"unanswerable", "underspecified"}
        and case.provenance.get("source") != "mtragun"
    ):
        problems.append(f"{prefix}: test negative must preserve MTRAG-UN provenance")
    return problems


def validate_dataset(
    cases_by_split: Mapping[str, Sequence[CuratedCase]],
    passages: Mapping[str, Passage],
    *,
    require_targets: bool = True,
    require_test_review: bool = True,
) -> dict[str, object]:
    problems: list[str] = []
    seen_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, dict[str, int]] = {}
    domains: dict[str, dict[str, int]] = {}
    tag_counts: dict[str, dict[str, int]] = {}
    for split, cases in cases_by_split.items():
        class_counts = Counter(case.case_class for case in cases)
        counts[split] = dict(sorted(class_counts.items()))
        domains[split] = dict(sorted(Counter(case.domain for case in cases).items()))
        tag_counts[split] = dict(sorted(Counter(tag for case in cases for tag in case.tags).items()))
        if require_targets and dict(class_counts) != SPLIT_TARGETS[split]:
            problems.append(f"{split}: class counts {dict(class_counts)} != {SPLIT_TARGETS[split]}")
        for case in cases:
            if case.case_id in seen_ids:
                problems.append(f"duplicate case_id: {case.case_id}")
            seen_ids.add(case.case_id)
            group_splits[case.source_document_group].add(split)
            problems.extend(
                validate_case(case, passages, require_test_review=require_test_review)
            )
    for group, splits in group_splits.items():
        if len(splits) > 1:
            problems.append(f"source-document leakage: {group} appears in {sorted(splits)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not problems,
        "total_cases": sum(len(cases) for cases in cases_by_split.values()),
        "counts": counts,
        "domains": domains,
        "tags": tag_counts,
        "problems": problems,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(data_dir: Path, report: Mapping[str, object], *, seed: int) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for split in SPLIT_TARGETS:
        path = cases_path(data_dir, split)
        if path.exists():
            files[split] = {
                "path": str(path.relative_to(data_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "frozen_test_sha256": files.get("test", {}).get("sha256"),
        "targets": SPLIT_TARGETS,
        "files": files,
        "validation": dict(report),
        "disclosure": (
            "Corpus-derived curated suite; not an official MTRAG-UN benchmark. "
            "Report its metrics separately from untouched MTRAG-UN results."
        ),
    }
