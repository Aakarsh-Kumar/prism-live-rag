from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .corpus_clean import clean_file
from .models import Passage, QueryTask
from .stream import SimulatedStream


DOMAIN_TO_COLLECTION = {"cloud": "ibmcloud", "govt": "govt"}


class DatasetError(RuntimeError):
    pass


def corpus_path(data_dir: Path, domain: str) -> Path:
    return data_dir / "corpora" / "passage_level" / f"{domain}.jsonl"


def qrels_path(data_dir: Path, domain: str) -> Path:
    return data_dir / "mtragun-human" / "retrieval_tasks" / "qrels" / f"{domain}.tsv"


def reference_path(data_dir: Path) -> Path:
    return data_dir / "mtragun-human" / "generation_tasks" / "reference.jsonl"


def streams_path(data_dir: Path, domain: str) -> Path:
    return data_dir / "simulated_streams" / f"{domain}.jsonl"


def load_streams(data_dir: Path, domain: str) -> list[SimulatedStream]:
    path = streams_path(data_dir, domain)
    if not path.exists():
        raise DatasetError(f"Missing simulated streams file: {path}")
    streams: list[SimulatedStream] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            streams.append(SimulatedStream.from_dict(json.loads(line)))
    return streams


def save_streams(data_dir: Path, domain: str, streams: list[SimulatedStream]) -> Path:
    path = streams_path(data_dir, domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for stream in streams:
            handle.write(json.dumps(stream.to_dict(), ensure_ascii=False) + "\n")
    return path


def iter_passages(
    data_dir: Path,
    domain: str,
    limit: int | None = None,
    *,
    clean: bool = True,
    protected_ids: Iterable[str] | None = None,
) -> Iterable[Passage]:
    """Yield passages for a domain, cleaned per ``docs/corpus-spec.md`` §3.

    The on-disk corpus files stay byte-identical to the official MTRAG-UN distribution;
    cleaning happens here at load time so the index gets clean text while the corpus
    remains citable. Pass ``clean=False`` to read the raw rows.

    By default the domain's qrel passage IDs are loaded and protected from
    duplicate-removal, so a gold passage is never deduplicated away and stays
    retrievable. This is deliberately automatic rather than opt-in: forgetting it
    silently costs 12 qrel passages at index time. Pass an explicit iterable to
    override, or an empty one to opt out.
    """
    path = corpus_path(data_dir, domain)
    if not path.exists():
        raise DatasetError(f"Missing corpus file: {path}")

    if not clean:
        emitted = 0
        for row in _iter_raw_rows(path):
            yield _to_passage(row, domain)
            emitted += 1
            if limit is not None and emitted >= limit:
                return
        return

    if protected_ids is None:
        gold = frozenset(
            pid for ids in load_qrels(data_dir, domain).values() for pid in ids
        )
    else:
        gold = frozenset(protected_ids)

    for row in clean_file(path, domain, gold_ids=gold, limit=limit):
        yield _to_passage(row, domain)


def _iter_raw_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _to_passage(row: dict, domain: str) -> Passage:
    cleaning = row.get("cleaning")
    flattened: tuple[tuple[str, str], ...] = ()
    if isinstance(cleaning, dict):
        # Values are coerced to str: ``stripped`` is a list and ``chrome_host`` can be
        # None, neither of which would keep Passage hashable.
        flattened = tuple(
            sorted(
                (str(key), "" if value is None else str(value))
                for key, value in cleaning.items()
            )
        )
    return Passage(
        id=str(row["_id"]),
        domain=domain,
        text=str(row.get("text", "")),
        title=str(row.get("title", "")),
        url=str(row.get("url", "")),
        links=tuple(row.get("links") or ()),
        text_raw_sha256=str(row.get("text_raw_sha256", "")),
        dropped_duplicate_of=str(row.get("dropped_duplicate_of", "")),
        cleaning=flattened,
    )


def load_qrels(data_dir: Path, domain: str) -> dict[str, list[str]]:
    path = qrels_path(data_dir, domain)
    if not path.exists():
        raise DatasetError(f"Missing qrels file: {path}")
    qrels: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels[row["query-id"]].append(row["corpus-id"])
    return dict(qrels)


def load_query_tasks(data_dir: Path, domain: str) -> list[QueryTask]:
    collection = DOMAIN_TO_COLLECTION[domain]
    qrels = load_qrels(data_dir, domain)
    tasks: list[QueryTask] = []
    with reference_path(data_dir).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if row.get("Collection") != collection or task_id not in qrels:
                continue
            user_turns = [
                turn.get("text", "")
                for turn in row.get("input", [])
                if turn.get("speaker") == "user"
            ]
            if not user_turns:
                continue
            tasks.append(
                QueryTask(
                    task_id=task_id,
                    domain=domain,
                    query=user_turns[-1],
                    answerability=tuple(row.get("answerability", [])),
                    qrel_passage_ids=tuple(qrels[task_id]),
                )
            )
    return tasks


def validate_domain(data_dir: Path, domain: str) -> dict[str, int]:
    tasks = load_query_tasks(data_dir, domain)
    qrel_ids = {pid for task in tasks for pid in task.qrel_passage_ids}
    # Gold IDs must be resolved *before* the corpus is loaded, so duplicate removal knows
    # which copies of a duplicated passage to protect.
    passage_ids = {p.id for p in iter_passages(data_dir, domain, protected_ids=qrel_ids)}
    missing = qrel_ids - passage_ids
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise DatasetError(f"{domain} qrels reference missing passages: {examples}")
    raw_count = sum(1 for _ in _iter_raw_rows(corpus_path(data_dir, domain)))
    return {
        "tasks": len(tasks),
        "qrel_passages": len(qrel_ids),
        "corpus_passages": len(passage_ids),
        "corpus_passages_raw": raw_count,
        "corpus_passages_dropped": raw_count - len(passage_ids),
    }

