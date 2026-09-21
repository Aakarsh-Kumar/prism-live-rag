from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import Passage, QueryTask


DOMAIN_TO_COLLECTION = {"cloud": "ibmcloud", "govt": "govt"}


class DatasetError(RuntimeError):
    pass


def corpus_path(data_dir: Path, domain: str) -> Path:
    return data_dir / "corpora" / "passage_level" / f"{domain}.jsonl"


def qrels_path(data_dir: Path, domain: str) -> Path:
    return data_dir / "mtragun-human" / "retrieval_tasks" / "qrels" / f"{domain}.tsv"


def reference_path(data_dir: Path) -> Path:
    return data_dir / "mtragun-human" / "generation_tasks" / "reference.jsonl"


def iter_passages(data_dir: Path, domain: str, limit: int | None = None) -> Iterable[Passage]:
    path = corpus_path(data_dir, domain)
    if not path.exists():
        raise DatasetError(f"Missing corpus file: {path}")
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield Passage(
                id=str(row["_id"]),
                domain=domain,
                text=str(row.get("text", "")),
                title=str(row.get("title", "")),
                url=str(row.get("url", "")),
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return


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
    passage_ids = {passage.id for passage in iter_passages(data_dir, domain)}
    qrel_ids = {pid for task in tasks for pid in task.qrel_passage_ids}
    missing = qrel_ids - passage_ids
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise DatasetError(f"{domain} qrels reference missing passages: {examples}")
    return {
        "tasks": len(tasks),
        "qrel_passages": len(qrel_ids),
        "corpus_passages": len(passage_ids),
    }

