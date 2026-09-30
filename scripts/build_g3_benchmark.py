"""Build a deterministic, review-pending G3 benchmark candidate file."""

from __future__ import annotations

import hashlib
import json
import random
import re
import csv
from pathlib import Path

from prism_live_rag.curated_dataset import load_cases


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/curated_dataset/decomposition/g3-candidates.jsonl"
REVIEW_OUTPUT = ROOT / "data/curated_dataset/decomposition/g3-review.csv"
SOURCE = ROOT / "data/curated_dataset/reviewed/test.jsonl"
SINGLE_SOURCE = ROOT / "data/curated_dataset/decomposition/single-intents.jsonl"
CONNECTORS = (
    "Also, {second}",
    "One more thing: {second}",
    "Separately, {second}",
    "And another question: {second}",
    "Could I also ask: {second}",
)
NON_SINGLE_RE = re.compile(r"\b(?:and|also|plus|while|whereas|versus|vs)\b|\?.*\?", re.I)


def stable_order(rows: list[dict], domain: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"g3-20260929:{domain}:{row['case_id']}".encode()
        ).hexdigest(),
    )


def build_chunks(text: str, seed: int) -> tuple[list[dict], dict]:
    words = text.split()
    rng = random.Random(seed)
    revision_word = rng.randrange(2, len(words) - 1) if len(words) > 8 and rng.random() < 0.2 else None
    wrong_word = None
    correction_word = None
    if revision_word is not None:
        original = words[revision_word]
        wrong_word = original[:-1] + ("x" if not original.endswith("x") else "z")
        correction_word = min(len(words), revision_word + 4)

    chunks = []
    revision_log = []
    prior_text = ""
    for end in range(4, len(words), 4):
        partial_words = words[:end]
        if revision_word is not None and revision_word < end < correction_word:
            partial_words[revision_word] = wrong_word
        chunk_text = " ".join(partial_words)
        if prior_text and prior_text != chunk_text and revision_word is not None:
            if wrong_word in chunk_text and wrong_word not in prior_text:
                revision_log.append({"chunk_index": len(chunks), "from": prior_text, "to": chunk_text})
            elif wrong_word in prior_text and wrong_word not in chunk_text:
                revision_log.append({"chunk_index": len(chunks), "from": prior_text, "to": chunk_text})
        chunks.append({
            "timestamp_s": round(len(chunks) * 0.8 + 0.4, 2),
            "text": chunk_text,
            "is_final": False,
            "confidence": round(0.72 + min(end / max(len(words), 1), 1.0) * 0.2, 2),
        })
        prior_text = chunk_text

    chunks.append({
        "timestamp_s": round(len(chunks) * 0.8 + 0.8, 2),
        "text": text,
        "is_final": True,
        "confidence": 0.98,
    })
    return chunks, {
        "engine": "simulated",
        "final_chunk_index": len(chunks) - 1,
        "supersedes": [row["chunk_index"] for row in revision_log],
        "obsoletes_prior_partials": bool(revision_log),
        "revision_log": revision_log,
    }


def main() -> None:
    reviewed = load_cases(SOURCE)
    by_domain: dict[str, list[dict]] = {"cloud": [], "govt": []}
    for case in reviewed:
        if case.review.get("status") != "approved" or NON_SINGLE_RE.search(case.query):
            continue
        by_domain[case.domain].append({"case_id": case.case_id, "query": case.query.strip()})

    original_singles = [json.loads(line) for line in SINGLE_SOURCE.read_text().splitlines() if line.strip()]
    for row in original_singles:
        by_domain[row["domain"]].append({"case_id": row["case_id"], "query": row["query"].strip()})

    singles: list[dict] = []
    for domain in ("cloud", "govt"):
        unique = {row["query"].casefold(): row for row in by_domain[domain]}
        selected = stable_order(list(unique.values()), domain)[:50]
        if len(selected) < 50:
            raise SystemExit(f"Need 50 reviewed single-intent source queries for {domain}; found {len(selected)}")
        for index, source in enumerate(selected):
            case_id = f"g3-single-{domain}-{index + 1:03d}"
            chunks, asr = build_chunks(source["query"], seed=20260929 + index)
            singles.append({
                "case_id": case_id,
                "domain": domain,
                "split": "candidate",
                "category": "single_intent",
                "query": source["query"],
                "expected_intents": [source["query"]],
                "chunks": chunks,
                "asr": asr,
                "provenance": {"source_case_id": source["case_id"], "source": "owner-approved G4 query; stream simulated"},
                "review_status": "pending_g3_review",
            })

    compounds: list[dict] = []
    by_domain_singles = {
        domain: [row for row in singles if row["domain"] == domain]
        for domain in ("cloud", "govt")
    }
    for domain, rows in by_domain_singles.items():
        for index, first in enumerate(rows):
            second = rows[(index + 7) % len(rows)]
            first_query = first["query"].rstrip()
            if first_query.endswith("."):
                first_query = first_query[:-1] + "?"
            elif first_query[-1:] not in "?!":
                first_query += "?"
            connector = CONNECTORS[index % len(CONNECTORS)]
            utterance = f"{first_query} {connector.format(second=second['query'].strip())}"
            case_id = f"g3-compound-{domain}-{index + 1:03d}"
            chunks, asr = build_chunks(utterance, seed=20300000 + index)
            compounds.append({
                "case_id": case_id,
                "domain": domain,
                "split": "candidate",
                "category": "compound",
                "query": utterance,
                "expected_intents": [first["query"], second["query"]],
                "chunks": chunks,
                "asr": asr,
                "provenance": {
                    "source_case_ids": [first["provenance"]["source_case_id"], second["provenance"]["source_case_id"]],
                    "source": "composed from owner-approved G4 single-intent queries; connector and stream simulated",
                },
                "review_status": "pending_g3_review",
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in compounds + singles:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not REVIEW_OUTPUT.exists():
        with REVIEW_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["case_id", "domain", "category", "query", "expected_intents", "review_status", "review_notes"],
            )
            writer.writeheader()
            for row in compounds + singles:
                writer.writerow({
                    "case_id": row["case_id"],
                    "domain": row["domain"],
                    "category": row["category"],
                    "query": row["query"],
                    "expected_intents": json.dumps(row["expected_intents"], ensure_ascii=False),
                    "review_status": "pending_g3_review",
                    "review_notes": "",
                })
    print(json.dumps({"path": str(OUTPUT), "review_csv": str(REVIEW_OUTPUT), "compounds": len(compounds), "single_controls": len(singles), "review_status": "pending_g3_review"}, indent=2))


if __name__ == "__main__":
    main()
