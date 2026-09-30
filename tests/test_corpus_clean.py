"""Tests for loader-side corpus cleaning (docs/corpus-spec.md §3)."""

import json
from pathlib import Path

from prism_live_rag.corpus_clean import (
    clean_row,
    extract_links,
    is_crawl_pagination,
    normalize_key,
    scan,
    text_digest,
)
from prism_live_rag.data import corpus_path, iter_passages, load_query_tasks, validate_domain

DATA_DIR = Path("data")

RAW = {
    "_id": "a1",
    "url": "https://cloud.ibm.com/docs/x?topic=x",
    "title": "",
    "text": "See [the setup guide](https://cloud.ibm.com/docs/setup) for details.\n\n\n\nMore prose here.",
}


def _scan_rows(rows, gold=frozenset()):
    return scan(iter(rows), gold)


def test_markdown_links_move_out_of_prose_into_links() -> None:
    text, links = extract_links(RAW["text"])
    assert links == ["https://cloud.ibm.com/docs/setup"]
    assert "https://cloud.ibm.com/docs/setup" not in text
    assert "the setup guide" in text
    assert "More prose here." in text


def test_bare_urls_are_extracted() -> None:
    text, links = extract_links("Visit https://example.com/a/b for the docs.")
    assert links == ["https://example.com/a/b"]
    assert "https://" not in text


def test_crawl_pagination_urls_are_recognised() -> None:
    assert is_crawl_pagination(
        "https://webarchive.library.unt.edu/eot2008/query?q=date&count=10000&start_page=1"
    )
    assert not is_crawl_pagination("https://www.va.gov/your-benefits/")


def test_crawl_pagination_rows_are_dropped() -> None:
    rows = [
        {
            "_id": "noise",
            "url": "https://webarchive.library.unt.edu/eot2008/query?q=x&count=10000",
            "text": "UNT Web Archive Enter URL: All 2024 2023",
        },
        {"_id": "keep", "url": "https://www.va.gov/faq/", "text": "Real question answer."},
    ]
    result = _scan_rows(rows)
    assert clean_row(rows[0], "govt", result) is None
    assert clean_row(rows[1], "govt", result) is not None


def test_duplicates_are_dropped_but_gold_copies_survive() -> None:
    rows = [
        {"_id": "dup2", "url": "https://www.va.gov/a", "text": "Same answer text."},
        {"_id": "keep", "url": "https://www.va.gov/b", "text": "Different answer text."},
    ]
    result = _scan_rows(rows)
    assert clean_row(rows[0], "govt", result) is not None
    assert clean_row(rows[1], "govt", result) is not None

    # Same group, but the later row is gold: it must be the one retained.
    rows_gold = [
        {"_id": "nongold", "url": "https://www.va.gov/a", "text": "Same answer text."},
        {"_id": "gold", "url": "https://www.va.gov/b", "text": "Same answer text."},
    ]
    result_gold = _scan_rows(rows_gold, frozenset({"gold"}))
    assert result_gold.duplicate_of[text_digest("Same answer text.")] == "gold"


def test_clean_row_emits_spec_provenance() -> None:
    result = _scan_rows([RAW])
    cleaned = clean_row(RAW, "cloud", result)
    assert cleaned is not None
    assert cleaned["_id"] == "a1"
    assert cleaned["domain"] == "cloud"
    assert len(cleaned["text_raw_sha256"]) == 64
    assert cleaned["cleaning"]["stripped"] == ["markdown_links", "whitespace"]
    assert cleaned["links"] == ["https://cloud.ibm.com/docs/setup"]
    assert cleaned["cleaning"]["chars_removed"] > 0


def test_link_extraction_guarding_is_not_inverted() -> None:
    """Regression: the MIN_LINK_PROSE_RATIO guard was inverted.

    It reverted extraction when prose was plentiful and applied it when links dominated,
    so 63.5% of cloud passages kept their URLs inline and the '18.8% of characters'
    claim was false. Harvesting must happen in normal passages and be suppressed only
    when it would leave a stub.
    """
    prose_heavy = {
        "_id": "a", "url": "https://www.va.gov/x",
        "text": ("To apply you must be a resident and have income documents. "
                 "See [the full checklist](https://www.va.gov/checklist) and submit."),
    }
    link_heavy = {
        "_id": "b", "url": "https://www.va.gov/y",
        "text": "[A](https://a.gov/1)[B](https://a.gov/2)[C](https://a.gov/3)",
    }
    result = _scan_rows([prose_heavy, link_heavy])

    cleaned = clean_row(prose_heavy, "govt", result)
    assert cleaned["links"] == ["https://www.va.gov/checklist"]
    assert "https://" not in cleaned["text"]
    assert "the full checklist" in cleaned["text"]

    stub = clean_row(link_heavy, "govt", result)
    assert stub["links"] == []
    assert stub["text"] == link_heavy["text"]  # untouched, not reduced to "ABC"


def test_normalize_key_and_digest_ignore_whitespace() -> None:
    assert normalize_key("a  b\n c") == "a b c"
    assert text_digest("a  b") == text_digest("a\nb")


def test_chrome_is_stripped_per_host_not_globally() -> None:
    """A line that is chrome on one host must survive on a host that never emits it."""
    # Emitted by 9 of 13 nasa.gov passages -> 69% document frequency, above the 0.60
    # threshold, so it is chrome for that host. va.gov never emits it at all.
    filler = [
        {"_id": f"n{i}", "url": "https://www.nasa.gov/a",
         "text": "Skip to main content\n" if i < 9 else f"body {i}"}
        for i in range(12)
    ]
    chrome_row = {"_id": "c1", "url": "https://www.nasa.gov/a", "text": "Skip to main content\nReal answer."}
    keep_row = {"_id": "k1", "url": "https://www.va.gov/a", "text": "Skip to main content\nReal answer."}
    result = _scan_rows(filler + [chrome_row, keep_row])
    assert "Skip to main content" in result.chrome.get("www.nasa.gov", set())
    assert "www.va.gov" not in result.chrome
    assert clean_row(chrome_row, "govt", result)["text"] == "Real answer."
    # va.gov never emits the line, so it is content there and must survive. This is the
    # whole point of modelling chrome per host.
    va_kept = clean_row(keep_row, "govt", result)
    assert va_kept["text"] == "Skip to main content\nReal answer."
    assert "nav_chrome" not in va_kept["cleaning"]["stripped"]


def test_whitespace_normalisation_preserves_paragraph_structure() -> None:
    """Regression: ``\\s+`` matched newlines and flattened every passage to one line.

    That discarded the paragraph structure raw MTRAG carries and made the blank-line
    collapse dead code.
    """
    from prism_live_rag.corpus_clean import _normalize_whitespace

    raw = "Line one.\n\n\n\nLine two.\nLine three."
    out = _normalize_whitespace(raw)
    assert out == "Line one.\n\nLine two.\nLine three."
    assert "\n" in out
    assert _normalize_whitespace("  A  b \t c  \n\n\n  D  ") == "A b c\n\nD"


def test_gold_qrel_ids_always_survive_cleaning() -> None:
    """The 504/504 join is the repo's most defensible asset; cleaning must not dent it."""
    for domain in ("cloud", "govt"):
        if not corpus_path(DATA_DIR, domain).exists():
            continue
        gold = frozenset(
            pid for task in load_query_tasks(DATA_DIR, domain) for pid in task.qrel_passage_ids
        )
        kept = {p.id for p in iter_passages(DATA_DIR, domain, protected_ids=gold)}
        assert gold <= kept, f"{domain}: {len(gold - kept)} gold passages lost"


def test_gold_protection_is_on_by_default() -> None:
    """12 qrel passages share text with another qrel passage. Protection must be
    automatic, not opt-in: forgetting it silently drops them from the index."""
    unprotected_total = 0
    protected_total = 0
    for domain in ("cloud", "govt"):
        if not corpus_path(DATA_DIR, domain).exists():
            continue
        gold = frozenset(
            pid for task in load_query_tasks(DATA_DIR, domain) for pid in task.qrel_passage_ids
        )
        default_ids = {p.id for p in iter_passages(DATA_DIR, domain)}
        explicit_ids = {p.id for p in iter_passages(DATA_DIR, domain, protected_ids=gold)}
        unprotected_ids = {p.id for p in iter_passages(DATA_DIR, domain, protected_ids=())}
        assert default_ids == explicit_ids
        assert gold <= default_ids
        protected_total += len(default_ids)
        unprotected_total += len(unprotected_ids)
    assert protected_total - unprotected_total == 12


def test_validate_domain_reports_raw_and_cleaned_counts() -> None:
    for domain in ("cloud", "govt"):
        if not corpus_path(DATA_DIR, domain).exists():
            continue
        stats = validate_domain(DATA_DIR, domain)
        assert stats["corpus_passages"] <= stats["corpus_passages_raw"]
        assert stats["corpus_passages_dropped"] >= 0
        assert stats["corpus_passages"] > stats["qrel_passages"]


def test_raw_mode_is_available_and_unchanged() -> None:
    if not corpus_path(DATA_DIR, "cloud").exists():
        return
    first = next(iter_passages(DATA_DIR, "cloud", limit=1, clean=False))
    assert first.id
    assert first.links == ()
    assert first.cleaning == ()


def test_passage_with_provenance_is_hashable() -> None:
    """``Passage.cleaning`` is documented as hashable; a list value would break that."""
    from prism_live_rag.data import _to_passage

    passage = _to_passage(
        {
            "_id": "x",
            "text": "Body.",
            "url": "https://www.va.gov/a",
            "links": ["https://a/1"],
            "text_raw_sha256": "0" * 64,
            "cleaning": {
                "stripped": ["nav_chrome", "markdown_links"],
                "chrome_host": "www.va.gov",
                "chars_removed": 412,
            },
        },
        "govt",
    )
    assert hash(passage)
    assert {passage}  # must be usable in a set
    assert dict(passage.cleaning) == {
        "chars_removed": "412",
        "chrome_host": "www.va.gov",
        "stripped": "['nav_chrome', 'markdown_links']",
    }
    assert passage.text_raw_sha256 == "0" * 64


def test_dropped_duplicate_of_records_the_displaced_id(tmp_path) -> None:
    """A gold duplicate kept in place of an incumbent records what it replaced."""
    from prism_live_rag.corpus_clean import clean_file

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"_id": "nongold", "url": "https://www.va.gov/a", "text": "Identical body."},
                {"_id": "gold", "url": "https://www.va.gov/b", "text": "Identical body."},
            )
        ),
        encoding="utf-8",
    )
    rows = list(clean_file(corpus, "govt", gold_ids=frozenset({"gold"})))
    assert [r["_id"] for r in rows] == ["gold"]
    assert rows[0]["dropped_duplicate_of"] == "nongold"
