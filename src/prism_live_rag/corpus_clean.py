"""Loader-side corpus cleaning.

Implements the cleaning operations in ``docs/corpus-spec.md`` §3.

Design rule (spec §0.5): the on-disk ``data/corpora/passage_level/*.jsonl`` files stay
byte-identical to the official MTRAG-UN distribution. Everything here happens at load
time, so the corpus remains citable while the index still gets clean text.

Four operations, all exact-match or frequency-based so every removal is reproducible and
stateable as a one-line rule:

1. Drop web-archive crawl-pagination records (govt: 20.5% of the domain, 0 gold).
2. Drop byte-exact duplicates, never dropping a gold-referenced copy (cloud: 13.7%).
3. Extract markdown links / bare URLs out of prose into a ``links`` field.
4. Strip per-host navigation chrome (govt: 7.8% of characters, 0 of 256 gold at risk).

Chrome detection is deliberately per-URL-host rather than global: a global blocklist
catches almost nothing, because the boilerplate differs per site (``www.va.gov`` has
none; ``www.nasa.gov`` emits ``Highlights`` / ``4 min read``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

# --- tunables (all justified by measurement in docs/corpus-spec.md §3) ---------------

#: Hosts serving paginated crawl-index pages whose "passages" are navigation records
#: (year-filter dropdowns, raw query parameters) rather than document content.
CRAWL_PAGINATION_PATTERNS = (
    re.compile(r"webarchive\.library\.unt\.edu/eot\d+/query", re.IGNORECASE),
)

#: A host needs this many passages before we trust a frequency estimate against it.
MIN_PASSAGES_FOR_CHROME_MODEL = 10

#: A line of this length or longer is content, never chrome.
MAX_CHROME_LINE_CHARS = 60

#: Fraction of a host's passages a line must appear in to count as that host's chrome.
#:
#: Chosen by sweeping this threshold against BM25 gold recall on the 191 qrel tasks, not
#: by taste. At 0.30 stripping cost 1.6 points of R@1 and R@10 -- the low bar was
#: deleting lines that are genuine content on a minority of pages. Recall at 0.50+ is
#: indistinguishable from not stripping at all, and 0.90 == chrome-off exactly. 0.60
#: keeps the real recall while still removing the worst boilerplate:
#:
#:     chrome off   R@1=23.6  R@10=47.1  R@100=69.6
#:     DF >= 0.30   R@1=22.0  R@10=46.6  R@100=70.2
#:     DF >= 0.50   R@1=23.6  R@10=46.1  R@100=69.1
#:     DF >= 0.60   R@1=23.6  R@10=46.6  R@100=69.6   <- chosen
#:     DF >= 0.75   R@1=23.6  R@10=47.1  R@100=69.6
CHROME_DOCUMENT_FREQUENCY = 0.60

#: Refuse to strip a passage down to less than this fraction of its original length.
MIN_TEXT_RETAINED = 0.30

#: Refuse to harvest links when that would leave less prose than this fraction of the
#: passage. Same "don't gut it" purpose as MIN_TEXT_RETAINED, so it uses the same
#: threshold; 0.60 here rejected ordinary passages that kept ~57% of their prose.
MIN_LINK_PROSE_RATIO = 0.30

_MARKDOWN_LINK = re.compile(r"\[([^\]\n]*)\]\(([^)\s]*)\)")
_BARE_URL = re.compile(r"(?<![(\[])https?://\S+")
#: Horizontal whitespace only. Deliberately not ``\s+``: that also matched newlines and
#: flattened every cleaned passage to a single line, which discarded the paragraph
#: structure the raw MTRAG text carries and made the ``_BLANK_RUN`` step dead code.
_HSPACE = re.compile(r"[ \t]+")
#: Three or more newlines (with optional trailing spaces) collapse to a single blank line.
_BLANK_RUN = re.compile(r"\n[ \t]*(?:\n[ \t]*){2,}")
_WS = re.compile(r"\s+")


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def normalize_key(text: str) -> str:
    """Whitespace-normalised text, used to detect byte-exact duplicates."""
    return _WS.sub(" ", text).strip()


def text_digest(text: str) -> str:
    """Stable short digest of normalised text. Used as the dedup map key so we never
    hold 122k full strings in memory during the first pass."""
    return hashlib.sha1(normalize_key(text).encode("utf-8")).hexdigest()


def is_crawl_pagination(url: str) -> bool:
    return any(pattern.search(url) for pattern in CRAWL_PAGINATION_PATTERNS)


def _normalize_whitespace(text: str) -> str:
    """Tidy ragged whitespace while preserving paragraph structure.

    Horizontal runs collapse to one space, blank-line runs to a single blank line, and
    single newlines survive as line breaks. Kept separate from :func:`extract_links` so
    ``chars_removed`` can attribute the characters honestly.
    """
    collapsed = _BLANK_RUN.sub("\n\n", text)
    collapsed = _HSPACE.sub(" ", collapsed)
    return "\n".join(line.strip() for line in collapsed.split("\n")).strip()


def extract_links(text: str) -> tuple[str, list[str]]:
    """Pull markdown links and bare URLs out of prose.

    A URL is real information about a product page even when it is noise inside a
    sentence, so it is kept in ``links`` rather than deleted. The visible link text is
    retained in the prose. Whitespace is deliberately left untouched here; see
    :func:`_normalize_whitespace`.
    """
    links: list[str] = []

    def take_markdown(match: re.Match[str]) -> str:
        links.append(match.group(2))
        return match.group(1)

    stripped = _MARKDOWN_LINK.sub(take_markdown, text)
    return _BARE_URL.sub(lambda m: links.append(m.group(0)) or " ", stripped), links


def _is_line_candidate(line: str) -> bool:
    stripped = line.strip()
    return 2 <= len(stripped) <= MAX_CHROME_LINE_CHARS


class _ScanResult:
    """What the first pass learns about a corpus file."""

    def __init__(self) -> None:
        self.chrome: dict[str, set[str]] = {}
        self.duplicate_of: dict[str, str] = {}
        self.crawl_pagination: set[str] = set()
        #: kept id -> the non-gold id it displaced. ``duplicate_of`` is redirected to the
        #: gold copy, so the loser is only knowable here, not from ``duplicate_of``.
        self.displaced: dict[str, str] = {}


def scan(rows: Iterable[dict], gold_ids: frozenset[str] = frozenset()) -> _ScanResult:
    """First pass: learn chrome lines per host and the duplicate map.

    ``gold_ids`` are never dropped: when several passages share identical text, the
    gold-referenced one is the copy we keep.
    """
    result = _ScanResult()
    per_host_lines: dict[str, Counter[str]] = defaultdict(Counter)
    host_counts: Counter[str] = Counter()
    # normalized-digest -> the id we keep. First writer wins, unless a later writer is
    # gold-referenced and the incumbent is not.
    first_by_digest: dict[str, str] = {}
    gold_by_digest: dict[str, str] = {}

    for row in rows:
        pid = str(row.get("_id", ""))
        url = str(row.get("url", ""))
        if is_crawl_pagination(url):
            result.crawl_pagination.add(pid)
            continue
        host = host_of(url)
        host_counts[host] += 1
        for line in set(l.strip() for l in str(row.get("text", "")).split("\n")):
            if _is_line_candidate(line):
                per_host_lines[host][line] += 1

        digest = text_digest(str(row.get("text", "")))
        if not digest:
            continue
        if digest in first_by_digest:
            if pid in gold_ids and digest not in gold_by_digest:
                gold_by_digest[digest] = pid
        else:
            first_by_digest[digest] = pid
            if pid in gold_ids:
                gold_by_digest[digest] = pid

    for host, counts in per_host_lines.items():
        if host_counts[host] < MIN_PASSAGES_FOR_CHROME_MODEL:
            continue
        threshold = host_counts[host] * CHROME_DOCUMENT_FREQUENCY
        lines = {line for line, count in counts.items() if count >= threshold}
        if lines:
            result.chrome[host] = lines

    # Anything sharing a digest with the kept copy, other than the copy itself, is a dupe.
    # Prefer the gold-referenced copy when a duplicate group contains one.
    for digest, kept in first_by_digest.items():
        winner = gold_by_digest.get(digest, kept)
        result.duplicate_of[digest] = winner
        if winner != kept:
            result.displaced[winner] = kept

    return result


def _strip_chrome(text: str, chrome: set[str]) -> tuple[str, bool]:
    if not chrome:
        return text, False
    kept = [line for line in text.split("\n") if line.strip() not in chrome]
    # Chrome almost always sits inside a run of blank lines, so removing it leaves
    # ragged gaps. Collapse the leftovers instead of preserving the hole.
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    if not cleaned or len(cleaned) < len(text) * MIN_TEXT_RETAINED:
        return text, False
    return cleaned, True


def clean_row(
    row: dict,
    domain: str,
    result: _ScanResult,
) -> dict | None:
    """Apply all four operations to one raw corpus row.

    Returns ``None`` if the row is dropped, otherwise the canonical cleaned dict from
    spec §3.5.
    """
    pid = str(row.get("_id", ""))
    if pid in result.crawl_pagination:
        return None

    raw_text = str(row.get("text", ""))
    stripped: list[str] = []
    chars_removed = 0

    # Order matters: chrome detection is line-based, and link extraction normalises
    # whitespace, which would collapse the newlines that chrome detection keys on. Strip
    # boilerplate off the raw line structure first, then harvest links from what is left.
    host = host_of(str(row.get("url", "")))
    chrome = result.chrome.get(host, set())
    text, did_strip = _strip_chrome(raw_text, chrome)
    if did_strip:
        stripped.append("nav_chrome")
        chars_removed += len(raw_text) - len(text)

    link_prose, links = extract_links(text)
    if links and len(link_prose) < len(text) * MIN_LINK_PROSE_RATIO:
        # Links dominate this passage: harvesting them would leave a stub like "ABC".
        # Keep the text intact instead and record no links.
        link_prose, links = text, []
    if links:
        stripped.append("markdown_links")
        chars_removed += len(text) - len(link_prose)
    text = link_prose

    normalized = _normalize_whitespace(text)
    if normalized != text:
        stripped.append("whitespace")
        chars_removed += len(text) - len(normalized)
    text = normalized

    if not text.strip():
        return None

    return {
        "_id": pid,
        "id": pid,
        "domain": domain,
        "url": str(row.get("url", "")),
        "title": str(row.get("title", "")),
        "text": text,
        "text_raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "links": links,
        "cleaning": {
            "stripped": stripped,
            "chrome_host": host if "nav_chrome" in stripped else None,
            "chars_removed": chars_removed,
        },
    }


def iter_raw_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def clean_file(
    path: Path,
    domain: str,
    *,
    gold_ids: frozenset[str] = frozenset(),
    limit: int | None = None,
) -> Iterator[dict]:
    """Yield canonical cleaned rows from a raw MTRAG passage file.

    Two passes over the file so we never hold every passage text in memory at once.
    """
    if not path.exists():
        return
    result = scan(iter_raw_rows(path), gold_ids)

    seen_digests: set[str] = set()
    emitted = 0
    for row in iter_raw_rows(path):
        if limit is not None and emitted >= limit:
            return
        pid = str(row.get("_id", ""))
        if pid in result.crawl_pagination:
            continue
        displaced = ""
        digest = text_digest(str(row.get("text", "")))
        if digest:
            chosen = result.duplicate_of.get(digest)
            # Two gold IDs can share one identical text (the benchmark registers the same
            # passage twice). 12 such IDs exist across cloud+govt, so every gold copy is
            # kept even when it duplicates another gold copy -- dropping one would break
            # the 504/504 qrel join for no retrieval benefit.
            if chosen is not None and chosen != pid:
                if pid not in gold_ids:
                    continue
                # This gold copy displaced the incumbent; record which, for audit.
                displaced = chosen
            if digest in seen_digests and pid not in gold_ids:
                continue
            seen_digests.add(digest)
            displaced = result.displaced.get(pid, displaced)
        cleaned = clean_row(row, domain, result)
        if cleaned is None:
            continue
        if displaced:
            cleaned["dropped_duplicate_of"] = displaced
        emitted += 1
        yield cleaned
