#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_BASE_URL = "https://github.com/IBM/mt-rag-benchmark/raw/refs/heads/main/corpora/passage_level"
DOMAINS = ("cloud", "govt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the Cloud/Govt passage-level corpora.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", default=None, help="Optional local mirror/root dir or URL base.")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted JSONL files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_dir = args.data_dir / "corpora" / "passage_level"
    target_dir.mkdir(parents=True, exist_ok=True)
    source = args.source or DEFAULT_BASE_URL
    for domain in args.domains:
        target = target_dir / f"{domain}.jsonl"
        if target.exists() and not args.force:
            print(f"ok: {target} exists")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / f"{domain}.jsonl.zip"
            _copy_or_download_zip(source, domain, archive)
            _extract_jsonl(archive, target, domain)
            _validate_nonempty(target)
            print(f"ok: wrote {target}")
    return 0


def _copy_or_download_zip(source: str, domain: str, archive: Path) -> None:
    if source.startswith(("http://", "https://")):
        url = f"{source.rstrip('/')}/{domain}.jsonl.zip"
        print(f"download: {url}")
        try:
            urllib.request.urlretrieve(url, archive)
        except Exception as exc:  # noqa: BLE001 - turn network failures into setup guidance.
            raise SystemExit(f"failed to download {url}: {exc}") from exc
        return

    root = Path(source)
    candidates = [
        root / "corpora" / "passage_level" / f"{domain}.jsonl.zip",
        root / "passage_level" / f"{domain}.jsonl.zip",
        root / f"{domain}.jsonl.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            shutil.copyfile(candidate, archive)
            print(f"copy: {candidate}")
            return
    raise SystemExit(
        f"missing {domain}.jsonl.zip. Provide --source pointing to a local mt-rag-benchmark clone "
        "or allow the default GitHub download URL."
    )


def _extract_jsonl(archive: Path, target: Path, domain: str) -> None:
    expected = f"{domain}.jsonl"
    with zipfile.ZipFile(archive) as handle:
        members = [name for name in handle.namelist() if name.endswith(expected)]
        if not members:
            raise SystemExit(f"{archive} does not contain {expected}")
        with handle.open(members[0]) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _validate_nonempty(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"extracted file is empty: {path}")


if __name__ == "__main__":
    sys.exit(main())
