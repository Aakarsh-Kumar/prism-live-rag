"""Create source/assets archives from explicit allowlists, excluding secrets/history."""
from __future__ import annotations
import hashlib
import argparse
import gzip
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def safe_member(member):
    if any(part in {".git", "__pycache__", ".venv"} for part in Path(member.name).parts):
        return None
    if Path(member.name).name == ".env" or member.name.endswith(".pyc"):
        return None
    return member


def write_asset_archive(destination: Path) -> None:
    def canonical_member(member):
        member = safe_member(member)
        if member is not None:
            member.mtime = 0
            member.uid = member.gid = 0
            member.uname = member.gname = ""
        return member
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                archive.add(ROOT / "submission-assets", arcname="submission-assets", filter=canonical_member)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--assets-only", action="store_true", help="Prepare the reproducible pinned release asset")
    modes.add_argument("--source-only", action="store_true", help="Refresh source/checksums without rewriting the pinned asset")
    args = parser.parse_args()
    out = ROOT / "submission"
    out.mkdir(exist_ok=True)
    if not args.source_only:
        write_asset_archive(out / "prism-live-rag-assets.tar.gz")
    if args.assets_only:
        print("Prepared submission/prism-live-rag-assets.tar.gz")
        return
    source_paths = ["README.md", "SECURITY.md", "AGENTS.md", "guide.md", "pyproject.toml", "requirements.lock",
                    "requirements-cpu.lock", "Dockerfile", "docker-compose.yml", "release-assets.json", ".dockerignore",
                    ".gitignore", ".env.example", "src", "scripts", "tests", "docs", "data/LICENSE"]
    with tarfile.open(out / "prism-live-rag-source.tar.gz", "w:gz", compresslevel=1) as archive:
        for relative in source_paths:
            archive.add(ROOT / relative, arcname=relative, filter=safe_member)
        evaluation = ROOT / "data/curated_dataset/evaluation"
        for pattern in ("*ablation*.json", "g3-candidates-cerebras-4rpm-after-fixes.json",
                        "g3-candidates-rule.json", "g4-dashboard-200-judge-final*.json*",
                        "dashboard-auto-200-judge-final-20260930.jsonl",
                        "g6-full-streaming-gpu-20260929.json*", "g5*20260929*.json*",
                        "dashboard-judge-cpu-200-20260930*",
                        "dashboard-judge-cpu-200-report-20260930.json",
                        "dashboard-provider-date-guard-20260930.jsonl"):
            for path in sorted(evaluation.glob(pattern)):
                archive.add(path, arcname=str(path.relative_to(ROOT)), filter=safe_member)
        for name in ("judge-release-session-exact-20260930.json",
                     "judge-release-cpu-200-traces-20260930.jsonl",
                     "judge-release-offline-2-20260930.jsonl",
                     "judge-release-offline-traces-20260930.jsonl",
                     "judge-release-offline-report-20260930.json",
                     "clone-bootstrap-2-20260930.jsonl",
                     "clone-bootstrap-traces-20260930.jsonl",
                     "clone-bootstrap-report-20260930.json"):
            path = ROOT / ".cache/verification" / name
            if path.exists():
                archive.add(path, arcname="verification/" + path.name)
    manifests = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "checksums.json":
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            manifests[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    (out / "checksums.json").write_text(json.dumps(manifests, indent=2) + "\n")
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
