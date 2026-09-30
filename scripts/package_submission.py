"""Create source/assets archives from explicit allowlists, excluding secrets/history."""
from __future__ import annotations
import hashlib
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


def main():
    out = ROOT / "submission"
    out.mkdir(exist_ok=True)
    source_paths = ["README.md", "SECURITY.md", "AGENTS.md", "guide.md", "pyproject.toml", "requirements.lock",
                    "requirements-cpu.lock", "Dockerfile", "docker-compose.yml", ".dockerignore",
                    ".gitignore", ".env.example", "src", "scripts", "tests", "docs"]
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
                     "judge-release-offline-report-20260930.json"):
            path = ROOT / ".cache/verification" / name
            if path.exists():
                archive.add(path, arcname="verification/" + path.name)
    with tarfile.open(out / "prism-live-rag-assets.tar.gz", "w:gz", compresslevel=1) as archive:
        archive.add(ROOT / "submission-assets", arcname="submission-assets", filter=safe_member)
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
