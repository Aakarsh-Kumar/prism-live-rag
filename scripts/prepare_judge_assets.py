"""Stage allowlisted corpus/model assets, never .env or local history."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def stage(source: Path, target: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Required judge asset missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, symlinks=False, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.lock", ".locks", ".git", "__pycache__"))
    else:
        shutil.copy2(source, target)
    for path in ([target] if target.is_file() else target.rglob("*")):
        if path.is_file():
            path.chmod(0o644)
        elif path.is_dir():
            path.chmod(0o755)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-hub", type=Path, default=Path.home() / ".cache/huggingface/hub")
    args = parser.parse_args()
    out = ROOT / "submission-assets"
    for relative in (
        "corpora/passage_level/cloud.jsonl", "corpora/passage_level/govt.jsonl",
        "mtragun-human/retrieval_tasks/qrels/cloud.tsv", "mtragun-human/retrieval_tasks/qrels/govt.tsv",
        "mtragun-human/generation_tasks/reference.jsonl", "simulated_streams",
        "curated_dataset/streams", "curated_dataset/decomposition",
        "curated_dataset/test.jsonl", "curated_dataset/manifest.json",
        "corpora/README.md", "mtragun-human/README.md",
    ):
        stage(ROOT / "data" / relative, out / "data" / relative)
    stage(ROOT / ".cache/lancedb", out / "lancedb")
    stage(ROOT / ".cache/fastembed/models--qdrant--bge-small-en-v1.5-onnx-q",
          out / "fastembed/models--qdrant--bge-small-en-v1.5-onnx-q")
    stage(args.hf_hub / "models--cross-encoder--ms-marco-MiniLM-L-12-v2",
          out / "huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-12-v2")
    manifest = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            manifest[str(path.relative_to(out))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Staged {len(manifest)} files in {out}; no source assets changed")

if __name__ == "__main__":
    main()
