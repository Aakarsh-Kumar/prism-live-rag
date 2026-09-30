"""Download pinned judge assets once, verify them, then launch the application."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class AssetError(RuntimeError):
    pass


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_relative(name: str) -> Path:
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or "\\" in name or any(
        part in {"..", ".env", ".git", ".venv"} for part in parts
    ):
        raise AssetError("Asset archive contains an unsafe path")
    return Path(*parts)


def load_descriptor(path: Path) -> dict:
    try:
        descriptor = json.loads(path.read_text())
        parsed = urlsplit(descriptor["url"])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Asset URL must be public HTTPS, without credentials")
        for key in ("sha256", "manifest_sha256"):
            value = descriptor[key]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid pinned checksum")
        if not isinstance(descriptor["bytes"], int) or not 0 < descriptor["bytes"] <= 2_000_000_000:
            raise ValueError("Invalid pinned archive size")
        return descriptor
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AssetError("Invalid or missing release-assets.json") from exc


def verify_assets(root: Path, descriptor: dict) -> None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or digest_file(manifest_path) != descriptor["manifest_sha256"]:
        raise AssetError("Asset file manifest failed checksum verification")
    try:
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("Empty asset manifest")
        for name, row in manifest.items():
            relative = checked_relative(name)
            path = root / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
                raise AssetError(f"Asset missing or wrong size: {relative}")
            if digest_file(path) != row["sha256"]:
                raise AssetError(f"Asset checksum mismatch: {relative}")
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        # Mutable model-runtime sidecars are not part of the trusted asset payload.
        # Installation verifies the exact payload; cached verification still checks
        # every pinned file and forbids injected secret files.
        if any(Path(name).name == ".env" for name in actual):
            raise AssetError("Unexpected secret file in asset cache")
    except (ValueError, TypeError, KeyError) as exc:
        raise AssetError("Malformed asset file manifest") from exc


def download_archive(descriptor: dict, target: Path) -> Path:
    """Resume interrupted transfers; never accept unverified bytes."""
    if target.exists() and target.stat().st_size == descriptor["bytes"]:
        if digest_file(target) == descriptor["sha256"]:
            return target
        target.rename(target.with_name(target.name + f".rejected-{time.time_ns()}"))
    for attempt in range(3):
        offset = target.stat().st_size if target.exists() else 0
        if offset > descriptor["bytes"]:
            raise AssetError("Cached archive exceeds pinned size")
        request = urllib.request.Request(descriptor["url"], headers={
            "User-Agent": "prism-live-rag-bootstrap", **({"Range": f"bytes={offset}-"} if offset else {})
        })
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if urlsplit(response.geturl()).scheme != "https":
                    raise AssetError("Asset download redirected away from HTTPS")
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise AssetError("Invalid resumed-download range")
                total = offset if append else 0
                next_log = total + 64 * 1024 * 1024
                with target.open("ab" if append else "wb") as handle:
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > descriptor["bytes"]:
                            raise AssetError("Downloaded archive exceeds pinned size")
                        handle.write(block)
                        if total >= next_log:
                            print(f"Judge assets: {100 * total / descriptor['bytes']:.0f}% downloaded", flush=True)
                            next_log = total + 64 * 1024 * 1024
            if target.stat().st_size != descriptor["bytes"]:
                raise OSError("Incomplete asset download")
            if digest_file(target) != descriptor["sha256"]:
                raise AssetError("Downloaded archive failed SHA-256 verification")
            return target
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 404}:
                raise AssetError("Judge asset release is unavailable. The repository owner must publish the public release referenced by release-assets.json; judges do not need GitHub credentials.") from exc
            if attempt == 2:
                raise AssetError(f"Asset download failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            if attempt == 2:
                raise AssetError("Asset download interrupted after three attempts; restart to resume") from exc
        time.sleep(2 ** attempt)
    raise AssetError("Asset download failed")


def install_archive(archive: Path, destination: Path, descriptor: dict) -> None:
    if archive.stat().st_size != descriptor["bytes"] or digest_file(archive) != descriptor["sha256"]:
        raise AssetError("Local archive failed pinned size/SHA-256 verification")
    with tempfile.TemporaryDirectory(prefix=".install-", dir=destination.parent) as staging:
        root = Path(staging)
        seen = set()
        expanded = 0
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle:
                relative = checked_relative(member.name)
                if relative.parts[0] != "submission-assets" or relative.as_posix() in seen:
                    raise AssetError("Archive has unexpected root or duplicate entries")
                seen.add(relative.as_posix())
                if not member.isdir() and not member.isfile():
                    raise AssetError("Archive links and special files are not allowed")
                target = root / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                expanded += member.size
                if member.size < 0 or expanded > 4_000_000_000:
                    raise AssetError("Asset archive exceeds extraction size limit")
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o644)
        payload = root / "submission-assets"
        verify_assets(payload, descriptor)
        manifest = json.loads((payload / "manifest.json").read_text())
        actual = {p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()}
        if actual != set(manifest) | {"manifest.json"}:
            raise AssetError("Archive contains files absent from its pinned manifest")
        payload.rename(destination)


def ensure_assets(descriptor: dict, cache: Path, local_archive: Path | None = None) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / descriptor["sha256"]
    if destination.exists():
        verify_assets(destination, descriptor)
        print("Judge assets: verified cached corpus, index and models", flush=True)
    else:
        print("Judge assets: preparing pinned corpus/index/models; no index rebuild", flush=True)
        archive = local_archive or download_archive(descriptor, cache / f"{descriptor['sha256']}.tar.gz.part")
        install_archive(archive, destination, descriptor)
        if not local_archive:
            archive.unlink()  # Only our verified, disposable download copy.
    current = cache / "current"
    if current.exists() and not current.is_symlink():
        raise AssetError("Refusing to replace a non-symlink asset cache path")
    temporary_link = cache / f".current-{os.getpid()}-{time.time_ns()}"
    temporary_link.symlink_to(destination.name, target_is_directory=True)
    temporary_link.replace(current)
    return destination


def main() -> None:
    try:
        descriptor = load_descriptor(Path(os.environ.get("PRISM_ASSET_DESCRIPTOR", "/app/release-assets.json")))
        local = os.environ.get("PRISM_ASSET_ARCHIVE")
        ensure_assets(descriptor, Path(os.environ.get("PRISM_ASSET_CACHE", "/opt/prism-assets")), Path(local) if local else None)
    except (AssetError, OSError, tarfile.TarError) as exc:
        print(f"Startup blocked: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    command = sys.argv[1:] or ["prism-rag", "serve", "--host", "0.0.0.0", "--port", "8080"]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
