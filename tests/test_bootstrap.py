import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from prism_live_rag.bootstrap import (
    AssetError, checked_relative, download_archive, ensure_assets, install_archive,
    load_descriptor, verify_assets,
)


def make_archive(tmp_path, extra=None):
    content = b"real fixture corpus data"
    name = "data/corpora/passage_level/cloud.jsonl"
    manifest = json.dumps({name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}).encode()
    archive = tmp_path / "assets.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path, body in ((name, content), ("manifest.json", manifest)):
            entry = tarfile.TarInfo("submission-assets/" + path)
            entry.size = len(body)
            tar.addfile(entry, io.BytesIO(body))
        if extra:
            entry = tarfile.TarInfo(extra[0])
            if extra[1] == "symlink":
                entry.type, entry.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tar.addfile(entry)
            else:
                entry.size = len(extra[1])
                tar.addfile(entry, io.BytesIO(extra[1]))
    descriptor = {"url": "https://example.com/assets.tar.gz", "bytes": archive.stat().st_size,
                  "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "manifest_sha256": hashlib.sha256(manifest).hexdigest()}
    return archive, descriptor


def test_verified_install_is_atomic_and_cached_without_download(tmp_path, monkeypatch):
    archive, descriptor = make_archive(tmp_path)
    cache = tmp_path / "cache"
    first = ensure_assets(descriptor, cache, archive)
    assert (cache / "current").resolve() == first
    monkeypatch.setattr("prism_live_rag.bootstrap.download_archive", lambda *a: pytest.fail("cache must not download"))
    assert ensure_assets(descriptor, cache) == first


def test_modified_cached_asset_fails_closed(tmp_path):
    archive, descriptor = make_archive(tmp_path)
    root = ensure_assets(descriptor, tmp_path / "cache", archive)
    (root / "data/corpora/passage_level/cloud.jsonl").write_bytes(b"tampered")
    with pytest.raises(AssetError):
        ensure_assets(descriptor, tmp_path / "cache", archive)


@pytest.mark.parametrize("extra", [
    ("submission-assets/../../escape", b"bad"),
    ("/tmp/escape", b"bad"),
    ("submission-assets/.env", b"private"),
    ("submission-assets/link", "symlink"),
    ("submission-assets/unlisted", b"bad"),
    ("other-root/file", b"bad"),
])
def test_unsafe_or_unlisted_members_are_rejected_without_publishing(tmp_path, extra):
    archive, descriptor = make_archive(tmp_path, extra)
    destination = tmp_path / "installed"
    with pytest.raises(AssetError):
        install_archive(archive, destination, descriptor)
    assert not destination.exists()


def test_wrong_archive_hash_and_manifest_hash_are_rejected(tmp_path):
    archive, descriptor = make_archive(tmp_path)
    with pytest.raises(AssetError):
        install_archive(archive, tmp_path / "bad", {**descriptor, "sha256": "0" * 64})
    with pytest.raises(AssetError):
        install_archive(archive, tmp_path / "bad", {**descriptor, "manifest_sha256": "0" * 64})


@pytest.mark.parametrize("url", ["http://example.com/a", "file:///tmp/a", "https://key:secret@example.com/a"])
def test_descriptor_rejects_insecure_or_credentialed_urls(tmp_path, url):
    _, descriptor = make_archive(tmp_path)
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps({**descriptor, "url": url}))
    with pytest.raises(AssetError):
        load_descriptor(path)


def test_download_resumes_and_verifies_exact_bytes(tmp_path, monkeypatch):
    _, descriptor = make_archive(tmp_path)
    content = (tmp_path / "assets.tar.gz").read_bytes()
    partial = tmp_path / "archive.part"
    partial.write_bytes(content[:10])

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 10-{len(content)-1}/{len(content)}"}
        def geturl(self):
            return descriptor["url"]

    def open_request(request, **kwargs):
        assert request.headers["Range"] == "bytes=10-"
        return Response(content[10:])
    monkeypatch.setattr("prism_live_rag.bootstrap.urllib.request.urlopen", open_request)
    assert download_archive(descriptor, partial).read_bytes() == content


def test_publication_error_is_actionable_and_does_not_retry(tmp_path, monkeypatch):
    from urllib.error import HTTPError
    _, descriptor = make_archive(tmp_path)
    def unavailable(*args, **kwargs):
        raise HTTPError(descriptor["url"], 404, "not found", {}, None)
    monkeypatch.setattr("prism_live_rag.bootstrap.urllib.request.urlopen", unavailable)
    with pytest.raises(AssetError, match="repository owner must publish"):
        download_archive(descriptor, tmp_path / "archive.part")


def test_server_ignoring_range_restarts_download_without_appending(tmp_path, monkeypatch):
    archive, descriptor = make_archive(tmp_path)
    content = archive.read_bytes()
    partial = tmp_path / "archive.part"
    partial.write_bytes(content[:10])
    class Response(io.BytesIO):
        status, headers = 200, {}
        def geturl(self):
            return descriptor["url"]
    monkeypatch.setattr("prism_live_rag.bootstrap.urllib.request.urlopen", lambda *a, **k: Response(content))
    assert download_archive(descriptor, partial).read_bytes() == content


def test_invalid_resume_range_fails_closed(tmp_path, monkeypatch):
    _, descriptor = make_archive(tmp_path)
    partial = tmp_path / "archive.part"
    partial.write_bytes(b"1234567890")
    class Response(io.BytesIO):
        status, headers = 206, {"Content-Range": "bytes 0-100/101"}
        def geturl(self):
            return descriptor["url"]
    monkeypatch.setattr("prism_live_rag.bootstrap.urllib.request.urlopen", lambda *a, **k: Response(b"bad"))
    with pytest.raises(AssetError, match="range"):
        download_archive(descriptor, partial)
    assert partial.read_bytes() == b"1234567890"


def test_asset_archive_is_reproducible_across_timestamp_changes(tmp_path):
    import os
    import runpy
    script = Path(__file__).parents[1] / "scripts/package_submission.py"
    writer = runpy.run_path(str(script))["write_asset_archive"]
    writer.__globals__["ROOT"] = tmp_path
    staged = tmp_path / "submission-assets"
    staged.mkdir()
    fixture = staged / "fixture.txt"
    fixture.write_text("same bytes")
    first, second = tmp_path / "first.gz", tmp_path / "second.gz"
    writer(first)
    os.utime(fixture, (100, 100))
    os.utime(staged, (100, 100))
    writer(second)
    assert first.read_bytes() == second.read_bytes()


def test_clean_clone_docker_has_no_local_asset_copy_dependency():
    root = Path(__file__).parents[1]
    dockerfile = (root / "Dockerfile").read_text()
    compose = (root / "docker-compose.yml").read_text()
    assert "COPY --chown=prism:prism submission-assets" not in dockerfile
    assert "prism_live_rag.bootstrap" in dockerfile
    assert "assets:/opt/prism-assets" in compose
    assert "127.0.0.1:${PRISM_PORT:-8080}" in compose
