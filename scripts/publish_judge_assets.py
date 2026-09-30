"""Maintainer-only release publication; never part of a judge's setup."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
from urllib.parse import urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prism_live_rag.bootstrap import digest_file, load_descriptor

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Create the GitHub release and upload the verified asset")
    parser.add_argument("--check-public", action="store_true", help="Verify anonymous judges can download the published asset")
    args = parser.parse_args()
    descriptor = load_descriptor(ROOT / "release-assets.json")
    parsed = urlsplit(descriptor["url"])
    parts = parsed.path.strip("/").split("/")
    if parsed.hostname != "github.com" or len(parts) != 6 or parts[2:4] != ["releases", "download"]:
        raise SystemExit("Expected a pinned public GitHub release URL")
    repository, tag, name = "/".join(parts[:2]), parts[4], parts[5]
    archive = ROOT / "submission" / name
    if not archive.is_file() or archive.stat().st_size != descriptor["bytes"] or digest_file(archive) != descriptor["sha256"]:
        raise SystemExit("Release archive is missing or does not match release-assets.json; regenerate/re-pin before publishing")
    print(f"Repository: {repository}\nRelease tag: {tag}\nAsset: {archive}\nSHA-256: {descriptor['sha256']}", flush=True)
    if args.check_public:
        response = requests.head(descriptor["url"], allow_redirects=True, timeout=30)
        if response.status_code != 200 or urlsplit(response.url).scheme != "https":
            raise SystemExit(f"Anonymous release download unavailable (HTTP {response.status_code}); check repo visibility and release publication")
        if int(response.headers.get("Content-Length", -1)) != descriptor["bytes"]:
            raise SystemExit("Published archive size does not match the pinned descriptor")
        print("Anonymous release download is available; clients still verify SHA-256 before extraction")
    if not args.publish:
        return
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token and shutil.which("gh"):
        credential = subprocess.run(["gh", "auth", "token", "--hostname", "github.com"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        if credential.returncode == 0:
            token = credential.stdout.strip()  # Never logged or written to a file.
    if not token:
        raise SystemExit("Authenticate with gh auth login, set GH_TOKEN/GITHUB_TOKEN locally, or upload through GitHub's release page. Never commit the token.")
    api = f"https://api.github.com/repos/{repository}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    existing = requests.get(f"{api}/releases/tags/{tag}", headers=headers, timeout=30)
    if existing.status_code == 404:
        existing = requests.post(f"{api}/releases", headers=headers, timeout=30, json={
            "tag_name": tag, "name": "Pinned CPU judge assets", "draft": True,
            "prerelease": False, "body": f"Corpus/index/model assets for automatic startup. SHA-256: `{descriptor['sha256']}`. No API keys or private runtime files.",
        })
    if existing.status_code not in {200, 201}:
        raise SystemExit(f"GitHub release operation failed (HTTP {existing.status_code}); check repo access and token permissions")
    release = existing.json()
    if any(asset["name"] == name for asset in release.get("assets", [])):
        raise SystemExit("An asset with this name already exists; refusing to overwrite it. Check the public download instead.")
    upload_url = release["upload_url"].split("{", 1)[0]
    if urlsplit(upload_url).hostname != "uploads.github.com":
        raise SystemExit("Unexpected GitHub upload host")
    print("Uploading verified release archive…", flush=True)
    with archive.open("rb") as body:
        response = requests.post(upload_url, params={"name": name}, data=body,
                                 headers={**headers, "Content-Type": "application/gzip"}, timeout=(30, 600))
    if response.status_code != 201:
        raise SystemExit(f"GitHub asset upload failed (HTTP {response.status_code})")
    if release.get("draft"):
        response = requests.patch(f"{api}/releases/{release['id']}", headers=headers,
                                  json={"draft": False}, timeout=30)
        if response.status_code != 200:
            raise SystemExit(f"Asset uploaded, but release publication failed (HTTP {response.status_code}); publish the draft through GitHub")
    print("Release asset uploaded. Run --check-public after making the repository publicly readable.")


if __name__ == "__main__":
    main()
