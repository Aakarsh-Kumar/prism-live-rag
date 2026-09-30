# One-time maintainer release publication

This is the repository owner's task, not a judge setup task. The judge workflow is
clone, supply a Cerebras key, run `docker compose up --build`. Runtime downloads
the public asset automatically and verifies the pinned archive/file checksums.

## Publish the existing pinned bundle

1. Push the reviewed source commits to the GitHub repository. No push or release
   upload has been performed by the implementation agent.
2. Ensure the repository and release are publicly readable. The configured
   origin `Aakarsh-Kumar/prism-live-rag` returned HTTP 404 anonymously during
   verification; that may indicate missing publication or private visibility.
3. Create a release with tag **`judge-assets-2026-09-30`**.
4. Attach exactly **`submission/prism-live-rag-assets.tar.gz`**. Do not attach `.env`,
   caches or private credentials. This archive is 639,158,194 bytes and has SHA-256:

   ```text
   1b1353af140c57f0637c533c85d05bc119854659ae9a38f44ca9f0cb71047969
   ```

5. Publish the release, not merely a draft. Use the tag and filename exactly as
   pinned in `release-assets.json`. If the repository changes, update only the
   public URL in that descriptor; keep the checksum tied to the actual archive.
6. Check anonymous availability:

   ```bash
   .venv/bin/python scripts/publish_judge_assets.py --check-public
   ```

7. Ask a friend to run the README workflow from a fresh clone and empty asset volume.
   Update verification status only after that actual public path succeeds.

GitHub's [release management instructions](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
describe publication through the web UI. Alternatively, the maintainer helper can
upload using `gh auth login` credentials or a locally supplied `GH_TOKEN`/
`GITHUB_TOKEN` with repository release write permissions. GitHub CLI credentials
are captured in memory, never printed or persisted by the helper:

```bash
.venv/bin/python scripts/publish_judge_assets.py --publish
```

This flag explicitly writes a release/upload. The default invocation only checks
the local archive and prints its intended destination. The helper refuses to
overwrite an existing release asset. Never commit the GitHub token, and never use
the Cerebras key as GitHub authentication.

## Regenerating assets later

`scripts/prepare_judge_assets.py` stages existing corpus/index/model files without
rebuilding or changing them and retains dataset licensing. The file manifest
covers every staged payload file, including nested manifests.
`scripts/package_submission.py --assets-only` generates a deterministic gzip/tar
archive with canonical timestamps/ownership. Re-pin the archive size/SHA-256 and
file-manifest SHA-256 in `release-assets.json` whenever payload contents change.
Use a new release tag for changed assets; do not mutate a previously pinned release.

`scripts/package_submission.py --source-only` refreshes source/checksum bundles
without rewriting an archive that is being uploaded. The legacy all-in-one image
archive is a different deployment snapshot; it is not required by the new clone path.
