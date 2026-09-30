# Security policy

Prism Live RAG is a local hackathon demonstration, not an authenticated public
service. Use the dashboard only through a loopback address. Compose publishes
`127.0.0.1:8080`; do not publish it to a LAN, public tunnel or the internet.

## Local protections and limits

- Provider credentials are loaded server-side from environment variables or
  ignored `.env` files. The example file contains no credentials.
- API run requests require JSON, bounded nonnegative content length and a local
  Host name. Cross-origin browser requests are rejected. Static asset paths are
  restricted to the packaged asset directory; dynamic text uses DOM text nodes.
- The Docker process is non-root. Models and the index are bundled for offline
  inference; normal startup does not fetch executable model code.
- There is no authentication, per-user isolation or production resource quota.
  The executor serializes runs, but its queue/history can grow. Use with trusted
  local users only. Five-RPM provider pacing is not an authorization mechanism.
- Traces and API snapshots contain transcript text, prior conversation and retrieved
  evidence. Do not submit private conversations or publish raw traces without review.

## Repository and submission hygiene

Do not commit `.env`, API keys, credentials, local model caches, database/index
files or generated archives. Do not include `.env` in Docker build contexts or
submission bundles. Review the staged diff before pushing. The current credential
check is an exact-match/pattern inspection, not a comprehensive security audit or
proof that all third-party dependencies are vulnerability-free.

If credentials are exposed, revoke them with the provider immediately; deleting
the file from the latest commit does not remove the secret from Git history.
Report suspected vulnerabilities privately to the repository owner rather than
posting credentials or private transcript data in a public issue.
