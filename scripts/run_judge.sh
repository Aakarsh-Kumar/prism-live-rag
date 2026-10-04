#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! docker info >/dev/null 2>&1; then
  echo "Start Docker Desktop/the Docker daemon, then run this command again." >&2
  exit 1
fi
docker compose up -d
echo "Dashboard: http://localhost:${PRISM_PORT:-8080}"
echo "Follow initialization: docker compose logs -f app"
if [[ "${1:-}" == "--verify" ]]; then
  ready=false
  for ((attempt=0; attempt<900; attempt++)); do
    if docker compose exec -T app python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health',timeout=3)" >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 2
  done
  if [[ "$ready" != true ]]; then
    echo "Dashboard not ready. Inspect docker compose logs app." >&2
    exit 1
  fi
  docker compose exec -T app python scripts/audit_dashboard.py --mode deterministic --limit 200 \
    --output .cache/telemetry/judge-replay.jsonl
  docker compose exec -T app python scripts/summarize_dashboard_audit.py .cache/telemetry/judge-replay.jsonl \
    --traces .cache/telemetry/dashboard-traces.jsonl --output .cache/telemetry/judge-replay-report.json
fi
