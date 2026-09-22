#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/prism-rag ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.lock
  .venv/bin/python -m pip install --no-deps -e .
fi

.venv/bin/python scripts/fetch-corpus.py
.venv/bin/prism-rag validate-data
.venv/bin/prism-rag run-demo --domain cloud --corpus-limit 1000 --refine-on-final
.venv/bin/prism-rag eval --domain cloud --mode retrieval --max-tasks 5
