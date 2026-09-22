#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/fetch-corpus.py
.venv/bin/prism-rag validate-data
.venv/bin/prism-rag index
.venv/bin/prism-rag run-demo --domain cloud --refine-on-final --use-dense
