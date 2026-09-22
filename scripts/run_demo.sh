#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/setup_embeddings.sh

.venv/bin/python scripts/fetch-corpus.py
.venv/bin/prism-rag validate-data
.venv/bin/prism-rag index --embedding-backend auto
.venv/bin/prism-rag run-demo --domain cloud --refine-on-final --use-dense --retrieval-leg hybrid
