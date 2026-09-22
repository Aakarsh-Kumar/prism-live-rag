#!/usr/bin/env bash
# Install the neural dense-retrieval stack, using the GPU when one is present.
#
# The dense encoder runs through onnxruntime. onnxruntime (CPU) and
# onnxruntime-gpu ship the same Python module, so exactly one may be installed.
# fastembed depends on the CPU package, so on a machine with an NVIDIA GPU we
# install the CPU stack first and then swap in the GPU wheel. The runtime code
# in prism_live_rag.embeddings auto-detects the provider and falls back to CPU,
# so this script is an optimization, not a requirement.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ORT_GPU_VERSION="${ORT_GPU_VERSION:-1.24.4}"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
PY=".venv/bin/python"

want_gpu="no"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  want_gpu="yes"
fi

# Skip the (slow) reinstall when the package is already present with the right
# onnxruntime flavor. Reinstalling from the lock would pull the CPU wheel back
# and undo the GPU swap, so this guard matters.
already_ready="no"
if "$PY" -c "import prism_live_rag" >/dev/null 2>&1; then
  current_ort="$("$PY" - <<'PY'
import importlib.metadata as metadata

names = {d.metadata["Name"].lower() for d in metadata.distributions() if d.metadata.get("Name")}
print("gpu" if "onnxruntime-gpu" in names else "cpu" if "onnxruntime" in names else "none")
PY
)"
  if [[ "$want_gpu" == "yes" && "$current_ort" == "gpu" ]]; then
    already_ready="yes"
  elif [[ "$want_gpu" == "no" && "$current_ort" == "cpu" ]]; then
    already_ready="yes"
  fi
fi

if [[ "$already_ready" == "yes" ]]; then
  echo "== neural stack already installed with the correct onnxruntime build =="
else
  echo "== installing pinned dependencies =="
  "$PY" -m pip install -r requirements.lock
  "$PY" -m pip install --no-deps -e .

  if [[ "$want_gpu" == "yes" ]]; then
    echo "== NVIDIA GPU detected; installing onnxruntime-gpu[cuda,cudnn]==${ORT_GPU_VERSION} =="
    echo "   (large download: ~1 GB of CUDA/cuDNN wheels; this runs once and is cached)"
    "$PY" -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
    "$PY" -m pip install "onnxruntime-gpu[cuda,cudnn]==${ORT_GPU_VERSION}"
  else
    echo "== no NVIDIA GPU detected; keeping the CPU onnxruntime build =="
  fi
fi

"$PY" - <<'PY'
from prism_live_rag.embeddings import available_device, cuda_available

print(f"onnxruntime CUDA provider available: {cuda_available()}")
print(f"embeddings will use: {available_device()}")
if not cuda_available():
    print("note: CPU fallback is expected here; the pipeline still runs.")
PY
