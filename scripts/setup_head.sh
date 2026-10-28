#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$1")
cd "$ROOT"
export PIP_CACHE_DIR="$ROOT/.runtime/cache/pip"
export HF_HOME="$ROOT/.runtime/hf"
export TORCH_HOME="$ROOT/.runtime/torch"
/opt/digitalhuman/avatar/bin/python -m scripts.setup_head --root "$ROOT"
