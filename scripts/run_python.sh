#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$1")
shift
cd "$ROOT"
export DIGITAL_HUMAN_ROOT="$ROOT"
export HF_HOME="$ROOT/.runtime/hf"
export TORCH_HOME="$ROOT/.runtime/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:/usr/lib/wsl/lib:$PATH"
export TORCHDYNAMO_DISABLE=0
export MAX_JOBS=${MAX_JOBS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.runtime/cache/inductor"
exec /opt/digitalhuman/avatar/bin/python "$@"
