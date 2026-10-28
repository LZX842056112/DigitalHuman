#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$1")
shift
mkdir -p "$ROOT/.runtime/logs" "$ROOT/.runtime/cache" "$ROOT/models" "$ROOT/third_party"
exec > >(tee -a "$ROOT/.runtime/logs/setup.log") 2>&1
bash "$ROOT/scripts/configure_pip.sh" "$ROOT"
export DEBIAN_FRONTEND=noninteractive
export PIP_CACHE_DIR="$ROOT/.runtime/cache/pip"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_FIND_LINKS="$ROOT/.runtime/wheels"
export PIP_RESUME_RETRIES=25
export PIP_DEFAULT_TIMEOUT=90
export MAX_JOBS=4
export HF_HOME="$ROOT/.runtime/hf"
export TORCH_HOME="$ROOT/.runtime/torch"
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
apt-get update
apt-get install -y python3.10 python3.10-venv python3-dev build-essential git curl ca-certificates cmake ninja-build ffmpeg libgl1 libglib2.0-0 pkg-config
if ! test -f /etc/apt/sources.list.d/cuda-wsl-ubuntu-x86_64.list; then
  curl -fL --retry 3 -o "$ROOT/.runtime/cuda-keyring.deb" https://developer.download.nvidia.cn/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i "$ROOT/.runtime/cuda-keyring.deb"
fi
sed -i 's/developer.download.nvidia.com/developer.download.nvidia.cn/g' /etc/apt/sources.list.d/cuda-wsl-ubuntu-x86_64.list
apt-get -o Acquire::Retries=3 update
# Runtime inference and extension builds need the compiler and headers, not Nsight/GUI tools.
apt-get -o Acquire::Retries=3 install -y cuda-nvcc-12-1 cuda-cudart-dev-12-1 libcurand-dev-12-1 libcublas-dev-12-1 libcusparse-dev-12-1 libcusolver-dev-12-1
mkdir -p /opt/digitalhuman
cd "$ROOT"
for env in avatar speech; do
  if ! test -x "/opt/digitalhuman/$env/bin/python"; then python3.10 -m venv "/opt/digitalhuman/$env"; fi
  "/opt/digitalhuman/$env/bin/python" -m scripts.prepare_wheels 'pip==25.2' 'setuptools==74.0.0' 'wheel==0.45.1'
  "/opt/digitalhuman/$env/bin/python" -m pip install --no-index 'pip==25.2' 'setuptools==74.0.0' 'wheel==0.45.1'
done
AVATAR=/opt/digitalhuman/avatar/bin/python
SPEECH=/opt/digitalhuman/speech/bin/python
TORCH_WHEEL="$ROOT/.runtime/torch-2.3.0+cu121-cp310-cp310-linux_x86_64.whl"
if ! "$AVATAR" -c 'import torch; assert torch.__version__.startswith("2.3.0")' 2>/dev/null; then
  cd "$ROOT"
  python3 -m scripts.fetch_artifact 'https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-cp310-cp310-linux_x86_64.whl' "$TORCH_WHEEL" '0a12aa9aa6bc442dff8823ac8b48d991fd0771562eaa38593f9c8196d65f7007' 780956465
  "$AVATAR" -m scripts.prepare_torch_dependencies
  # --find-links does not give local files priority over the package index.
  # Install the exact, checksum-verified CUDA set offline before resolving Torch.
  "$AVATAR" -c 'import json, subprocess, sys; deps=json.load(open(".runtime/torch-cuda-dependencies.json"))["requirements"]; subprocess.check_call([sys.executable,"-m","pip","install","--no-index","--no-deps",*deps])'
  "$AVATAR" -m pip install "$TORCH_WHEEL"
fi
"$AVATAR" -m pip install 'torchvision==0.18.0' 'torchaudio==2.3.0' --index-url https://download.pytorch.org/whl/cu121
"$AVATAR" -m scripts.prepare_wheels 'open3d==0.19.0' 'matplotlib==3.5.3' 'xformers==0.0.26.post1' 'numpy==1.23.5' 'opencv-python==4.11.0.86' 'opencv-python-headless==4.11.0.86'
"$AVATAR" -m pip install --no-index --no-deps 'open3d==0.19.0' 'matplotlib==3.5.3' 'xformers==0.0.26.post1' 'numpy==1.23.5' 'opencv-python==4.11.0.86' 'opencv-python-headless==4.11.0.86'
"$AVATAR" -m pip install 'xformers==0.0.26.post1' --index-url https://download.pytorch.org/whl/cu121
cd "$ROOT"
"$AVATAR" -m scripts.fetch_source_files --root "$ROOT" --only lhmpp a2e gaga_track gpt_sovits
sed -e 's/^numpy==1.23.0$/numpy==1.23.5/' \
    -e 's/^opencv_python$/opencv-python==4.11.0.86/' \
    -e 's/^opencv_python_headless$/opencv-python-headless==4.11.0.86/' \
    "$ROOT/third_party/lhmpp/requirements.txt" > "$ROOT/.runtime/lhmpp-requirements.txt"
"$AVATAR" -m pip install -r "$ROOT/.runtime/lhmpp-requirements.txt"
"$AVATAR" -m pip install -e "$ROOT[test,asr]"
"$AVATAR" -m pip install 'face-alignment==1.4.1' 'lmdb==1.5.1' 'rich==13.8.1' 'onnx==1.16.1' 'onnxruntime==1.19.2' 'onnx2torch==1.5.15' 'soundfile==0.13.1' 'addict==2.4.0'
"$AVATAR" -m pip install 'gsplat==1.4.0' 'torch_scatter==2.1.2+pt23cu121' -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
"$AVATAR" -m pip install 'fvcore==0.1.5.post20221221' 'iopath==0.1.10'
"$AVATAR" -m pip install --no-index 'pytorch3d==0.7.6' -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html
"$AVATAR" -m pip install "$ROOT/third_party/lhmpp/lib/pointops" spconv-cu121
"$AVATAR" -m pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git git+https://github.com/camenduru/simple-knn.git
"$SPEECH" -m pip install 'torch==2.5.1' 'torchaudio==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cpu
"$AVATAR" -m scripts.speech_requirements
"$SPEECH" -m pip install -r "$ROOT/.runtime/speech-requirements.txt" -c "$ROOT/configs/speech-constraints.txt"
"$SPEECH" -m pip install 'numpy==1.26.4' 'transformers==4.51.3' 'pydantic==2.10.6' 'omegaconf==2.3.0' 'einops==0.8.1' 'fastapi==0.115.12' 'uvicorn==0.34.2' 'librosa==0.10.2' 'soundfile==0.13.1' 'httpx==0.28.1'
"$AVATAR" -m pip freeze > "$ROOT/.runtime/avatar-requirements.lock"
"$SPEECH" -m pip freeze > "$ROOT/.runtime/speech-requirements.lock"
if [[ " $* " == *' --environment-only '* ]]; then
  echo 'Environment installation finished. Models and services have not been started.'
  exit 0
fi
if [[ " $* " != *' --skip-models '* ]]; then
  cd "$ROOT"
  "$AVATAR" -m scripts.download_modelscope --only lhmpp prior a2e gpt_sovits voice_text_assets llm asr wav2vec
  "$AVATAR" -m scripts.arrange_assets
  "$AVATAR" -m scripts.warmup_assets
fi
if ! test -x /opt/digitalhuman/llama.cpp/build/bin/llama-server; then
  if ! test -d /opt/digitalhuman/llama.cpp/.git; then
    git clone --depth 1 --branch b11897 https://github.com/ggml-org/llama.cpp.git /opt/digitalhuman/llama.cpp
  fi
  cmake -S /opt/digitalhuman/llama.cpp -B /opt/digitalhuman/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build /opt/digitalhuman/llama.cpp/build --config Release -j 8 --target llama-server
  git -C /opt/digitalhuman/llama.cpp rev-parse HEAD > "$ROOT/.runtime/llama.commit"
fi
if ! "$AVATAR" -m scripts.check_frontend; then
if ! command -v node >/dev/null || [[ $(node -p 'parseInt(process.versions.node)') -lt 22 ]]; then
  curl -fL --retry 3 -o "$ROOT/.runtime/node.tar.xz" https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.xz
  tar -xf "$ROOT/.runtime/node.tar.xz" -C /opt/digitalhuman
  ln -sfn /opt/digitalhuman/node-v22.22.0-linux-x64/bin/node /usr/local/bin/node
  ln -sfn /opt/digitalhuman/node-v22.22.0-linux-x64/bin/npm /usr/local/bin/npm
fi
cd "$ROOT/frontend"
npm ci --cache "$ROOT/.runtime/cache/npm" --registry=https://registry.npmmirror.com
npm run build
fi
cd "$ROOT"
"$AVATAR" -m scripts.doctor --json
echo 'Run scripts/start.ps1 to start the local studio.'
