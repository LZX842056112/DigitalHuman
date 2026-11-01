#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$1")
shift
mkdir -p "$ROOT/.runtime/logs" "$ROOT/.runtime/cache" "$ROOT/models" "$ROOT/third_party"
exec > >(tee -a "$ROOT/.runtime/logs/setup.log") 2>&1
export DEBIAN_FRONTEND=noninteractive
export PIP_CACHE_DIR="$ROOT/.runtime/cache/pip"
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121
export PIP_RESUME_RETRIES=25
export PIP_DEFAULT_TIMEOUT=90
export HF_HOME="$ROOT/.runtime/hf"
export TORCH_HOME="$ROOT/.runtime/torch"
apt-get update
apt-get install -y python3.10 python3.10-venv python3-dev build-essential git curl ca-certificates cmake ffmpeg libgl1 libglib2.0-0 pkg-config
mkdir -p /opt/digitalhuman
cd "$ROOT"
for environment in portrait emotional-voice; do
  prefix="/opt/digitalhuman/$environment"
  if ! test -x "$prefix/bin/python"; then python3.10 -m venv "$prefix"; fi
  python="$prefix/bin/python"
  "$python" -m pip install 'pip==25.2' 'setuptools==74.0.0' 'wheel==0.45.1'
  "$python" -m scripts.reuse_environment \
    --target "$prefix/lib/python3.10/site-packages" \
    --source /opt/digitalhuman/avatar/lib/python3.10/site-packages \
    --source /opt/digitalhuman/speech/lib/python3.10/site-packages \
    --requirements "$ROOT/configs/$environment-requirements.txt"
  "$python" -m pip install --prefer-binary -r "$ROOT/configs/$environment-requirements.txt"
done
PORTRAIT=/opt/digitalhuman/portrait/bin/python
VOICE=/opt/digitalhuman/emotional-voice/bin/python
"$PORTRAIT" -m pip install --prefer-binary -c "$ROOT/configs/portrait-requirements.txt" -e "$ROOT[test,asr]"
"$PORTRAIT" -m scripts.photo_assets --only sources
"$PORTRAIT" -m pip check
"$VOICE" -m pip check
"$PORTRAIT" -m pip freeze > "$ROOT/.runtime/portrait-requirements.lock"
"$VOICE" -m pip freeze > "$ROOT/.runtime/emotional-voice-requirements.lock"
if [[ " $* " == *' --environment-only '* ]]; then
  echo '环境安装完成。未下载权重，未启动服务。'
  exit 0
fi
if [[ " $* " != *' --skip-models '* ]]; then
  "$PORTRAIT" -m scripts.photo_assets --only models
fi
if ! test -x /opt/digitalhuman/llama.cpp/build/bin/llama-server; then
  if ! test -d /opt/digitalhuman/llama.cpp/.git; then
    git clone --depth 1 --branch b11897 https://github.com/ggml-org/llama.cpp.git /opt/digitalhuman/llama.cpp
  fi
  # Keep an existing checkout intact. Reject a different revision before building.
  test "$(git -C /opt/digitalhuman/llama.cpp rev-parse HEAD)" = "$(git -C /opt/digitalhuman/llama.cpp rev-parse b11897^{commit})" || {
    echo 'llama.cpp 版本不匹配。请使用 b11897。'; exit 2;
  }
  cmake -S /opt/digitalhuman/llama.cpp -B /opt/digitalhuman/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build /opt/digitalhuman/llama.cpp/build --config Release -j 8 --target llama-server
  git -C /opt/digitalhuman/llama.cpp rev-parse HEAD > "$ROOT/.runtime/llama.commit"
fi
if ! "$PORTRAIT" -m scripts.check_frontend; then
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
"$PORTRAIT" -m scripts.doctor --json
echo '运行 scripts/start.ps1 启动本机界面。'
