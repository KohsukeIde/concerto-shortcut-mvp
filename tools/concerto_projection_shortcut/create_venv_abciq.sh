#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"

source /etc/profile.d/modules.sh 2>/dev/null || true
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PYG_WHEEL_URL="${PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.5.0+cu124.html}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
INSTALL_POINTGROUP_OPS="${INSTALL_POINTGROUP_OPS:-0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn==2.7.4.post1}"
FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"
MAX_JOBS="${MAX_JOBS:-8}"

mkdir -p \
  "${POINTCEPT_DATA_ROOT}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_XET_CACHE}" \
  "${TORCH_HOME}" \
  "$(dirname "${VENV_DIR}")"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME_AUTO="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_AUTO}}"
fi
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS
export FLASH_ATTENTION_FORCE_BUILD
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME

python -m pip install --upgrade pip setuptools wheel ninja packaging

python -m pip install \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==2.5.0+cu124" \
  "torchvision==0.20.0+cu124" \
  "torchaudio==2.5.0+cu124"

python -m pip install \
  numpy==1.26.4 \
  h5py \
  pyyaml \
  tensorboard \
  tensorboardx \
  wandb \
  yapf \
  addict \
  einops \
  scipy \
  plyfile \
  termcolor \
  timm \
  ftfy \
  regex \
  tqdm \
  matplotlib \
  black \
  open3d \
  SharedArray \
  huggingface_hub \
  transformers==4.50.3 \
  peft \
  spconv-cu124

python -m pip install \
  --find-links "${PYG_WHEEL_URL}" \
  torch-cluster \
  torch-scatter \
  torch-sparse \
  torch-geometric

python -m pip install --no-build-isolation ./libs/pointops

if [ "${INSTALL_POINTGROUP_OPS}" = "1" ]; then
  python -m pip install ./libs/pointgroup_ops
fi

if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
  python -m pip install --no-build-isolation "${FLASH_ATTN_SPEC}"
fi

python - <<'PY'
import torch

print("python import: OK")
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
try:
    import torch_scatter
    print("torch_scatter", torch_scatter.__version__)
except Exception as exc:
    print("torch_scatter import failed", repr(exc))
    raise
try:
    import spconv
    print("spconv import: OK")
except Exception as exc:
    print("spconv import failed", repr(exc))
    raise
try:
    import pointops
    print("pointops import: OK")
except Exception as exc:
    print("pointops import failed", repr(exc))
    raise
import transformers
print("transformers", transformers.__version__)
import pointcept
print("pointcept import: OK")
try:
    import flash_attn
    print("flash_attn", getattr(flash_attn, "__version__", "OK"))
except Exception as exc:
    print("flash_attn import failed", repr(exc))
    raise
PY

echo "[ok] venv: ${VENV_DIR}"
echo "[ok] python: $(command -v python)"
echo "[ok] HF_HOME: ${HF_HOME}"
echo "[ok] TORCH_HOME: ${TORCH_HOME}"
