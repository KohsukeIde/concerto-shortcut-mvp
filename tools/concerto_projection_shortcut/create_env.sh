#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

INSTALL_POINTGROUP_OPS="${INSTALL_POINTGROUP_OPS:-0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

if ! command -v conda >/dev/null 2>&1 || [ "$(type -t conda)" != "function" ]; then
  set +u
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  set -u
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
fi

set +u
conda activate "${CONDA_ENV_NAME}"
set -u

conda install -y \
  python=3.10

if conda list | rg -q '^(pytorch|torchvision|torchaudio|pytorch-cuda|pytorch-mutex|torchtriton)\s'; then
  conda remove -y \
    pytorch \
    torchvision \
    torchaudio \
    pytorch-cuda \
    pytorch-mutex \
    torchtriton
fi

conda install -y \
  -c nvidia/label/cuda-12.1.0 \
  cuda-nvcc=12.1.66 \
  cuda-cudart-dev=12.1.55 \
  libcublas-dev=12.1.0.26 \
  libcusparse-dev=12.0.2.55

if [ -L "${CONDA_PREFIX}/lib/libcudart.so" ] && [ ! -e "${CONDA_PREFIX}/lib/libcudart.so" ] && [ -e "${CONDA_PREFIX}/lib/libcudart.so.12" ]; then
  ln -sfn "$(basename "$(readlink -f "${CONDA_PREFIX}/lib/libcudart.so.12")")" "${CONDA_PREFIX}/lib/libcudart.so"
fi

python -m pip uninstall -y torch torchvision torchaudio triton || true
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  numpy==1.26.4 \
  ninja \
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
  SharedArray

python -m pip install \
  --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html \
  torch-cluster \
  torch-scatter \
  torch-sparse \
  torch-geometric \
  spconv-cu121 \
  transformers==4.50.3 \
  peft

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST
python -m pip install --no-build-isolation ./libs/pointops

if [ "${INSTALL_POINTGROUP_OPS}" = "1" ]; then
  python -m pip install ./libs/pointgroup_ops
fi

if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
  python -m pip install git+https://github.com/Dao-AILab/flash-attention.git
fi

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/pointcept_paths.sh" <<EOF
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE}"
export TORCH_HOME="${TORCH_HOME}"
export CUDA_HOME="\${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
EOF

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TORCH_HOME}" "${POINTCEPT_DATA_ROOT}"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
PY

echo "[ok] conda env: ${CONDA_ENV_NAME}"
echo "[ok] python: $(command -v python)"
echo "[ok] HF_HOME: ${HF_HOME}"
echo "[ok] TORCH_HOME: ${TORCH_HOME}"
