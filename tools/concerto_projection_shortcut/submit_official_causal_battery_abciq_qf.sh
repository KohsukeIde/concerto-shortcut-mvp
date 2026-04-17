#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -N official_causal
#PBS -j oe

set -euo pipefail

cd "${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}" || exit 1

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source tools/concerto_projection_shortcut/device_defaults.sh
ensure_venv_active

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/official_causal_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== ABCI-Q official Concerto causal battery ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "python_module=${PYTHON_MODULE}"
echo "cuda_module=${CUDA_MODULE}"
echo "official_weight=${OFFICIAL_WEIGHT:-${WEIGHT_DIR}/pretrain-concerto-v1m1-2-large-video.pth}"
echo "official_config_stem=${OFFICIAL_CONFIG_STEM:-pretrain-concerto-v1m1-2-large-video-probe-enc2d}"
echo "arkit_root=${ARKIT_FULL_META_ROOT}"
echo "scannet_imagepoint_root=${SCANNET_IMAGEPOINT_ROOT}"
echo "scannet_imagepoint_meta_root=${SCANNET_IMAGEPOINT_META_ROOT}"
echo "max_batches=${MAX_BATCHES:-64}"
echo "batch_size=${BATCH_SIZE:-1}"
echo "num_worker=${NUM_WORKER:-2}"
nvidia-smi -L || true
nvcc --version || true

"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
PY

bash tools/concerto_projection_shortcut/run_official_causal_battery.sh

echo "[done] official causal battery completed"
echo "[log] ${LOG_PATH}"
