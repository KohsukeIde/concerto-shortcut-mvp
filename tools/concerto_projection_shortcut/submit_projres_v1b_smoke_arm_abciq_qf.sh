#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l select=1
#PBS -l walltime=01:15:00
#PBS -N projres_v1b_smoke
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

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq" "${POINTCEPT_DATA_ROOT}/runs/projres_v1b"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/projres_v1b_smoke_${ARM_NAME:-arm}_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

ensure_venv_active
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export POINTCEPT_TRAIN_LAUNCHER="${POINTCEPT_TRAIN_LAUNCHER:-torchrun}"
export NCCL_STABLE_MODE="${NCCL_STABLE_MODE:-1}"
if [ "${NCCL_STABLE_MODE}" = "1" ]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
fi

echo "=== ABCI-Q ProjRes v1b smoke arm ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "arm=${ARM_NAME:-}"
echo "alpha=${COORD_PROJECTION_ALPHA:-}"
echo "beta=${COORD_PROJECTION_BETA:-}"
echo "walltime=${PBS_WALLTIME:-}"
echo "pointcept_train_launcher=${POINTCEPT_TRAIN_LAUNCHER}"
echo "nccl_stable_mode=${NCCL_STABLE_MODE}"
echo "nccl_p2p_disable=${NCCL_P2P_DISABLE:-}"
echo "nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL:-}"
nvidia-smi -L || true
nvcc --version || true

bash tools/concerto_projection_shortcut/run_projres_v1b_smoke_arm.sh

echo "[done] projres v1b smoke arm completed"
echo "[log] ${LOG_PATH}"
