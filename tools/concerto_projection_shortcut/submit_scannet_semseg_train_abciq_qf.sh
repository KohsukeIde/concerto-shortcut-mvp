#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=00:40:00
#PBS -N scn_semseg
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

export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export POINTCEPT_TRAIN_LAUNCHER="${POINTCEPT_TRAIN_LAUNCHER:-torchrun}"
export NCCL_STABLE_MODE="${NCCL_STABLE_MODE:-1}"
if [ "${NCCL_STABLE_MODE}" = "1" ]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
fi

DATASET_NAME="${DATASET_NAME:-concerto}"
CONFIG_NAME="${CONFIG_NAME:?CONFIG_NAME is required}"
EXP_NAME="${EXP_NAME:?EXP_NAME is required}"
WEIGHT_PATH="${WEIGHT_PATH:-${WEIGHT_DIR}/concerto_base_origin.pth}"
TRAIN_RESUME="${TRAIN_RESUME:-false}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/scannet_semseg_origin}"
LOG_DIR="${LOG_DIR:-${EXP_MIRROR_ROOT}/logs}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"

if [ "${WEIGHT_PATH}" != "None" ] && [ ! -f "${WEIGHT_PATH}" ] && [ "${TRAIN_RESUME}" != "true" ]; then
  echo "[error] missing weight: ${WEIGHT_PATH}" >&2
  exit 2
fi

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq" "${EXP_MIRROR_ROOT}" "${LOG_DIR}"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/scannet_semseg_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export DATASET_NAME CONFIG_NAME EXP_NAME WEIGHT_PATH TRAIN_RESUME
export GPU_IDS_CSV NPROC_PER_NODE LOG_DIR EXP_MIRROR_ROOT

echo "=== ABCI-Q ScanNet semseg train ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "pbs_nodefile=${PBS_NODEFILE:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "config=${CONFIG_NAME}"
echo "exp=${EXP_NAME}"
echo "weight=${WEIGHT_PATH}"
echo "resume=${TRAIN_RESUME}"
echo "gpu_ids=${GPU_IDS_CSV}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "exp_mirror_root=${EXP_MIRROR_ROOT}"
echo "pointcept_train_launcher=${POINTCEPT_TRAIN_LAUNCHER}"
echo "run_preflight=${RUN_PREFLIGHT}"
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
  echo "nodes:"
  awk '!seen[$0]++{print}' "${PBS_NODEFILE}" | nl -ba
fi
nvidia-smi -L || true
nvcc --version || true

EXP_LINK="exp/${DATASET_NAME}/${EXP_NAME}"
EXP_TARGET="${EXP_MIRROR_ROOT}/exp/${EXP_NAME}"
mkdir -p "exp/${DATASET_NAME}" "${EXP_TARGET}"
if [ ! -e "${EXP_LINK}" ]; then
  ln -s "${EXP_TARGET}" "${EXP_LINK}"
fi

if [ "${TRAIN_RESUME}" != "true" ] && [ -f "${EXP_LINK}/model/model_last.pth" ]; then
  echo "[done] checkpoint already exists: ${EXP_LINK}/model/model_last.pth"
  echo "[log] ${LOG_PATH}"
  exit 0
fi

if [ "${RUN_PREFLIGHT}" = "1" ] && [ "${TRAIN_RESUME}" != "true" ]; then
  echo "[preflight] build config/model only"
  "${PYTHON_BIN}" - <<PY
from pointcept.utils.config import Config
from pointcept.models import build_model
cfg = Config.fromfile("configs/${DATASET_NAME}/${CONFIG_NAME}.py")
model = build_model(cfg.model)
print(type(model).__name__)
print("batch_size", cfg.batch_size, "epoch", cfg.epoch, "eval_epoch", cfg.eval_epoch)
PY
fi

bash tools/concerto_projection_shortcut/run_pointcept_train_multinode_pbsdsh.sh

echo "[done] ScanNet semseg train completed"
echo "[log] ${LOG_PATH}"
