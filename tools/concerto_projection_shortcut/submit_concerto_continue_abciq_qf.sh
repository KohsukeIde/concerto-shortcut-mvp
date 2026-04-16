#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=8
#PBS -l walltime=03:50:00
#PBS -N concerto_cont
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

EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_long}"
mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq" "${EXP_MIRROR_ROOT}"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/concerto_continue_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

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
LOG_DIR="${LOG_DIR:-${EXP_MIRROR_ROOT}/logs}"
CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-32}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-3}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-64}"
CONCERTO_MAX_TRAIN_ITER="${CONCERTO_MAX_TRAIN_ITER:-0}"
CONCERTO_EPOCH="${CONCERTO_EPOCH:-25}"
CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

if [ ! -f "${WEIGHT_PATH}" ] && [ "${TRAIN_RESUME}" != "true" ]; then
  echo "[error] missing weight: ${WEIGHT_PATH}" >&2
  exit 2
fi

export DATASET_NAME CONFIG_NAME EXP_NAME WEIGHT_PATH TRAIN_RESUME
export GPU_IDS_CSV NPROC_PER_NODE LOG_DIR EXP_MIRROR_ROOT
export CONCERTO_GLOBAL_BATCH_SIZE CONCERTO_GRAD_ACCUM CONCERTO_NUM_WORKER
export CONCERTO_MAX_TRAIN_ITER CONCERTO_EPOCH CONCERTO_ENABLE_FLASH
export COORD_PRIOR_PATH="${COORD_PRIOR_PATH:-}"
export COORD_PROJECTION_ALPHA="${COORD_PROJECTION_ALPHA:-0.0}"
export COORD_PROJECTION_BETA="${COORD_PROJECTION_BETA:-1.0}"

echo "=== ABCI-Q Concerto multi-node continuation ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
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
echo "exp_mirror_root=${EXP_MIRROR_ROOT}"
echo "concerto_global_batch_size=${CONCERTO_GLOBAL_BATCH_SIZE}"
echo "concerto_grad_accum=${CONCERTO_GRAD_ACCUM}"
echo "concerto_num_worker=${CONCERTO_NUM_WORKER}"
echo "concerto_max_train_iter=${CONCERTO_MAX_TRAIN_ITER}"
echo "concerto_epoch=${CONCERTO_EPOCH}"
echo "concerto_enable_flash=${CONCERTO_ENABLE_FLASH}"
echo "coord_prior_path=${COORD_PRIOR_PATH}"
echo "coord_projection_alpha=${COORD_PROJECTION_ALPHA}"
echo "coord_projection_beta=${COORD_PROJECTION_BETA}"
echo "pointcept_train_launcher=${POINTCEPT_TRAIN_LAUNCHER}"
echo "nccl_stable_mode=${NCCL_STABLE_MODE}"
echo "nccl_p2p_disable=${NCCL_P2P_DISABLE:-}"
echo "nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL:-}"
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
  env CUDA_VISIBLE_DEVICES="$(awk -F',' '{print $1}' <<< "${GPU_IDS_CSV}")" \
    CONCERTO_GLOBAL_BATCH_SIZE=1 \
    CONCERTO_NUM_WORKER=0 \
    "${PYTHON_BIN}" tools/concerto_projection_shortcut/preflight.py \
    --check-data --check-batch --check-forward --config "${CONFIG_NAME}" \
    --data-root "${ARKIT_FULL_META_ROOT}"
fi

bash tools/concerto_projection_shortcut/run_pointcept_train_multinode_pbsdsh.sh

echo "[done] concerto continuation completed"
echo "[log] ${LOG_PATH}"
