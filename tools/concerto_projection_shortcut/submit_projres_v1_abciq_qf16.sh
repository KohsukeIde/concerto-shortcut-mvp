#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=4
#PBS -l walltime=06:00:00
#PBS -N projres_v1_qf16
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

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq" "${POINTCEPT_DATA_ROOT}/runs/projres_v1"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/projres_v1_qf16_${PBS_JOBID:-manual}.log}"
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

GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1}"
EXP_TAG="${EXP_TAG:--h10016}"
BASE_CONFIG="${BASE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-continue}"
SMOKE_CONFIG="${SMOKE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-projres-v1a-smoke-h10016}"
CONTINUE_CONFIG="${CONTINUE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-projres-v1a-continue-h10016}"
CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-32}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-3}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-64}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-1024}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-256}"
MAX_ROWS_PER_BATCH="${MAX_ROWS_PER_BATCH:-512}"
RUN_FAST_GATE="${RUN_FAST_GATE:-1}"

export GPU_IDS_CSV EXP_MIRROR_ROOT EXP_TAG BASE_CONFIG SMOKE_CONFIG CONTINUE_CONFIG
export CONCERTO_GLOBAL_BATCH_SIZE CONCERTO_GRAD_ACCUM CONCERTO_NUM_WORKER
export MAX_TRAIN_BATCHES MAX_VAL_BATCHES MAX_ROWS_PER_BATCH
export MULTINODE_TRAIN=1
export SMOKE_PARALLEL=0

echo "=== ABCI-Q ProjRes v1 H100x16 chain ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "pbs_nodefile=${PBS_NODEFILE:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "gpu_ids=${GPU_IDS_CSV}"
echo "exp_mirror_root=${EXP_MIRROR_ROOT}"
echo "exp_tag=${EXP_TAG}"
echo "base_config=${BASE_CONFIG}"
echo "smoke_config=${SMOKE_CONFIG}"
echo "continue_config=${CONTINUE_CONFIG}"
echo "concerto_global_batch_size=${CONCERTO_GLOBAL_BATCH_SIZE}"
echo "concerto_grad_accum=${CONCERTO_GRAD_ACCUM}"
echo "concerto_num_worker=${CONCERTO_NUM_WORKER}"
echo "run_fast_gate=${RUN_FAST_GATE}"
echo "pointcept_train_launcher=${POINTCEPT_TRAIN_LAUNCHER}"
echo "nccl_stable_mode=${NCCL_STABLE_MODE}"
echo "nccl_p2p_disable=${NCCL_P2P_DISABLE:-}"
echo "nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL:-}"
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
  echo "nodes:"
  awk '!seen[$0]++{print}' "${PBS_NODEFILE}" | nl -ba
fi
nvidia-smi -L || true

"${PYTHON_BIN}" - <<'PY'
import importlib
import torch
print("python import: OK")
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("flash_attn", getattr(importlib.import_module("flash_attn"), "__version__", "OK"))
PY

bash tools/concerto_projection_shortcut/check_setup_status.sh

"${PYTHON_BIN}" tools/concerto_projection_shortcut/preflight.py \
  --check-data --check-batch --check-forward \
  --config "${BASE_CONFIG}" \
  --data-root "${ARKIT_FULL_META_ROOT}"

DRY_RUN=1 \
bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh

if [ "${RUN_FAST_GATE}" != "1" ]; then
  echo "[done] dry-run only"
  echo "[log] ${LOG_PATH}"
  exit 0
fi

bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh

echo "[done] projres v1 H100x16 fast gate completed"
echo "[log] ${LOG_PATH}"
