#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -N projres_smoke_qf1
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
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/projres_v1_smoke_qf1_${PBS_JOBID:-manual}.log}"
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
export POINTCEPT_TRACE_STEPS="${POINTCEPT_TRACE_STEPS:-0}"
export POINTCEPT_TRACE_FORWARD="${POINTCEPT_TRACE_FORWARD:-0}"
export POINTCEPT_TRACE_FORWARD_ITER="${POINTCEPT_TRACE_FORWARD_ITER:-}"
NCCL_STABLE_MODE="${NCCL_STABLE_MODE:-1}"
if [ "${NCCL_STABLE_MODE}" = "1" ]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
fi

GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1}"
EXP_TAG="${EXP_TAG:--h10016-qf1}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}}"
SMOKE_CONFIG="${SMOKE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-projres-v1a-smoke-h10016}"
CONTINUE_CONFIG="${CONTINUE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-projres-v1a-continue-h10016}"
ALPHAS_CSV="${ALPHAS_CSV:-0.05,0.10}"
COORD_PROJECTION_BETA="${COORD_PROJECTION_BETA:-1.0}"

# Keep per-GPU batch at 2 on a single rt_QF node, and cap the smoke to a
# short sanity run so a broken job does not burn a long walltime allocation.
CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-8}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-12}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-1}"
CONCERTO_MAX_TRAIN_ITER="${CONCERTO_MAX_TRAIN_ITER:-64}"
CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"

MULTINODE_TRAIN="${MULTINODE_TRAIN:-0}"
SMOKE_ALL_GPUS="${SMOKE_ALL_GPUS:-1}"
SMOKE_PARALLEL="${SMOKE_PARALLEL:-0}"
STOP_AFTER_SMOKE="${STOP_AFTER_SMOKE:-1}"

export GPU_IDS_CSV EXP_MIRROR_ROOT EXP_TAG SUMMARY_ROOT SMOKE_CONFIG CONTINUE_CONFIG ALPHAS_CSV
export COORD_PROJECTION_BETA
export CONCERTO_GLOBAL_BATCH_SIZE CONCERTO_GRAD_ACCUM CONCERTO_NUM_WORKER CONCERTO_MAX_TRAIN_ITER CONCERTO_ENABLE_FLASH
export MULTINODE_TRAIN SMOKE_ALL_GPUS SMOKE_PARALLEL STOP_AFTER_SMOKE NCCL_STABLE_MODE

echo "=== ABCI-Q ProjRes v1 single-node smoke ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "gpu_ids=${GPU_IDS_CSV}"
echo "exp_mirror_root=${EXP_MIRROR_ROOT}"
echo "exp_tag=${EXP_TAG}"
echo "summary_root=${SUMMARY_ROOT}"
echo "smoke_config=${SMOKE_CONFIG}"
echo "alphas=${ALPHAS_CSV}"
echo "coord_projection_beta=${COORD_PROJECTION_BETA}"
echo "concerto_global_batch_size=${CONCERTO_GLOBAL_BATCH_SIZE}"
echo "concerto_grad_accum=${CONCERTO_GRAD_ACCUM}"
echo "concerto_num_worker=${CONCERTO_NUM_WORKER}"
echo "concerto_max_train_iter=${CONCERTO_MAX_TRAIN_ITER}"
echo "concerto_enable_flash=${CONCERTO_ENABLE_FLASH}"
echo "smoke_all_gpus=${SMOKE_ALL_GPUS}"
echo "smoke_parallel=${SMOKE_PARALLEL}"
echo "pointcept_train_launcher=${POINTCEPT_TRAIN_LAUNCHER}"
echo "pointcept_trace_steps=${POINTCEPT_TRACE_STEPS:-0}"
echo "pointcept_trace_forward=${POINTCEPT_TRACE_FORWARD:-0}"
echo "pointcept_trace_forward_iter=${POINTCEPT_TRACE_FORWARD_ITER:-}"
echo "nccl_stable_mode=${NCCL_STABLE_MODE}"
echo "nccl_p2p_disable=${NCCL_P2P_DISABLE:-}"
echo "nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL:-}"
nvidia-smi -L || true

bash tools/concerto_projection_shortcut/check_setup_status.sh

bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh

echo "[done] projres v1 single-node smoke completed"
echo "[log] ${LOG_PATH}"
