#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l select=1
#PBS -l walltime=00:08:00
#PBS -N projres_ddp_batch
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
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/debug_arkit_ddp_batches_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_STABLE_MODE="${NCCL_STABLE_MODE:-1}"
if [ "${NCCL_STABLE_MODE}" = "1" ]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
fi

GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS_CSV}"
NUM_GPUS="$(awk -F',' '{print NF}' <<<"${GPU_IDS_CSV}")"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((10000 + 0x$(echo -n "projres/ddp-batch/${PBS_JOBID:-manual}" | md5sum | cut -c 1-4) % 20000))}"

DEBUG_CONFIG="${DEBUG_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-projres-v1a-smoke-h10016}"
DEBUG_MAX_BATCHES="${DEBUG_MAX_BATCHES:-16}"
DEBUG_BATCH_SIZE="${DEBUG_BATCH_SIZE:-8}"
DEBUG_NUM_WORKER="${DEBUG_NUM_WORKER:-1}"
DEBUG_DIST_TIMEOUT_SEC="${DEBUG_DIST_TIMEOUT_SEC:-180}"
DEBUG_SAVE_PATH="${DEBUG_SAVE_PATH:-${POINTCEPT_DATA_ROOT}/runs/projres_v1/debug/ddp_batches_${PBS_JOBID:-manual}}"

echo "=== ABCI-Q ARKit DDP batch diagnostic ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "python_module=${PYTHON_MODULE}"
echo "cuda_module=${CUDA_MODULE}"
echo "gpu_ids=${GPU_IDS_CSV}"
echo "num_gpus=${NUM_GPUS}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "debug_config=${DEBUG_CONFIG}"
echo "debug_max_batches=${DEBUG_MAX_BATCHES}"
echo "debug_batch_size=${DEBUG_BATCH_SIZE}"
echo "debug_num_worker=${DEBUG_NUM_WORKER}"
echo "debug_save_path=${DEBUG_SAVE_PATH}"
echo "nccl_stable_mode=${NCCL_STABLE_MODE}"
echo "nccl_p2p_disable=${NCCL_P2P_DISABLE:-}"
echo "nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL:-}"
nvidia-smi -L || true
nvcc --version || true

python -m torch.distributed.run \
  --nproc_per_node="${NUM_GPUS}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  tools/concerto_projection_shortcut/debug_arkit_ddp_batches.py \
  --config "${DEBUG_CONFIG}" \
  --data-root "${ARKIT_FULL_META_ROOT}" \
  --save-path "${DEBUG_SAVE_PATH}" \
  --max-batches "${DEBUG_MAX_BATCHES}" \
  --batch-size "${DEBUG_BATCH_SIZE}" \
  --num-worker "${DEBUG_NUM_WORKER}" \
  --dist-timeout-sec "${DEBUG_DIST_TIMEOUT_SEC}"

echo "[done] ARKit DDP batch diagnostic completed"
echo "[log] ${LOG_PATH}"
