#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -N projres_v1_qf
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
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/projres_v1_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

ensure_venv_active
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-1024}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-256}"
MAX_ROWS_PER_BATCH="${MAX_ROWS_PER_BATCH:-512}"
RUN_FAST_GATE="${RUN_FAST_GATE:-1}"

echo "=== ABCI-Q ProjRes v1 chain ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "gpu_ids=${GPU_IDS_CSV}"
echo "exp_mirror_root=${EXP_MIRROR_ROOT}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "max_rows_per_batch=${MAX_ROWS_PER_BATCH}"
echo "run_fast_gate=${RUN_FAST_GATE}"
nvidia-smi -L || true

bash tools/concerto_projection_shortcut/check_setup_status.sh

"${PYTHON_BIN}" tools/concerto_projection_shortcut/preflight.py \
  --check-data --check-batch --check-forward \
  --config pretrain-concerto-v1m1-0-arkit-full-continue \
  --data-root "${ARKIT_FULL_META_ROOT}"

EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT}" \
GPU_IDS_CSV="${GPU_IDS_CSV}" \
DRY_RUN=1 \
bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh

if [ "${RUN_FAST_GATE}" != "1" ]; then
  echo "[done] dry-run only"
  echo "[log] ${LOG_PATH}"
  exit 0
fi

EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT}" \
GPU_IDS_CSV="${GPU_IDS_CSV}" \
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES}" \
MAX_VAL_BATCHES="${MAX_VAL_BATCHES}" \
MAX_ROWS_PER_BATCH="${MAX_ROWS_PER_BATCH}" \
bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh

echo "[done] projres v1 fast gate completed"
echo "[log] ${LOG_PATH}"
