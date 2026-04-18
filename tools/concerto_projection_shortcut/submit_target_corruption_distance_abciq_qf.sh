#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=00:45:00
#PBS -N target_dist
#PBS -j oe

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}
cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}"
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
export PYTHON_BIN="${VENV_DIR}/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/target_corruption_distance_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ABCI-Q target corruption distance ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "datasets=${TARGET_DISTANCE_DATASETS:-arkit,scannet}"
echo "tag=${TARGET_DISTANCE_TAG:-main-origin-six-step05}"
echo "max_batches=${TARGET_DISTANCE_MAX_BATCHES:-32}"
echo "max_rows_per_batch=${TARGET_DISTANCE_MAX_ROWS_PER_BATCH:-4096}"
nvidia-smi -L || true
nvcc --version || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/measure_target_corruption_distance.py \
  --repo-root "${REPO_ROOT}" \
  --weight "${TARGET_DISTANCE_WEIGHT:-${WEIGHT_DIR}/concerto_base_origin.pth}" \
  --datasets "${TARGET_DISTANCE_DATASETS:-arkit,scannet}" \
  --tag "${TARGET_DISTANCE_TAG:-main-origin-six-step05}" \
  --max-batches-per-dataset "${TARGET_DISTANCE_MAX_BATCHES:-32}" \
  --max-rows-per-batch "${TARGET_DISTANCE_MAX_ROWS_PER_BATCH:-4096}" \
  --batch-size "${TARGET_DISTANCE_BATCH_SIZE:-4}" \
  --num-worker "${TARGET_DISTANCE_NUM_WORKER:-2}" \
  --seed "${TARGET_DISTANCE_SEED:-0}"

echo "[done] target corruption distance"
echo "[log] ${LOG_FILE}"
