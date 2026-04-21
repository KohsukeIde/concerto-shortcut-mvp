#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=01:30:00
#PBS -N main6eval
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

WEIGHT_PATH="${WEIGHT_PATH:-${REPO_ROOT}/data/runs/main_variant_enc2d_headfit/main-origin-six-step05/model_last.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/runs/main_variant_enc2d_headfit/main-origin-six-step05-all6eval-rerun2}"
DATASETS="${DATASETS:-arkit,scannet,scannetpp,s3dis,hm3d,structured3d}"
NUM_WORKER="${NUM_WORKER:-2}"
MAX_VAL_BATCHES_PER_DATASET="${MAX_VAL_BATCHES_PER_DATASET:-32}"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/main_variant_all6eval_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ABCI-Q main-variant six-dataset eval-only ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "weight_path=${WEIGHT_PATH}"
echo "output_root=${OUTPUT_ROOT}"
echo "datasets=${DATASETS}"
echo "max_val_batches_per_dataset=${MAX_VAL_BATCHES_PER_DATASET}"

"${PYTHON_BIN}" tools/concerto_projection_shortcut/fit_main_variant_enc2d_head.py \
  --repo-root "${REPO_ROOT}" \
  --weight "${WEIGHT_PATH}" \
  --datasets "${DATASETS}" \
  --eval-datasets "${DATASETS}" \
  --output-root "${OUTPUT_ROOT}" \
  --tag "$(basename "${OUTPUT_ROOT}")" \
  --epochs 0 \
  --max-train-batches-per-dataset 128 \
  --max-val-batches-per-dataset "${MAX_VAL_BATCHES_PER_DATASET}" \
  --num-worker "${NUM_WORKER}"

echo "[done] main-variant six-dataset eval-only"
echo "[log] ${LOG_FILE}"
