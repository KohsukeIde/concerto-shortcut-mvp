#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=00:30:00
#PBS -N scn_conf
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

CONFIG="${CONFIG:?CONFIG is required}"
WEIGHT="${WEIGHT:?WEIGHT is required}"
LABEL="${LABEL:?LABEL is required}"
OUT_DIR="${OUT_DIR:-${POINTCEPT_DATA_ROOT}/runs/scannet_confusion_one/${LABEL}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKER="${NUM_WORKER:-8}"
MAX_BATCHES="${MAX_BATCHES:--1}"
GPU_ID="${GPU_ID:-0}"

if [ ! -f "${WEIGHT}" ]; then
  echo "[error] missing weight: ${WEIGHT}" >&2
  exit 2
fi

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq" "${OUT_DIR}"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/scannet_confusion_one_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

echo "=== ABCI-Q ScanNet one-checkpoint confusion ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "python=$(command -v python)"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "label=${LABEL}"
echo "out_dir=${OUT_DIR}"
echo "batch_size=${BATCH_SIZE}"
echo "num_worker=${NUM_WORKER}"
echo "max_batches=${MAX_BATCHES}"
echo "gpu_id=${GPU_ID}"
nvidia-smi -L || true
nvcc --version || true

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_scannet_semseg_confusion.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --label "${LABEL}" \
  --output-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-worker "${NUM_WORKER}" \
  --max-batches "${MAX_BATCHES}"

echo "[done] ScanNet one-checkpoint confusion"
echo "[log] ${LOG_PATH}"
