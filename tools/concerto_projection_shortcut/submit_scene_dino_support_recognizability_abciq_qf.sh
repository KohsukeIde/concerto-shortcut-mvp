#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=01:30:00
#PBS -N dino_supp2d
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/logs/

set -euo pipefail

cd "${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}" || exit 1

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

source tools/concerto_projection_shortcut/device_defaults.sh
ensure_venv_active

mkdir -p tools/concerto_projection_shortcut/logs
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

OUT_DIR="${OUT_DIR:-tools/concerto_projection_shortcut/results_scene_dino_support_recognizability}"
DATA_ROOT_A="${DATA_ROOT_A:-${SCANNET_IMAGEPOINT_META_ROOT}}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov2-with-registers-giant}"
CONDITIONS="${CONDITIONS:-clean,random_keep80,random_keep50,random_keep20,random_keep10,structured_keep80,structured_keep50,structured_keep20,structured_keep10,instance_keep20}"

mkdir -p "${OUT_DIR}"

echo "=== Scene DINO support recognizability ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "out_dir=${OUT_DIR}"
echo "data_root=${DATA_ROOT_A}"
echo "model=${MODEL_NAME}"
echo "conditions=${CONDITIONS}"
nvidia-smi -L || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_scene_dino_support_recognizability.py \
  --data-root "${DATA_ROOT_A}" \
  --output-dir "${OUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --conditions "${CONDITIONS}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --max-train-images "${MAX_TRAIN_IMAGES:-256}" \
  --max-val-images "${MAX_VAL_IMAGES:-128}" \
  --max-per-class "${MAX_PER_CLASS:-1200}" \
  --patches-per-image-per-class "${PATCHES_PER_IMAGE_PER_CLASS:-6}" \
  --min-points-per-patch "${MIN_POINTS_PER_PATCH:-4}" \
  --majority-threshold "${MAJORITY_THRESHOLD:-0.6}" \
  --structured-cell-size "${STRUCTURED_CELL_SIZE:-1.28}" \
  --steps "${STEPS:-800}"

cp "${OUT_DIR}/scene_dino_support_recognizability.md" \
  tools/concerto_projection_shortcut/results_scene_dino_support_recognizability.md
cp "${OUT_DIR}/scene_dino_support_recognizability.csv" \
  tools/concerto_projection_shortcut/results_scene_dino_support_recognizability.csv

echo "[done] ${OUT_DIR}"
