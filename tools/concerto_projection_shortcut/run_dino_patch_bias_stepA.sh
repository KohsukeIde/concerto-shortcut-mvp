#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME

OUT_DIR="${OUT_DIR:-${POINTCEPT_DATA_ROOT}/runs/dino_patch_bias_stepA/scannet_pic_wall_smoke}"
DATA_ROOT_A="${DATA_ROOT_A:-${SCANNET_IMAGEPOINT_META_ROOT}}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov2-with-registers-giant}"
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-256}"
MAX_VAL_IMAGES="${MAX_VAL_IMAGES:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SEM_PER_CLASS="${MAX_SEM_PER_CLASS:-6000}"
MAX_POS_PATCHES="${MAX_POS_PATCHES:-20000}"
POS_PATCHES_PER_IMAGE="${POS_PATCHES_PER_IMAGE:-64}"
SEM_PATCHES_PER_IMAGE_PER_CLASS="${SEM_PATCHES_PER_IMAGE_PER_CLASS:-8}"
MIN_POINTS_PER_PATCH="${MIN_POINTS_PER_PATCH:-4}"
MAJORITY_THRESHOLD="${MAJORITY_THRESHOLD:-0.6}"
LOGREG_STEPS="${LOGREG_STEPS:-600}"

mkdir -p "${OUT_DIR}"

echo "=== DINO patch bias Step A' ==="
echo "date=$(date -Is)"
echo "repo=${REPO_ROOT}"
echo "out_dir=${OUT_DIR}"
echo "data_root=${DATA_ROOT_A}"
echo "model=${MODEL_NAME}"

"${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_dino_patch_bias_stepA.py \
  --data-root "${DATA_ROOT_A}" \
  --output-dir "${OUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-images "${MAX_TRAIN_IMAGES}" \
  --max-val-images "${MAX_VAL_IMAGES}" \
  --max-sem-per-class "${MAX_SEM_PER_CLASS}" \
  --max-pos-patches "${MAX_POS_PATCHES}" \
  --pos-patches-per-image "${POS_PATCHES_PER_IMAGE}" \
  --sem-patches-per-image-per-class "${SEM_PATCHES_PER_IMAGE_PER_CLASS}" \
  --min-points-per-patch "${MIN_POINTS_PER_PATCH}" \
  --majority-threshold "${MAJORITY_THRESHOLD}" \
  --logreg-steps "${LOGREG_STEPS}"

SUMMARY_MD="${REPO_ROOT}/tools/concerto_projection_shortcut/results_dino_patch_bias_stepA.md"
SUMMARY_CSV="${REPO_ROOT}/tools/concerto_projection_shortcut/results_dino_patch_bias_stepA.csv"
cp "${OUT_DIR}/dino_patch_bias_stepA.md" "${SUMMARY_MD}"
cp "${OUT_DIR}/dino_patch_bias_stepA.csv" "${SUMMARY_CSV}"
echo "[write] ${SUMMARY_MD}"
echo "[write] ${SUMMARY_CSV}"
