#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME

OUT_DIR="${OUT_DIR:-${POINTCEPT_DATA_ROOT}/runs/concerto3d_patch_separation_stepA/scannet_pic_wall_smoke}"
DATA_ROOT_A="${DATA_ROOT_A:-${SCANNET_IMAGEPOINT_META_ROOT}}"
CONFIG_NAME="${CONFIG_NAME:-pretrain-concerto-v1m1-2-large-video}"
WEIGHT_PATH="${WEIGHT_PATH:-${WEIGHT_DIR}/pretrain-concerto-v1m1-2-large-video.pth}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-128}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKER="${NUM_WORKER:-4}"
MAX_SEM_PER_CLASS="${MAX_SEM_PER_CLASS:-8000}"
MIN_POINTS_PER_PATCH="${MIN_POINTS_PER_PATCH:-4}"
MAJORITY_THRESHOLD="${MAJORITY_THRESHOLD:-0.6}"
LOGREG_STEPS="${LOGREG_STEPS:-600}"

mkdir -p "${OUT_DIR}"

echo "=== Concerto 3D patch separation Step A ==="
echo "date=$(date -Is)"
echo "repo=${REPO_ROOT}"
echo "out_dir=${OUT_DIR}"
echo "data_root=${DATA_ROOT_A}"
echo "config=${CONFIG_NAME}"
echo "weight=${WEIGHT_PATH}"

"${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_concerto3d_patch_separation_stepA.py \
  --config "${CONFIG_NAME}" \
  --weight "${WEIGHT_PATH}" \
  --data-root "${DATA_ROOT_A}" \
  --output-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-worker "${NUM_WORKER}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-sem-per-class "${MAX_SEM_PER_CLASS}" \
  --min-points-per-patch "${MIN_POINTS_PER_PATCH}" \
  --majority-threshold "${MAJORITY_THRESHOLD}" \
  --logreg-steps "${LOGREG_STEPS}"

SUMMARY_MD="${REPO_ROOT}/tools/concerto_projection_shortcut/results_concerto3d_patch_separation_stepA.md"
SUMMARY_CSV="${REPO_ROOT}/tools/concerto_projection_shortcut/results_concerto3d_patch_separation_stepA.csv"
cp "${OUT_DIR}/concerto3d_patch_separation_stepA.md" "${SUMMARY_MD}"
cp "${OUT_DIR}/concerto3d_patch_separation_stepA.csv" "${SUMMARY_CSV}"
echo "[write] ${SUMMARY_MD}"
echo "[write] ${SUMMARY_CSV}"
