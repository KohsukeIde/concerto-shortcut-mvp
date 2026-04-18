#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME

OUT_DIR="${OUT_DIR:-${POINTCEPT_DATA_ROOT}/runs/scannet_point_stagewise_trace/scannet_medium}"
DATA_ROOT_A="${DATA_ROOT_A:-${SCANNET_EXTRACT_DIR}}"
LINEAR_CONFIG="${LINEAR_CONFIG:-exp/concerto/scannet-proxy-large-video-official-lin/config.py}"
LINEAR_WEIGHT="${LINEAR_WEIGHT:-exp/concerto/scannet-proxy-large-video-official-lin/model/model_last.pth}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-256}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKER="${NUM_WORKER:-4}"
MAX_PER_CLASS="${MAX_PER_CLASS:-60000}"
LOGREG_STEPS="${LOGREG_STEPS:-600}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-100}"
CLASS_PAIRS="${CLASS_PAIRS:-picture:wall,counter:cabinet,desk:wall,desk:table,sink:cabinet,door:wall,shower curtain:wall}"

mkdir -p "${OUT_DIR}"

echo "=== ScanNet point-level stage-wise trace ==="
echo "date=$(date -Is)"
echo "repo=${REPO_ROOT}"
echo "out_dir=${OUT_DIR}"
echo "class_pairs=${CLASS_PAIRS}"

"${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_scannet_point_stagewise_trace.py \
  --linear-config "${LINEAR_CONFIG}" \
  --linear-weight "${LINEAR_WEIGHT}" \
  --data-root "${DATA_ROOT_A}" \
  --output-dir "${OUT_DIR}" \
  --class-pairs "${CLASS_PAIRS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-worker "${NUM_WORKER}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --logreg-steps "${LOGREG_STEPS}" \
  --bootstrap-iters "${BOOTSTRAP_ITERS}"

SUMMARY_MD="${REPO_ROOT}/tools/concerto_projection_shortcut/results_scannet_point_stagewise_trace.md"
SUMMARY_CSV="${REPO_ROOT}/tools/concerto_projection_shortcut/results_scannet_point_stagewise_trace.csv"
SUMMARY_CONF="${REPO_ROOT}/tools/concerto_projection_shortcut/results_scannet_point_stagewise_trace_confusion.csv"
cp "${OUT_DIR}/scannet_point_stagewise_trace.md" "${SUMMARY_MD}"
cp "${OUT_DIR}/scannet_point_stagewise_trace.csv" "${SUMMARY_CSV}"
cp "${OUT_DIR}/scannet_point_stagewise_trace_confusion.csv" "${SUMMARY_CONF}"
echo "[write] ${SUMMARY_MD}"
echo "[write] ${SUMMARY_CSV}"
echo "[write] ${SUMMARY_CONF}"
