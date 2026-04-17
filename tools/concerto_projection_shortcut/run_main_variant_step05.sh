#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

MAIN_ORIGIN_WEIGHT="${MAIN_ORIGIN_WEIGHT:-${WEIGHT_DIR}/concerto_base_origin.pth}"
MAIN_VARIANT_TAG="${MAIN_VARIANT_TAG:-main-origin-six-step05}"
DATASETS="${DATASETS:-arkit,scannet,scannetpp,s3dis,hm3d,structured3d}"
EVAL_DATASETS="${EVAL_DATASETS:-arkit,scannet}"
ALLOW_MISSING_DATASETS="${ALLOW_MISSING_DATASETS:-0}"
DRY_RUN="${DRY_RUN:-0}"
HEADFIT_EPOCHS="${HEADFIT_EPOCHS:-1}"
HEADFIT_MAX_TRAIN_BATCHES="${HEADFIT_MAX_TRAIN_BATCHES:-128}"
HEADFIT_MAX_VAL_BATCHES="${HEADFIT_MAX_VAL_BATCHES:-32}"
COORD_MAX_TRAIN_BATCHES="${COORD_MAX_TRAIN_BATCHES:-256}"
COORD_MAX_VAL_BATCHES="${COORD_MAX_VAL_BATCHES:-64}"
MAX_ROWS_PER_BATCH="${MAX_ROWS_PER_BATCH:-512}"
NUM_WORKER="${NUM_WORKER:-2}"
HEADFIT_ROOT="${HEADFIT_ROOT:-${POINTCEPT_DATA_ROOT}/runs/main_variant_enc2d_headfit/${MAIN_VARIANT_TAG}}"
COORD_RIVAL_ROOT="${COORD_RIVAL_ROOT:-${POINTCEPT_DATA_ROOT}/runs/main_variant_coord_mlp_rival/${MAIN_VARIANT_TAG}}"

if [ ! -f "${MAIN_ORIGIN_WEIGHT}" ]; then
  echo "[error] missing main-variant weight: ${MAIN_ORIGIN_WEIGHT}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${MAIN_ORIGIN_WEIGHT}" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
enc2d = sum("enc2d_head" in key or "patch_proj" in key for key in state_dict)
student = sum(key.startswith("student.") or key.startswith("module.student.") for key in state_dict)
print(f"[weight] {path} keys={len(state_dict)} enc2d_or_patch_keys={enc2d} student_keys={student}")
if enc2d:
    raise SystemExit("[error] expected concerto_base_origin.pth to be bare-backbone; got enc2d/patch keys")
PY

VERIFY_ARGS=()
if [ "${ALLOW_MISSING_DATASETS}" = "1" ]; then
  VERIFY_ARGS+=(--allow-missing)
fi
"${PYTHON_BIN}" tools/concerto_projection_shortcut/verify_concerto_six_datasets.py "${VERIFY_ARGS[@]}"

COMMON_ARGS=()
if [ "${ALLOW_MISSING_DATASETS}" = "1" ]; then
  COMMON_ARGS+=(--allow-missing-datasets)
fi
if [ "${DRY_RUN}" = "1" ]; then
  COMMON_ARGS+=(--dry-run)
fi

echo "[stage] main-variant frozen-backbone enc2d head-refit"
"${PYTHON_BIN}" tools/concerto_projection_shortcut/fit_main_variant_enc2d_head.py \
  --repo-root "${REPO_ROOT}" \
  --weight "${MAIN_ORIGIN_WEIGHT}" \
  --datasets "${DATASETS}" \
  --eval-datasets "${EVAL_DATASETS}" \
  --output-root "${HEADFIT_ROOT}" \
  --tag "${MAIN_VARIANT_TAG}" \
  --epochs "${HEADFIT_EPOCHS}" \
  --max-train-batches-per-dataset "${HEADFIT_MAX_TRAIN_BATCHES}" \
  --max-val-batches-per-dataset "${HEADFIT_MAX_VAL_BATCHES}" \
  --num-worker "${NUM_WORKER}" \
  "${COMMON_ARGS[@]}"

echo "[stage] main-variant coord-MLP rival fit"
"${PYTHON_BIN}" tools/concerto_projection_shortcut/fit_main_variant_coord_mlp_rival.py \
  --repo-root "${REPO_ROOT}" \
  --weight "${MAIN_ORIGIN_WEIGHT}" \
  --datasets "${DATASETS}" \
  --output-root "${COORD_RIVAL_ROOT}" \
  --tag "${MAIN_VARIANT_TAG}" \
  --max-train-batches-per-dataset "${COORD_MAX_TRAIN_BATCHES}" \
  --max-val-batches-per-dataset "${COORD_MAX_VAL_BATCHES}" \
  --max-rows-per-batch "${MAX_ROWS_PER_BATCH}" \
  --num-worker "${NUM_WORKER}" \
  --causal-csv "${HEADFIT_ROOT}/results_main_variant_causal_battery.csv" \
  "${COMMON_ARGS[@]}"

echo "[done] main-variant Step 0/0.5 diagnostics"
echo "[headfit] ${HEADFIT_ROOT}"
echo "[coord-rival] ${COORD_RIVAL_ROOT}"
