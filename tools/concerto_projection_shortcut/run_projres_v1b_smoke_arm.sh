#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

device_count() {
  awk -F',' '{print NF}' <<< "$1"
}

selected_prior_path_from_json() {
  local path="$1"
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["selected_path"])
PY
}

ensure_mirror_exp() {
  local exp_name="$1"
  local link_path="${REPO_ROOT}/exp/${DATASET_NAME}/${exp_name}"
  local target_path="${EXP_MIRROR_ROOT}/exp/${exp_name}"
  mkdir -p "${REPO_ROOT}/exp/${DATASET_NAME}" "${target_path}"
  if [ -L "${link_path}" ]; then
    return 0
  fi
  if [ -e "${link_path}" ]; then
    echo "[$(timestamp)] keep existing non-symlink exp: ${link_path}" >&2
    return 0
  fi
  ln -s "${target_path}" "${link_path}"
}

summarize_smoke() {
  local log_path="$1"
  local out_json="$2"
  "${PYTHON_BIN}" - \
    "${log_path}" \
    "${out_json}" \
    "${ARM_NAME}" \
    "${EXP_NAME}" \
    "${COORD_PROJECTION_ALPHA}" \
    "${COORD_PROJECTION_BETA}" \
    "${MIN_RESIDUAL_NORM}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_json = Path(sys.argv[2])
arm_name = sys.argv[3]
exp_name = sys.argv[4]
alpha = float(sys.argv[5])
beta = float(sys.argv[6])
min_residual_norm = float(sys.argv[7])

keys = [
    "loss",
    "enc2d_loss",
    "coord_residual_enc2d_loss",
    "coord_alignment_loss",
    "coord_target_energy",
    "coord_removed_energy",
    "coord_pred_energy",
    "coord_residual_norm",
    "coord_projection_loss_check",
]
values = {key: [] for key in keys}
pattern = re.compile(r"([A-Za-z0-9_]+):\s*([-+0-9.eE]+)")
if log_path.exists():
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Train:" not in line and "Train result:" not in line:
            continue
        for key, value in pattern.findall(line):
            if key in values:
                try:
                    values[key].append(float(value))
                except ValueError:
                    pass

def finite(seq):
    return bool(seq) and all(math.isfinite(x) for x in seq)

def last(key, default=None):
    return values[key][-1] if values[key] else default

payload = {
    "arm": arm_name,
    "exp": exp_name,
    "alpha": alpha,
    "beta": beta,
    "log": str(log_path),
    "pass": False,
    "reason": "missing_metrics",
    "first": {},
    "last": {},
}
for key, seq in values.items():
    if seq:
        payload["first"][key] = seq[0]
        payload["last"][key] = seq[-1]

required = [
    "loss",
    "enc2d_loss",
    "coord_residual_enc2d_loss",
    "coord_alignment_loss",
    "coord_pred_energy",
    "coord_residual_norm",
]
loss_check = values["coord_projection_loss_check"]
if all(finite(values[key]) for key in required):
    enc_last = last("enc2d_loss")
    residual_last = last("coord_residual_norm")
    max_loss_check = max(loss_check) if loss_check else float("inf")
    payload["metrics_consistent"] = max_loss_check <= 1e-3
    payload["score"] = (
        enc_last
        + 10.0 * max(0.0, min_residual_norm - residual_last)
        + 25.0 * max(0.0, last("coord_pred_energy") - 0.003)
    )
    if max_loss_check > 1e-3:
        payload["reason"] = "projection_loss_check_failed"
    elif residual_last < min_residual_norm:
        payload["reason"] = "residual_norm_below_min"
    elif enc_last <= 0.01 or enc_last >= 50:
        payload["reason"] = "enc2d_loss_collapse_or_explode"
    else:
        payload["pass"] = True
        payload["reason"] = "pass"

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
if not payload["pass"]:
    raise SystemExit(2)
PY
}

if [ "${SKIP_VENV_ACTIVATE:-0}" != "1" ]; then
  ensure_venv_active
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
fi

DATASET_NAME="${DATASET_NAME:-concerto}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1b}"
EXP_TAG="${EXP_TAG:--h10016-qf1}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}}"
LOG_DIR="${LOG_DIR:-${EXP_MIRROR_ROOT}/logs/smoke}"
CONFIG_NAME="${CONFIG_NAME:-pretrain-concerto-v1m1-0-arkit-full-projres-v1b-smoke-h10016}"
OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${WEIGHT_DIR}/concerto_base_origin.pth}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
ARM_NAME="${ARM_NAME:?ARM_NAME is required}"
COORD_PROJECTION_ALPHA="${COORD_PROJECTION_ALPHA:?COORD_PROJECTION_ALPHA is required}"
COORD_PROJECTION_BETA="${COORD_PROJECTION_BETA:?COORD_PROJECTION_BETA is required}"
SELECTED_PRIOR_JSON="${SELECTED_PRIOR_JSON:-${POINTCEPT_DATA_ROOT}/runs/projres_v1/priors/selected_prior.json}"
COORD_PRIOR_PATH="${COORD_PRIOR_PATH:-$(selected_prior_path_from_json "${SELECTED_PRIOR_JSON}")}"
EXP_NAME="${EXP_NAME:-arkit-full-projres-v1b-${ARM_NAME}${EXP_TAG}-smoke}"
SUMMARY_JSON="${SUMMARY_JSON:-${SUMMARY_ROOT}/${EXP_NAME}.json}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
MIN_RESIDUAL_NORM="${MIN_RESIDUAL_NORM:-0.80}"

CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-8}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-12}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-1}"
CONCERTO_MAX_TRAIN_ITER="${CONCERTO_MAX_TRAIN_ITER:-0}"
CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"

export COORD_PRIOR_PATH COORD_PROJECTION_ALPHA COORD_PROJECTION_BETA
export CONCERTO_GLOBAL_BATCH_SIZE CONCERTO_GRAD_ACCUM CONCERTO_NUM_WORKER
export CONCERTO_MAX_TRAIN_ITER CONCERTO_ENABLE_FLASH

mkdir -p "${SUMMARY_ROOT}" "${LOG_DIR}"

if [ ! -f "${OFFICIAL_WEIGHT}" ]; then
  echo "[$(timestamp)] error: missing official weight: ${OFFICIAL_WEIGHT}" >&2
  exit 2
fi
if [ ! -f "${COORD_PRIOR_PATH}" ]; then
  echo "[$(timestamp)] error: missing coord prior: ${COORD_PRIOR_PATH}" >&2
  exit 2
fi

echo "[$(timestamp)] start projres v1b smoke arm"
echo "arm=${ARM_NAME}"
echo "alpha=${COORD_PROJECTION_ALPHA}"
echo "beta=${COORD_PROJECTION_BETA}"
echo "config=${CONFIG_NAME}"
echo "exp=${EXP_NAME}"
echo "summary_json=${SUMMARY_JSON}"
echo "gpu_ids=${GPU_IDS_CSV}"
echo "coord_prior_path=${COORD_PRIOR_PATH}"
echo "concerto_global_batch_size=${CONCERTO_GLOBAL_BATCH_SIZE}"
echo "concerto_grad_accum=${CONCERTO_GRAD_ACCUM}"
echo "concerto_num_worker=${CONCERTO_NUM_WORKER}"
echo "concerto_max_train_iter=${CONCERTO_MAX_TRAIN_ITER}"

ensure_mirror_exp "${EXP_NAME}"

if [ "${RUN_PREFLIGHT}" = "1" ]; then
  env CUDA_VISIBLE_DEVICES="$(awk -F',' '{print $1}' <<< "${GPU_IDS_CSV}")" \
    CONCERTO_GLOBAL_BATCH_SIZE=1 \
    CONCERTO_NUM_WORKER=0 \
    "${PYTHON_BIN}" tools/concerto_projection_shortcut/preflight.py \
      --check-data --check-batch --check-forward --config "${CONFIG_NAME}" \
      --data-root "${ARKIT_FULL_META_ROOT}"
fi

checkpoint="${REPO_ROOT}/exp/${DATASET_NAME}/${EXP_NAME}/model/model_last.pth"
if [ -f "${checkpoint}" ]; then
  echo "[$(timestamp)] skip train: ${checkpoint}"
else
  CUDA_VISIBLE_DEVICES="${GPU_IDS_CSV}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    bash "${REPO_ROOT}/scripts/train.sh" \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g "$(device_count "${GPU_IDS_CSV}")" \
      -c "${CONFIG_NAME}" \
      -n "${EXP_NAME}" \
      -w "${OFFICIAL_WEIGHT}"
fi

summarize_smoke "${REPO_ROOT}/exp/${DATASET_NAME}/${EXP_NAME}/train.log" "${SUMMARY_JSON}"
echo "[$(timestamp)] done projres v1b smoke arm"
