#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

alpha_tag() {
  local alpha="$1"
  printf '%s' "${alpha}" | tr -d '.'
}

checkpoint_path() {
  local exp_name="$1"
  printf '%s/exp/%s/%s/model/model_last.pth' "${REPO_ROOT}" "${DATASET_NAME}" "${exp_name}"
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

selected_alpha_from_json() {
  local path="$1"
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["selected"]["alpha"])
PY
}

selected_prior_path_from_json() {
  local path="$1"
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["selected_path"])
PY
}

write_linear_gate() {
  local fix_exp="$1"
  local out_json="$2"
  local fix_log="${REPO_ROOT}/exp/${DATASET_NAME}/${fix_exp}/train.log"
  local tmp_csv="${SUMMARY_ROOT}/projres_v1_linear_gate.csv"
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_semseg_logs.py "${fix_log}" > "${tmp_csv}"
  "${PYTHON_BIN}" - \
    "${tmp_csv}" \
    "tools/concerto_projection_shortcut/results_scannet_proxy_lin.csv" \
    "${out_json}" \
    "${LINEAR_GO_MARGIN}" <<'PY'
import csv
import json
import sys
from pathlib import Path

fix_csv = Path(sys.argv[1])
baseline_csv = Path(sys.argv[2])
out_json = Path(sys.argv[3])
margin = float(sys.argv[4])

fix = next(csv.DictReader(fix_csv.open()))
baselines = list(csv.DictReader(baseline_csv.open()))
orig = next(row for row in baselines if "scannet-proxy-concerto-continue-lin" in row["log"])
noenc = next(row for row in baselines if "scannet-proxy-no-enc2d-renorm-continue-lin" in row["log"])

def f(row, key):
    return float(str(row[key]).rstrip("."))

fix_last = f(fix, "val_miou_last")
fix_best = f(fix, "best_metric_value")
orig_last = f(orig, "val_miou_last")
orig_best = f(orig, "best_metric_value")
noenc_last = f(noenc, "val_miou_last")
noenc_best = f(noenc, "best_metric_value")
strong_go = fix_last >= orig_last + margin or fix_best >= orig_best + margin
payload = {
    "strong_go": strong_go,
    "reason": "strong_go" if strong_go else "linear_gate_not_strong_go",
    "margin": margin,
    "fix": {"last_miou": fix_last, "best_miou": fix_best},
    "original": {"last_miou": orig_last, "best_miou": orig_best},
    "no_enc2d_renorm": {"last_miou": noenc_last, "best_miou": noenc_best},
    "delta_vs_original": {
        "last_miou": fix_last - orig_last,
        "best_miou": fix_best - orig_best,
    },
    "delta_vs_no_enc2d_renorm": {
        "last_miou": fix_last - noenc_last,
        "best_miou": fix_best - noenc_best,
    },
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
}

run_stress() {
  if [ "${RUN_STRESS}" != "1" ]; then
    echo "[$(timestamp)] skip stress: RUN_STRESS=${RUN_STRESS}"
    return 0
  fi
  if [ -f "${STRESS_CSV}" ]; then
    echo "[$(timestamp)] skip stress: ${STRESS_CSV}"
    cat "${STRESS_CSV}"
    return 0
  fi
  local tmp_csv="${STRESS_CSV}.tmp"
  echo "[$(timestamp)] stress: gpu=${STRESS_GPU} ${CONTINUE_EXP} -> ${STRESS_CSV}"
  env CUDA_VISIBLE_DEVICES="${STRESS_GPU}" \
      COORD_PRIOR_PATH="${COORD_PRIOR_PATH}" \
      COORD_PROJECTION_ALPHA="${SELECTED_ALPHA}" \
      "${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_enc2d_stress.py \
      --config "${CONTINUE_CONFIG}" \
      --weight "${CONTINUE_CKPT}" \
      --data-root "${ARKIT_FULL_META_ROOT}" \
      --max-batches "${MAX_STRESS_BATCHES}" \
      > "${tmp_csv}"
  mv "${tmp_csv}" "${STRESS_CSV}"
  cat "${STRESS_CSV}"
}

run_linear() {
  if [ "${RUN_LINEAR}" != "1" ]; then
    echo "[$(timestamp)] skip linear: RUN_LINEAR=${RUN_LINEAR}"
    return 0
  fi
  ensure_mirror_exp "${LINEAR_EXP}"
  local linear_ckpt
  linear_ckpt="$(checkpoint_path "${LINEAR_EXP}")"
  if [ -f "${linear_ckpt}" ]; then
    echo "[$(timestamp)] skip linear train: ${linear_ckpt}"
  else
    echo "[$(timestamp)] linear: gpu=${LINEAR_GPU} config=${LINEAR_CONFIG} exp=${LINEAR_EXP}"
    env CUDA_VISIBLE_DEVICES="${LINEAR_GPU}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        POINTCEPT_TRAIN_LAUNCHER="${LINEAR_TRAIN_LAUNCHER}" \
        bash "${REPO_ROOT}/scripts/train.sh" \
          -p "${PYTHON_BIN}" \
          -d "${DATASET_NAME}" \
          -g 1 \
          -c "${LINEAR_CONFIG}" \
          -n "${LINEAR_EXP}" \
          -w "${CONTINUE_CKPT}"
  fi
  echo "[$(timestamp)] linear gate: ${LINEAR_GATE_JSON}"
  write_linear_gate "${LINEAR_EXP}" "${LINEAR_GATE_JSON}"
}

if [ "${SKIP_VENV_ACTIVATE:-0}" != "1" ]; then
  ensure_venv_active
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
fi

DATASET_NAME="${DATASET_NAME:-concerto}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1}"
EXP_TAG="${EXP_TAG:--h10032-qf32}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}}"
LOG_DIR="${LOG_DIR:-${EXP_MIRROR_ROOT}/logs/followup}"
CONTINUE_CONFIG="${CONTINUE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-projres-v1a-continue-h10016}"
LINEAR_CONFIG="${LINEAR_CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly}"
PRIOR_ROOT="${PRIOR_ROOT:-${EXP_MIRROR_ROOT}/priors}"
SELECTED_PRIOR_JSON="${SELECTED_PRIOR_JSON:-${PRIOR_ROOT}/selected_prior.json}"
SELECTED_SMOKE_JSON="${SELECTED_SMOKE_JSON:-${EXP_MIRROR_ROOT}/summaries/h10016-qf1fixed64/selected_smoke.json}"
LINEAR_GO_MARGIN="${LINEAR_GO_MARGIN:-0.01}"
MAX_STRESS_BATCHES="${MAX_STRESS_BATCHES:-20}"
RUN_STRESS="${RUN_STRESS:-1}"
RUN_LINEAR="${RUN_LINEAR:-1}"
FOLLOWUP_PARALLEL="${FOLLOWUP_PARALLEL:-1}"
STRESS_GPU="${STRESS_GPU:-0}"
LINEAR_GPU="${LINEAR_GPU:-1}"
LINEAR_TRAIN_LAUNCHER="${LINEAR_TRAIN_LAUNCHER:-pointcept}"

mkdir -p "${SUMMARY_ROOT}" "${LOG_DIR}"

if [ -z "${SELECTED_ALPHA:-}" ]; then
  SELECTED_ALPHA="$(selected_alpha_from_json "${SELECTED_SMOKE_JSON}")"
fi
SELECTED_TAG="${SELECTED_TAG:-$(alpha_tag "${SELECTED_ALPHA}")}"
CONTINUE_EXP="${CONTINUE_EXP:-arkit-full-projres-v1a-alpha${SELECTED_TAG}${EXP_TAG}-continue}"
LINEAR_EXP="${LINEAR_EXP:-scannet-proxy-projres-v1a-alpha${SELECTED_TAG}${EXP_TAG}-lin}"
CONTINUE_CKPT="${CONTINUE_CKPT:-$(checkpoint_path "${CONTINUE_EXP}")}"
COORD_PRIOR_PATH="${COORD_PRIOR_PATH:-$(selected_prior_path_from_json "${SELECTED_PRIOR_JSON}")}"
STRESS_CSV="${STRESS_CSV:-${SUMMARY_ROOT}/${CONTINUE_EXP}_stress.csv}"
LINEAR_GATE_JSON="${LINEAR_GATE_JSON:-${SUMMARY_ROOT}/${LINEAR_EXP}_gate.json}"

export COORD_PRIOR_PATH SELECTED_ALPHA

echo "[$(timestamp)] start projres v1 follow-up"
echo "exp_mirror_root=${EXP_MIRROR_ROOT}"
echo "summary_root=${SUMMARY_ROOT}"
echo "log_dir=${LOG_DIR}"
echo "selected_alpha=${SELECTED_ALPHA}"
echo "coord_prior_path=${COORD_PRIOR_PATH}"
echo "continue_config=${CONTINUE_CONFIG}"
echo "continue_exp=${CONTINUE_EXP}"
echo "continue_ckpt=${CONTINUE_CKPT}"
echo "linear_config=${LINEAR_CONFIG}"
echo "linear_exp=${LINEAR_EXP}"
echo "stress_csv=${STRESS_CSV}"
echo "linear_gate_json=${LINEAR_GATE_JSON}"
echo "run_stress=${RUN_STRESS}"
echo "run_linear=${RUN_LINEAR}"
echo "followup_parallel=${FOLLOWUP_PARALLEL}"

if [ ! -f "${CONTINUE_CKPT}" ]; then
  echo "[$(timestamp)] error: missing continuation checkpoint: ${CONTINUE_CKPT}" >&2
  exit 2
fi

if [ "${FOLLOWUP_PARALLEL}" = "1" ] && [ "${RUN_STRESS}" = "1" ] && [ "${RUN_LINEAR}" = "1" ]; then
  run_stress > "${LOG_DIR}/${CONTINUE_EXP}.stress.log" 2>&1 &
  stress_pid="$!"
  run_linear > "${LOG_DIR}/${LINEAR_EXP}.linear.log" 2>&1 &
  linear_pid="$!"
  wait "${stress_pid}"
  wait "${linear_pid}"
  cat "${LOG_DIR}/${CONTINUE_EXP}.stress.log"
  tail -n 80 "${LOG_DIR}/${LINEAR_EXP}.linear.log" || true
else
  run_stress
  run_linear
fi

echo "[$(timestamp)] done projres v1 follow-up"
