#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

DATASET_NAME="${DATASET_NAME:-concerto}"
OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${WEIGHT_DIR}/concerto_base_origin.pth}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

BASE_CONFIG="${BASE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-continue}"
SMOKE_CONFIG="${SMOKE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-projres-v1a-smoke}"
CONTINUE_CONFIG="${CONTINUE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-projres-v1a-continue}"
LINEAR_CONFIG="${LINEAR_CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly}"
FT_CONFIG="${FT_CONFIG:-semseg-ptv3-base-v1m1-0c-scannet-ft-proxy-safe}"

PRIOR_ROOT="${PRIOR_ROOT:-${EXP_MIRROR_ROOT}/priors}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries}"
LOG_DIR="${LOG_DIR:-${EXP_MIRROR_ROOT}/logs}"
ALPHAS_CSV="${ALPHAS_CSV:-0.05,0.10}"
EXP_TAG="${EXP_TAG:-}"
MULTINODE_TRAIN="${MULTINODE_TRAIN:-0}"
MULTINODE_TRAIN_LAUNCHER="${MULTINODE_TRAIN_LAUNCHER:-${REPO_ROOT}/tools/concerto_projection_shortcut/run_pointcept_train_multinode_pbsdsh.sh}"
SMOKE_GPU_IDS_CSV="${SMOKE_GPU_IDS_CSV:-${GPU_IDS_CSV}}"
SMOKE_ALL_GPUS="${SMOKE_ALL_GPUS:-0}"
STOP_AFTER_SMOKE="${STOP_AFTER_SMOKE:-0}"
if [ -z "${SMOKE_PARALLEL+x}" ]; then
  if [ "${MULTINODE_TRAIN}" = "1" ]; then
    SMOKE_PARALLEL=0
  else
    SMOKE_PARALLEL=1
  fi
fi

MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-4096}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-512}"
MAX_ROWS_PER_BATCH="${MAX_ROWS_PER_BATCH:-512}"
PRIOR_EPOCHS="${PRIOR_EPOCHS:-20}"
PRIOR_BATCH_SIZE="${PRIOR_BATCH_SIZE:-8192}"
MAX_STRESS_BATCHES="${MAX_STRESS_BATCHES:-20}"
LINEAR_GO_MARGIN="${LINEAR_GO_MARGIN:-0.01}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

device_count() {
  awk -F',' '{print NF}' <<< "$1"
}

checkpoint_path() {
  local exp_name="$1"
  printf '%s/exp/%s/%s/model/model_last.pth' "${REPO_ROOT}" "${DATASET_NAME}" "${exp_name}"
}

ensure_mirror_exp() {
  local exp_name="$1"
  local link_path="${REPO_ROOT}/exp/${DATASET_NAME}/${exp_name}"
  local target_path="${EXP_MIRROR_ROOT}/exp/${exp_name}"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] ensure symlink ${link_path} -> ${target_path}"
    return 0
  fi
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

run_cmd() {
  echo "[$(timestamp)] $*"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  "$@"
}

run_train() {
  local config_name="$1"
  local exp_name="$2"
  local weight_path="$3"
  local devices="$4"
  local alpha="${5:-0.05}"
  local local_only="${6:-0}"
  local gpu_count
  gpu_count="$(device_count "${devices}")"
  ensure_mirror_exp "${exp_name}"
  if [ -f "$(checkpoint_path "${exp_name}")" ]; then
    echo "[$(timestamp)] skip: ${exp_name} already has model_last.pth"
    return 0
  fi
  echo "[$(timestamp)] train: gpus=${devices} alpha=${alpha} config=${config_name} exp=${exp_name}"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  if [ "${MULTINODE_TRAIN}" = "1" ] && [ "${local_only}" != "1" ]; then
    DATASET_NAME="${DATASET_NAME}" \
      CONFIG_NAME="${config_name}" \
      EXP_NAME="${exp_name}" \
      WEIGHT_PATH="${weight_path}" \
      TRAIN_GPU_IDS_CSV="${devices}" \
      COORD_PRIOR_PATH="${COORD_PRIOR_PATH}" \
      COORD_PROJECTION_ALPHA="${alpha}" \
      COORD_PROJECTION_BETA="${COORD_PROJECTION_BETA:-1.0}" \
      LOG_DIR="${LOG_DIR}" \
      bash "${MULTINODE_TRAIN_LAUNCHER}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${devices}" \
    COORD_PRIOR_PATH="${COORD_PRIOR_PATH}" \
    COORD_PROJECTION_ALPHA="${alpha}" \
    COORD_PROJECTION_BETA="${COORD_PROJECTION_BETA:-1.0}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    bash "${REPO_ROOT}/scripts/train.sh" \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g "${gpu_count}" \
      -c "${config_name}" \
      -n "${exp_name}" \
      -w "${weight_path}"
}

alpha_tag() {
  local alpha="$1"
  printf '%s' "${alpha}" | tr -d '.'
}

arkit_exp_name() {
  local tag="$1"
  local phase="$2"
  printf 'arkit-full-projres-v1a-alpha%s%s-%s\n' "${tag}" "${EXP_TAG}" "${phase}"
}

scannet_exp_name() {
  local tag="$1"
  local phase="$2"
  printf 'scannet-proxy-projres-v1a-alpha%s%s-%s\n' "${tag}" "${EXP_TAG}" "${phase}"
}

first_gpu() {
  awk -F',' '{print $1}' <<< "${GPU_IDS_CSV}"
}

second_gpu() {
  awk -F',' '{print ($2 == "" ? $1 : $2)}' <<< "${GPU_IDS_CSV}"
}

smoke_devices_for_idx() {
  local idx="$1"
  if [ "${MULTINODE_TRAIN}" = "1" ] || [ "${SMOKE_ALL_GPUS}" = "1" ]; then
    printf '%s\n' "${SMOKE_GPU_IDS_CSV}"
  else
    printf '%s\n' "${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  fi
}

run_smoke_one() {
  local idx="$1"
  local alpha="$2"
  local exp_name="$3"
  local smoke_json="$4"
  local devices
  devices="$(smoke_devices_for_idx "${idx}")"
  run_train "${SMOKE_CONFIG}" "${exp_name}" "${OFFICIAL_WEIGHT}" "${devices}" "${alpha}"
  write_smoke_summary "${exp_name}" "${alpha}" "${smoke_json}"
}

write_smoke_summary() {
  local exp_name="$1"
  local alpha="$2"
  local out_json="$3"
  local log_path="${REPO_ROOT}/exp/${DATASET_NAME}/${exp_name}/train.log"
  "${PYTHON_BIN}" - "${log_path}" "${alpha}" "${out_json}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
alpha = float(sys.argv[2])
out_json = Path(sys.argv[3])
keys = [
    "loss",
    "enc2d_loss",
    "coord_pred_energy",
    "coord_target_energy",
    "coord_removed_energy",
    "coord_residual_norm",
    "coord_alignment_loss",
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

summary = {
    "alpha": alpha,
    "beta": float(__import__("os").environ.get("COORD_PROJECTION_BETA", "1.0")),
    "log": str(log_path),
    "pass": False,
    "reason": "missing_metrics",
    "first": {},
    "last": {},
}
for key, seq in values.items():
    if seq:
        summary["first"][key] = seq[0]
        summary["last"][key] = seq[-1]

if finite(values["loss"]) and finite(values["enc2d_loss"]) and finite(values["coord_pred_energy"]) and finite(values["coord_residual_norm"]):
    pred_first = values["coord_pred_energy"][0]
    pred_last = values["coord_pred_energy"][-1]
    residual_last = values["coord_residual_norm"][-1]
    enc_last = values["enc2d_loss"][-1]
    if residual_last < 0.70:
        summary["reason"] = "residual_norm_below_0.70"
    elif pred_last > pred_first + 1e-4:
        summary["reason"] = "coord_pred_energy_not_lower"
    elif enc_last <= 0.01 or enc_last >= 50:
        summary["reason"] = "enc2d_loss_collapse_or_explode"
    elif values.get("coord_projection_loss_check") and max(values["coord_projection_loss_check"]) > 1e-3:
        summary["reason"] = "projection_loss_check_failed"
    else:
        summary["pass"] = True
        summary["reason"] = "pass"

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
}

select_smoke() {
  local selected_json="$1"
  shift
  "${PYTHON_BIN}" - "${selected_json}" "$@" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = [json.loads(Path(path).read_text()) for path in sys.argv[2:]]
passed = [row for row in rows if row.get("pass")]
if not passed:
    payload = {"pass": False, "reason": "no_smoke_passed", "candidates": rows}
else:
    passed.sort(key=lambda row: row["last"].get("enc2d_loss", float("inf")))
    payload = {"pass": True, "selected": passed[0], "candidates": rows}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
if not payload["pass"]:
    raise SystemExit(2)
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

if [ "${SKIP_VENV_ACTIVATE:-0}" != "1" ]; then
  ensure_venv_active
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
fi
mkdir -p "${PRIOR_ROOT}" "${SUMMARY_ROOT}" "${LOG_DIR}"

if [ ! -f "${OFFICIAL_WEIGHT}" ]; then
  echo "[$(timestamp)] error: missing official weight: ${OFFICIAL_WEIGHT}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a ALPHAS <<< "${ALPHAS_CSV}"
if [ "${#ALPHAS[@]}" -lt 1 ]; then
  echo "[$(timestamp)] error: ALPHAS_CSV is empty" >&2
  exit 2
fi

echo "[$(timestamp)] start projres v1 chain"
echo "exp_mirror_root=${EXP_MIRROR_ROOT}"
echo "gpu_ids=${GPU_IDS_CSV}"
echo "exp_tag=${EXP_TAG}"
echo "multinode_train=${MULTINODE_TRAIN}"
echo "smoke_parallel=${SMOKE_PARALLEL}"

if [ "${RUN_PREFLIGHT}" = "1" ]; then
  run_cmd env CUDA_VISIBLE_DEVICES="$(first_gpu)" "${PYTHON_BIN}" \
    tools/concerto_projection_shortcut/preflight.py \
    --check-data --check-batch --check-forward --config "${BASE_CONFIG}" \
    --data-root "${ARKIT_FULL_META_ROOT}"
fi

SELECTED_PRIOR_JSON="${PRIOR_ROOT}/selected_prior.json"
if [ ! -f "${SELECTED_PRIOR_JSON}" ]; then
  run_cmd env CUDA_VISIBLE_DEVICES="$(first_gpu)" "${PYTHON_BIN}" \
    tools/concerto_projection_shortcut/fit_coord_prior.py \
    --config "${BASE_CONFIG}" \
    --weight "${OFFICIAL_WEIGHT}" \
    --data-root "${ARKIT_FULL_META_ROOT}" \
    --output-root "${PRIOR_ROOT}" \
    --max-train-batches "${MAX_TRAIN_BATCHES}" \
    --max-val-batches "${MAX_VAL_BATCHES}" \
    --max-rows-per-batch "${MAX_ROWS_PER_BATCH}" \
    --prior-epochs "${PRIOR_EPOCHS}" \
    --prior-batch-size "${PRIOR_BATCH_SIZE}"
else
  echo "[$(timestamp)] skip prior fit: ${SELECTED_PRIOR_JSON}"
fi

if [ "${DRY_RUN}" = "1" ]; then
  COORD_PRIOR_PATH="${PRIOR_ROOT}/linear/model_last.pth"
else
  COORD_PRIOR_PATH="$("${PYTHON_BIN}" - "${SELECTED_PRIOR_JSON}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["selected_path"])
PY
)"
fi
export COORD_PRIOR_PATH
echo "coord_prior_path=${COORD_PRIOR_PATH}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[dry-run] would preflight fix config=${SMOKE_CONFIG} alpha=${ALPHAS[0]}"
  for idx in "${!ALPHAS[@]}"; do
    alpha="${ALPHAS[$idx]}"
    tag="$(alpha_tag "${alpha}")"
    exp_name="$(arkit_exp_name "${tag}" smoke)"
    run_train "${SMOKE_CONFIG}" "${exp_name}" "${OFFICIAL_WEIGHT}" "$(smoke_devices_for_idx "${idx}")" "${alpha}"
  done
  first_alpha="${ALPHAS[0]}"
  first_tag="$(alpha_tag "${first_alpha}")"
  continue_exp="$(arkit_exp_name "${first_tag}" continue)"
  run_train "${CONTINUE_CONFIG}" "${continue_exp}" "${OFFICIAL_WEIGHT}" "${GPU_IDS_CSV}" "${first_alpha}"
  echo "[dry-run] would run stress for ${continue_exp}"
  run_train "${LINEAR_CONFIG}" "$(scannet_exp_name "${first_tag}" lin)" "${EXP_MIRROR_ROOT}/exp/${continue_exp}/model/model_last.pth" "$(first_gpu)" "${first_alpha}" 1
  echo "[dry-run] would run fine-tune only if linear gate is strong_go"
  echo "[$(timestamp)] dry-run complete"
  exit 0
fi

if [ "${RUN_PREFLIGHT}" = "1" ]; then
  run_cmd env CUDA_VISIBLE_DEVICES="$(first_gpu)" \
    COORD_PRIOR_PATH="${COORD_PRIOR_PATH}" \
    COORD_PROJECTION_ALPHA="${ALPHAS[0]}" \
    COORD_PROJECTION_BETA="${COORD_PROJECTION_BETA:-1.0}" \
    "${PYTHON_BIN}" tools/concerto_projection_shortcut/preflight.py \
    --check-data --check-batch --check-forward --config "${SMOKE_CONFIG}" \
    --data-root "${ARKIT_FULL_META_ROOT}"
fi

SMOKE_JSONS=()
PIDS=()
for idx in "${!ALPHAS[@]}"; do
  alpha="${ALPHAS[$idx]}"
  tag="$(alpha_tag "${alpha}")"
  exp_name="$(arkit_exp_name "${tag}" smoke)"
  smoke_json="${SUMMARY_ROOT}/${exp_name}.json"
  SMOKE_JSONS+=("${smoke_json}")
  if [ -f "${smoke_json}" ]; then
    echo "[$(timestamp)] skip smoke summary: ${smoke_json}"
    continue
  fi
  if [ "${SMOKE_PARALLEL}" = "1" ]; then
    (
      run_smoke_one "${idx}" "${alpha}" "${exp_name}" "${smoke_json}"
    ) > "${LOG_DIR}/${exp_name}.launch.log" 2>&1 &
    PIDS+=("$!")
  else
    run_smoke_one "${idx}" "${alpha}" "${exp_name}" "${smoke_json}" \
      > "${LOG_DIR}/${exp_name}.launch.log" 2>&1
  fi
done
if [ "${#PIDS[@]}" -gt 0 ]; then
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done
fi

SELECTED_SMOKE_JSON="${SUMMARY_ROOT}/selected_smoke.json"
select_smoke "${SELECTED_SMOKE_JSON}" "${SMOKE_JSONS[@]}"
if [ "${STOP_AFTER_SMOKE}" = "1" ]; then
  echo "[$(timestamp)] stop after smoke: ${SELECTED_SMOKE_JSON}"
  exit 0
fi
SELECTED_ALPHA="$("${PYTHON_BIN}" - "${SELECTED_SMOKE_JSON}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["selected"]["alpha"])
PY
)"
SELECTED_TAG="$(alpha_tag "${SELECTED_ALPHA}")"
echo "selected_alpha=${SELECTED_ALPHA}"

CONTINUE_EXP="$(arkit_exp_name "${SELECTED_TAG}" continue)"
run_train "${CONTINUE_CONFIG}" "${CONTINUE_EXP}" "${OFFICIAL_WEIGHT}" "${GPU_IDS_CSV}" "${SELECTED_ALPHA}"
CONTINUE_CKPT="$(checkpoint_path "${CONTINUE_EXP}")"

STRESS_CSV="${SUMMARY_ROOT}/${CONTINUE_EXP}_stress.csv"
if [ ! -f "${STRESS_CSV}" ]; then
  echo "[$(timestamp)] stress: ${CONTINUE_EXP} -> ${STRESS_CSV}"
  env CUDA_VISIBLE_DEVICES="$(first_gpu)" \
      COORD_PRIOR_PATH="${COORD_PRIOR_PATH}" \
      COORD_PROJECTION_ALPHA="${SELECTED_ALPHA}" \
      COORD_PROJECTION_BETA="${COORD_PROJECTION_BETA:-1.0}" \
      "${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_enc2d_stress.py \
      --config "${CONTINUE_CONFIG}" \
      --weight "${CONTINUE_CKPT}" \
      --data-root "${ARKIT_FULL_META_ROOT}" \
      --max-batches "${MAX_STRESS_BATCHES}" \
      > "${STRESS_CSV}"
else
  echo "[$(timestamp)] skip stress: ${STRESS_CSV}"
fi

LINEAR_EXP="$(scannet_exp_name "${SELECTED_TAG}" lin)"
run_train "${LINEAR_CONFIG}" "${LINEAR_EXP}" "${CONTINUE_CKPT}" "$(first_gpu)" "${SELECTED_ALPHA}" 1
LINEAR_GATE_JSON="${SUMMARY_ROOT}/${LINEAR_EXP}_gate.json"
write_linear_gate "${LINEAR_EXP}" "${LINEAR_GATE_JSON}"

if "${PYTHON_BIN}" - "${LINEAR_GATE_JSON}" <<'PY'
import json
import sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text())["strong_go"] else 1)
PY
then
  FT_EXP="$(scannet_exp_name "${SELECTED_TAG}" ft)"
  run_train "${FT_CONFIG}" "${FT_EXP}" "${CONTINUE_CKPT}" "$(first_gpu)" "${SELECTED_ALPHA}" 1
else
  echo "[$(timestamp)] stop before fine-tune: linear gate is not strong_go"
fi

echo "[$(timestamp)] done projres v1 chain"
