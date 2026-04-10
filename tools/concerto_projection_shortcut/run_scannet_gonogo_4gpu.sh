#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

MODE="${1:-gonogo}"
DATASET_NAME="${DATASET_NAME:-concerto}"
OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${REPO_ROOT}/weights/concerto/concerto_base_origin.pth}"

# Original, non-safe config set from the pre-safe ScanNet proxy script.
LINEAR_CONFIG="${LINEAR_CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy}"

# Priority layout for the decisive pair:
# stage 1: original + coord-mlp continuations
# stage 2: their ScanNet linears + no-enc2d continuation
# stage 3: no-enc2d ScanNet linear
CONTINUE_PAIR_A="${CONTINUE_PAIR_A:-0,1}"
CONTINUE_PAIR_B="${CONTINUE_PAIR_B:-2,3}"
LINEAR_GPU_ORIG="${LINEAR_GPU_ORIG:-0}"
LINEAR_GPU_COORD="${LINEAR_GPU_COORD:-1}"
LINEAR_GPU_NOENC2D="${LINEAR_GPU_NOENC2D:-2}"

SUMMARY_CSV="${SUMMARY_CSV:-tools/concerto_projection_shortcut/results_scannet_proxy_lin.csv}"
SUMMARY_MD="${SUMMARY_MD:-tools/concerto_projection_shortcut/results_scannet_proxy_lin.md}"
LOG_DIR="${LOG_DIR:-tools/concerto_projection_shortcut/logs}"

CONTINUE_CONFIGS=(
  "pretrain-concerto-v1m1-0-arkit-full-continue-a1004"
  "pretrain-concerto-v1m1-0-arkit-full-no-enc2d-continue-a1004"
  "pretrain-concerto-v1m1-0-arkit-full-coord-mlp-continue-a1004"
)
CONTINUE_NAMES=(
  "arkit-full-continue-concerto"
  "arkit-full-continue-no-enc2d"
  "arkit-full-continue-coord-mlp"
)
PROBE_LABELS=(
  "concerto-continue"
  "no-enc2d-continue"
  "coord-mlp-continue"
)

ORIG_IDX=0
NOENC2D_IDX=1
COORD_IDX=2

ensure_conda_active
mkdir -p "${LOG_DIR}"
LAUNCH_PID=""

checkpoint_path() {
  local exp_name="$1"
  printf '%s/exp/%s/%s/model/model_last.pth' "${REPO_ROOT}" "${DATASET_NAME}" "${exp_name}"
}

stash_incomplete_exp() {
  local exp_name="$1"
  local exp_dir="${REPO_ROOT}/exp/${DATASET_NAME}/${exp_name}"
  local checkpoint
  checkpoint="$(checkpoint_path "${exp_name}")"
  if [ -d "${exp_dir}" ] && [ ! -f "${checkpoint}" ]; then
    local stamp stale_dir
    stamp="$(date +%Y%m%d-%H%M%S)"
    stale_dir="${exp_dir}-stale-${stamp}"
    mv "${exp_dir}" "${stale_dir}"
    echo "[stash] ${exp_dir} -> ${stale_dir}"
  fi
}

linear_exp_name() {
  local idx="$1"
  printf 'scannet-proxy-%s-lin' "${PROBE_LABELS[$idx]}"
}

device_count() {
  awk -F',' '{print NF}' <<< "$1"
}

csv_contains() {
  local csv="$1"
  local value="$2"
  local item=""
  IFS=',' read -r -a _items <<< "${csv}"
  for item in "${_items[@]}"; do
    if [ "${item}" = "${value}" ]; then
      return 0
    fi
  done
  return 1
}

validate_layout() {
  if [ "$(device_count "${CONTINUE_PAIR_A}")" -ne 2 ] || [ "$(device_count "${CONTINUE_PAIR_B}")" -ne 2 ]; then
    echo "[error] continuation pairs must each contain exactly 2 GPUs" >&2
    exit 2
  fi
  if csv_contains "${CONTINUE_PAIR_B}" "${LINEAR_GPU_ORIG}" || csv_contains "${CONTINUE_PAIR_B}" "${LINEAR_GPU_COORD}"; then
    echo "[error] LINEAR_GPU_ORIG/LINEAR_GPU_COORD must not overlap CONTINUE_PAIR_B for gonogo mode" >&2
    exit 2
  fi
}

run_train() {
  local config_name="$1"
  local exp_name="$2"
  local weight_path="$3"
  local devices="$4"
  local checkpoint
  checkpoint="$(checkpoint_path "${exp_name}")"
  if [ -f "${checkpoint}" ]; then
    echo "[skip] ${exp_name} already has ${checkpoint}"
    return 0
  fi
  stash_incomplete_exp "${exp_name}"

  local gpu_count
  gpu_count="$(device_count "${devices}")"
  echo "[run] gpus=${devices} ${config_name} -> ${exp_name}"
  CUDA_VISIBLE_DEVICES="${devices}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    bash "${REPO_ROOT}/scripts/train.sh" \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g "${gpu_count}" \
      -c "${config_name}" \
      -n "${exp_name}" \
      -w "${weight_path}"
}

launch_train_bg() {
  local config_name="$1"
  local exp_name="$2"
  local weight_path="$3"
  local devices="$4"
  local launch_log="${LOG_DIR}/${exp_name}.launch.log"
  local checkpoint
  LAUNCH_PID=""
  checkpoint="$(checkpoint_path "${exp_name}")"
  if [ -f "${checkpoint}" ]; then
    echo "[skip] ${exp_name} already has ${checkpoint}" >&2
    return 0
  fi

  echo "[launch] gpus=${devices} ${config_name} -> ${exp_name} log=${launch_log}"
  (
    run_train "${config_name}" "${exp_name}" "${weight_path}" "${devices}"
  ) >"${launch_log}" 2>&1 &
  LAUNCH_PID="$!"
}

wait_optional_pid() {
  local pid="$1"
  local label="$2"
  if [ -z "${pid}" ]; then
    return 0
  fi
  if ! wait "${pid}"; then
    echo "[error] ${label} failed" >&2
    exit 1
  fi
}

write_linear_summary() {
  local files=()
  while IFS= read -r file; do
    files+=("${file}")
  done < <(find "exp/${DATASET_NAME}" -path "*/scannet-proxy-*-lin/train.log" | sort)

  if [ "${#files[@]}" -eq 0 ]; then
    return 0
  fi

  "${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_semseg_logs.py "${files[@]}" > "${SUMMARY_CSV}"
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))
lines = [
    f"# {md_path.stem}",
    "",
    "| experiment | val mIoU | val mAcc | val allAcc | best metric | best value | eval count |",
    "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
]
for row in rows:
    exp_name = Path(row["log"]).parent.name
    lines.append(
        f"| {exp_name} | {row['val_miou_last']} | {row['val_macc_last']} | "
        f"{row['val_allacc_last']} | {row['best_metric_name']} | {row['best_metric_value']} | "
        f"{row['val_eval_count']} |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  echo "[done] updated ${SUMMARY_CSV} and ${SUMMARY_MD}"
}

run_priority_continuations() {
  local pid_orig=""
  local pid_coord=""
  launch_train_bg \
    "${CONTINUE_CONFIGS[$ORIG_IDX]}" \
    "${CONTINUE_NAMES[$ORIG_IDX]}" \
    "${OFFICIAL_WEIGHT}" \
    "${CONTINUE_PAIR_A}"
  pid_orig="${LAUNCH_PID}"
  launch_train_bg \
    "${CONTINUE_CONFIGS[$COORD_IDX]}" \
    "${CONTINUE_NAMES[$COORD_IDX]}" \
    "${OFFICIAL_WEIGHT}" \
    "${CONTINUE_PAIR_B}"
  pid_coord="${LAUNCH_PID}"
  wait_optional_pid "${pid_orig}" "${CONTINUE_NAMES[$ORIG_IDX]}"
  wait_optional_pid "${pid_coord}" "${CONTINUE_NAMES[$COORD_IDX]}"
}

run_remaining_continuation() {
  local devices="${1:-${CONTINUE_PAIR_A}}"
  local pid_noenc=""
  launch_train_bg \
    "${CONTINUE_CONFIGS[$NOENC2D_IDX]}" \
    "${CONTINUE_NAMES[$NOENC2D_IDX]}" \
    "${OFFICIAL_WEIGHT}" \
    "${devices}"
  pid_noenc="${LAUNCH_PID}"
  wait_optional_pid "${pid_noenc}" "${CONTINUE_NAMES[$NOENC2D_IDX]}"
}

run_linear_trio() {
  local orig_ckpt coord_ckpt noenc_ckpt
  orig_ckpt="$(checkpoint_path "${CONTINUE_NAMES[$ORIG_IDX]}")"
  coord_ckpt="$(checkpoint_path "${CONTINUE_NAMES[$COORD_IDX]}")"
  noenc_ckpt="$(checkpoint_path "${CONTINUE_NAMES[$NOENC2D_IDX]}")"
  for ckpt in "${orig_ckpt}" "${coord_ckpt}" "${noenc_ckpt}"; do
    if [ ! -f "${ckpt}" ]; then
      echo "[error] missing continuation checkpoint: ${ckpt}" >&2
      exit 1
    fi
  done

  local pid_orig pid_coord pid_noenc
  launch_train_bg "${LINEAR_CONFIG}" "$(linear_exp_name "${ORIG_IDX}")" "${orig_ckpt}" "${LINEAR_GPU_ORIG}"
  pid_orig="${LAUNCH_PID}"
  launch_train_bg "${LINEAR_CONFIG}" "$(linear_exp_name "${COORD_IDX}")" "${coord_ckpt}" "${LINEAR_GPU_COORD}"
  pid_coord="${LAUNCH_PID}"
  launch_train_bg "${LINEAR_CONFIG}" "$(linear_exp_name "${NOENC2D_IDX}")" "${noenc_ckpt}" "${LINEAR_GPU_NOENC2D}"
  pid_noenc="${LAUNCH_PID}"
  wait_optional_pid "${pid_orig}" "$(linear_exp_name "${ORIG_IDX}")"
  wait_optional_pid "${pid_coord}" "$(linear_exp_name "${COORD_IDX}")"
  wait_optional_pid "${pid_noenc}" "$(linear_exp_name "${NOENC2D_IDX}")"
  write_linear_summary
}

run_gonogo_pipeline() {
  local orig_ckpt coord_ckpt noenc_ckpt
  local pid_lin_orig="" pid_lin_coord="" pid_cont_noenc="" pid_lin_noenc=""

  run_priority_continuations

  orig_ckpt="$(checkpoint_path "${CONTINUE_NAMES[$ORIG_IDX]}")"
  coord_ckpt="$(checkpoint_path "${CONTINUE_NAMES[$COORD_IDX]}")"
  if [ ! -f "${orig_ckpt}" ] || [ ! -f "${coord_ckpt}" ]; then
    echo "[error] decisive continuation checkpoints are missing" >&2
    exit 1
  fi

  launch_train_bg "${LINEAR_CONFIG}" "$(linear_exp_name "${ORIG_IDX}")" "${orig_ckpt}" "${LINEAR_GPU_ORIG}"
  pid_lin_orig="${LAUNCH_PID}"
  launch_train_bg "${LINEAR_CONFIG}" "$(linear_exp_name "${COORD_IDX}")" "${coord_ckpt}" "${LINEAR_GPU_COORD}"
  pid_lin_coord="${LAUNCH_PID}"
  launch_train_bg \
    "${CONTINUE_CONFIGS[$NOENC2D_IDX]}" \
    "${CONTINUE_NAMES[$NOENC2D_IDX]}" \
    "${OFFICIAL_WEIGHT}" \
    "${CONTINUE_PAIR_B}"
  pid_cont_noenc="${LAUNCH_PID}"

  wait_optional_pid "${pid_cont_noenc}" "${CONTINUE_NAMES[$NOENC2D_IDX]}"

  noenc_ckpt="$(checkpoint_path "${CONTINUE_NAMES[$NOENC2D_IDX]}")"
  if [ ! -f "${noenc_ckpt}" ]; then
    echo "[error] missing no-enc2d continuation checkpoint: ${noenc_ckpt}" >&2
    exit 1
  fi
  launch_train_bg "${LINEAR_CONFIG}" "$(linear_exp_name "${NOENC2D_IDX}")" "${noenc_ckpt}" "${LINEAR_GPU_NOENC2D}"
  pid_lin_noenc="${LAUNCH_PID}"

  wait_optional_pid "${pid_lin_orig}" "$(linear_exp_name "${ORIG_IDX}")"
  wait_optional_pid "${pid_lin_coord}" "$(linear_exp_name "${COORD_IDX}")"
  wait_optional_pid "${pid_lin_noenc}" "$(linear_exp_name "${NOENC2D_IDX}")"
  write_linear_summary
}

print_status() {
  cat <<EOF
mode=${MODE}
official_weight=${OFFICIAL_WEIGHT}
continue_pair_a=${CONTINUE_PAIR_A}
continue_pair_b=${CONTINUE_PAIR_B}
linear_gpu_orig=${LINEAR_GPU_ORIG}
linear_gpu_coord=${LINEAR_GPU_COORD}
linear_gpu_noenc2d=${LINEAR_GPU_NOENC2D}
linear_config=${LINEAR_CONFIG}

priority_order:
  1. ${CONTINUE_NAMES[$ORIG_IDX]}
  2. ${CONTINUE_NAMES[$COORD_IDX]}
  3. ${CONTINUE_NAMES[$NOENC2D_IDX]}
EOF
}

if [ ! -f "${OFFICIAL_WEIGHT}" ]; then
  echo "[error] missing official weight: ${OFFICIAL_WEIGHT}" >&2
  exit 1
fi

validate_layout

case "${MODE}" in
  status)
    print_status
    ;;
  pretrain)
    run_priority_continuations
    run_remaining_continuation "${CONTINUE_PAIR_A}"
    ;;
  lin)
    run_linear_trio
    ;;
  gonogo|all)
    run_gonogo_pipeline
    ;;
  *)
    echo "usage: $0 [status|pretrain|lin|gonogo|all]" >&2
    exit 2
    ;;
esac
