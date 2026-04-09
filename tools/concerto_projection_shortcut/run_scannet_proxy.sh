#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_GPU="${NUM_GPU:-2}"
DATASET_NAME="${DATASET_NAME:-concerto}"
MODE="${1:-all}"
OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${REPO_ROOT}/weights/concerto/concerto_base_origin.pth}"
GATE_NUM_GPU="${GATE_NUM_GPU:-1}"
PRETRAIN_NUM_GPU="${PRETRAIN_NUM_GPU:-1}"
LIN_NUM_GPU="${LIN_NUM_GPU:-1}"
FT_NUM_GPU="${FT_NUM_GPU:-1}"
GATE_CONFIG="${GATE_CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-safe}"
LIN_CONFIG="${LIN_CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-safe}"
FT_CONFIG="${FT_CONFIG:-semseg-ptv3-base-v1m1-0c-scannet-ft-proxy-safe}"
PARALLEL_SINGLE_GPU="${PARALLEL_SINGLE_GPU:-1}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1}"

CONTINUE_CONFIGS=(
  "pretrain-concerto-v1m1-0-arkit-full-continue-safe"
  "pretrain-concerto-v1m1-0-arkit-full-no-enc2d-continue-safe"
  "pretrain-concerto-v1m1-0-arkit-full-coord-mlp-continue-safe"
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

run_train() {
  local config_name="$1"
  local exp_name="$2"
  local weight_path="$3"
  local gpu_count="${4:-${NUM_GPU}}"
  local gpu_id="${5:-}"
  local checkpoint="exp/${DATASET_NAME}/${exp_name}/model/model_last.pth"
  if [ -f "${checkpoint}" ]; then
    echo "[skip] ${exp_name} already has ${checkpoint}"
    return 0
  fi
  if [ -n "${gpu_id}" ]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" bash scripts/train.sh \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g "${gpu_count}" \
      -c "${config_name}" \
      -n "${exp_name}" \
      -w "${weight_path}"
  else
    bash scripts/train.sh \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g "${gpu_count}" \
      -c "${config_name}" \
      -n "${exp_name}" \
      -w "${weight_path}"
  fi
}

run_parallel_specs() {
  local gpu_count="$1"
  shift
  local specs=("$@")
  local -a gpu_ids=()
  IFS=',' read -r -a gpu_ids <<< "${GPU_IDS_CSV}"

  if [ "${PARALLEL_SINGLE_GPU}" = "1" ] && [ "${gpu_count}" = "1" ] && [ "${#gpu_ids[@]}" -ge 2 ]; then
    local next_idx=0
    local -a slot_pid=()
    local -a slot_label=()
    local active=0
    while [ "${next_idx}" -lt "${#specs[@]}" ] || [ "${active}" -gt 0 ]; do
      local slot=""
      for slot in "${!gpu_ids[@]}"; do
        local pid="${slot_pid[$slot]:-}"
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
          if ! wait "${pid}"; then
            echo "[warn] ${slot_label[$slot]} exited non-zero"
          fi
          slot_pid[$slot]=""
          slot_label[$slot]=""
        fi
        if [ -z "${slot_pid[$slot]:-}" ]; then
          while [ "${next_idx}" -lt "${#specs[@]}" ]; do
            local config_name=""
            local exp_name=""
            local weight_path=""
            IFS='|' read -r config_name exp_name weight_path <<< "${specs[$next_idx]}"
            next_idx=$((next_idx + 1))
            local checkpoint="exp/${DATASET_NAME}/${exp_name}/model/model_last.pth"
            if [ -f "${checkpoint}" ]; then
              echo "[skip] ${exp_name} already has ${checkpoint}"
              continue
            fi
            local gpu_id="${gpu_ids[$slot]}"
            echo "[launch] gpu=${gpu_id} ${config_name} -> ${exp_name}"
            (
              run_train "${config_name}" "${exp_name}" "${weight_path}" 1 "${gpu_id}"
            ) &
            slot_pid[$slot]="$!"
            slot_label[$slot]="${exp_name}"
            break
          done
        fi
      done
      active=0
      for slot in "${!gpu_ids[@]}"; do
        if [ -n "${slot_pid[$slot]:-}" ]; then
          active=$((active + 1))
        fi
      done
      if [ "${active}" -gt 0 ]; then
        sleep 10
      fi
    done
  else
    local default_gpu=""
    if [ "${gpu_count}" = "1" ]; then
      default_gpu="${gpu_ids[0]:-}"
    fi
    local spec=""
    for spec in "${specs[@]}"; do
      local config_name=""
      local exp_name=""
      local weight_path=""
      IFS='|' read -r config_name exp_name weight_path <<< "${spec}"
      run_train "${config_name}" "${exp_name}" "${weight_path}" "${gpu_count}" "${default_gpu}"
    done
  fi
}

run_official_gate() {
  echo "[gate] official ScanNet linear probe"
  run_parallel_specs "${GATE_NUM_GPU}" "${GATE_CONFIG}|scannet-proxy-official-origin-lin|${OFFICIAL_WEIGHT}"
}

run_official_ft() {
  echo "[gate] official ScanNet fine-tune"
  run_parallel_specs "${FT_NUM_GPU}" "${FT_CONFIG}|scannet-proxy-official-origin-ft|${OFFICIAL_WEIGHT}"
}

run_continuations() {
  local -a specs=()
  local idx=""
  for idx in "${!CONTINUE_CONFIGS[@]}"; do
    echo "[continue] ${CONTINUE_CONFIGS[$idx]} -> ${CONTINUE_NAMES[$idx]}"
    specs+=("${CONTINUE_CONFIGS[$idx]}|${CONTINUE_NAMES[$idx]}|${OFFICIAL_WEIGHT}")
  done
  run_parallel_specs "${PRETRAIN_NUM_GPU}" "${specs[@]}"
}

run_downstream_from_checkpoints() {
  local task_config="$1"
  local task_suffix="$2"
  local gpu_count="$3"
  local -a specs=()
  local idx=""
  for idx in "${!CONTINUE_NAMES[@]}"; do
    local checkpoint="exp/${DATASET_NAME}/${CONTINUE_NAMES[$idx]}/model/model_last.pth"
    local exp_name="scannet-proxy-${PROBE_LABELS[$idx]}-${task_suffix}"
    echo "[downstream] ${task_config} <- ${checkpoint}"
    specs+=("${task_config}|${exp_name}|${checkpoint}")
  done
  run_parallel_specs "${gpu_count}" "${specs[@]}"
}

write_summary() {
  local csv_glob="$1"
  local csv_path="$2"
  local md_path="$3"
  local files=()
  while IFS= read -r file; do
    files+=("${file}")
  done < <(find "exp/${DATASET_NAME}" -path "${csv_glob}" | sort)
  if [ "${#files[@]}" -eq 0 ]; then
    return 0
  fi
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_semseg_logs.py "${files[@]}" > "${csv_path}"
  "${PYTHON_BIN}" - "${csv_path}" "${md_path}" <<'PY'
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
}

case "${MODE}" in
  gate)
    run_official_gate
    ;;
  gate-ft)
    run_official_ft
    ;;
  pretrain)
    run_continuations
    ;;
  lin)
    run_downstream_from_checkpoints "${LIN_CONFIG}" "lin" "${LIN_NUM_GPU}"
    ;;
  ft)
    run_downstream_from_checkpoints "${FT_CONFIG}" "ft" "${FT_NUM_GPU}"
    ;;
  all)
    run_official_gate
    run_continuations
    run_downstream_from_checkpoints "${LIN_CONFIG}" "lin" "${LIN_NUM_GPU}"
    ;;
  *)
    echo "usage: $0 [gate|gate-ft|pretrain|lin|ft|all]" >&2
    exit 2
    ;;
esac

write_summary \
  "*/scannet-proxy-*-lin/train.log" \
  "tools/concerto_projection_shortcut/results_scannet_proxy_lin.csv" \
  "tools/concerto_projection_shortcut/results_scannet_proxy_lin.md"
write_summary \
  "*/scannet-proxy-*-ft/train.log" \
  "tools/concerto_projection_shortcut/results_scannet_proxy_ft.csv" \
  "tools/concerto_projection_shortcut/results_scannet_proxy_ft.md"

echo "[done] downstream proxy summaries updated"
