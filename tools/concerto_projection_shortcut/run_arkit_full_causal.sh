#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_GPU="${NUM_GPU:-2}"
DATASET_NAME="${DATASET_NAME:-concerto}"
EXP_PREFIX="${EXP_PREFIX:-arkit-full-causal}"
INCLUDE_FIX="${INCLUDE_FIX:-0}"
RUN_STRESS="${RUN_STRESS:-0}"
MAX_STRESS_BATCHES="${MAX_STRESS_BATCHES:-20}"
PARALLEL_SINGLE_GPU="${PARALLEL_SINGLE_GPU:-1}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1}"
START_AT_INDEX="${START_AT_INDEX:-0}"
ARKIT_FULL_SOURCE_ROOT="${ARKIT_FULL_SOURCE_ROOT:-/home/cvrt/datasets/arkitscenes/arkitscenes}"
ARKIT_FULL_META_ROOT="${ARKIT_FULL_META_ROOT:-/home/cvrt/datasets/arkitscenes/arkitscenes_absmeta}"

"${PYTHON_BIN}" tools/concerto_projection_shortcut/prepare_arkit_full_splits.py \
  --source-root "${ARKIT_FULL_SOURCE_ROOT}" \
  --output-root "${ARKIT_FULL_META_ROOT}"

declare -a CONFIGS=(
  "pretrain-concerto-v1m1-0-probe-enc2d-full-baseline"
  "pretrain-concerto-v1m1-0-probe-enc2d-full-coord-mlp"
  "pretrain-concerto-v1m1-0-probe-enc2d-full-global-target-permutation"
  "pretrain-concerto-v1m1-0-probe-enc2d-full-cross-image-target-swap"
  "pretrain-concerto-v1m1-0-probe-enc2d-full-cross-scene-target-swap"
)

declare -a NAMES=(
  "${EXP_PREFIX}-baseline"
  "${EXP_PREFIX}-coord-mlp"
  "${EXP_PREFIX}-global-target-permutation"
  "${EXP_PREFIX}-cross-image-target-swap"
  "${EXP_PREFIX}-cross-scene-target-swap"
)

if [ "${INCLUDE_FIX}" = "1" ]; then
  CONFIGS+=("pretrain-concerto-v1m1-0-probe-enc2d-full-coord-residual-target")
  NAMES+=("${EXP_PREFIX}-coord-residual-target")
fi

RESULT_CSV="tools/concerto_projection_shortcut/results_arkit_full_causal.csv"
RESULT_MD="tools/concerto_projection_shortcut/results_arkit_full_causal.md"
DONE_STAMP="tools/concerto_projection_shortcut/arkit_full_causal.done"

rm -f "${DONE_STAMP}"
write_result_summary() {
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_logs.py \
    "exp/${DATASET_NAME}/${EXP_PREFIX}-*/train.log" > "${RESULT_CSV}"

  "${PYTHON_BIN}" - "${RESULT_CSV}" "${RESULT_MD}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))
lines = [
    "# ARKit Full Causal Branch",
    "",
    "| experiment | enc2d first | enc2d last | enc2d min | count |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for row in rows:
    exp_name = Path(row["log"]).parent.name
    lines.append(
        f"| {exp_name} | {row['enc2d_loss_first']} | {row['enc2d_loss_last']} | "
        f"{row['enc2d_loss_min']} | {row['enc2d_loss_count']} |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

run_one() {
  local config_name="$1"
  local exp_name="$2"
  local gpu_count="$3"
  local gpu_id="${4:-}"
  local checkpoint="exp/${DATASET_NAME}/${exp_name}/model/model_last.pth"
  if [ -f "${checkpoint}" ]; then
    echo "[skip] ${exp_name} already has ${checkpoint}"
    return 0
  fi
  echo "[run] ${config_name} -> ${exp_name}"
  if [ -n "${gpu_id}" ]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" bash scripts/train.sh \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g "${gpu_count}" \
      -c "${config_name}" \
      -n "${exp_name}"
  else
    bash scripts/train.sh \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g "${gpu_count}" \
      -c "${config_name}" \
      -n "${exp_name}"
  fi
}

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
if [ "${PARALLEL_SINGLE_GPU}" = "1" ] && [ "${#GPU_IDS[@]}" -ge 2 ]; then
  idx="${START_AT_INDEX}"
  while [ "${idx}" -lt "${#CONFIGS[@]}" ]; do
    pids=()
    jobs_started=0
    for offset in 0 1; do
      cfg_idx=$((idx + offset))
      if [ "${cfg_idx}" -ge "${#CONFIGS[@]}" ]; then
        continue
      fi
      checkpoint="exp/${DATASET_NAME}/${NAMES[$cfg_idx]}/model/model_last.pth"
      if [ -f "${checkpoint}" ]; then
        echo "[skip] ${NAMES[$cfg_idx]} already has ${checkpoint}"
        continue
      fi
      gpu_id="${GPU_IDS[$offset]}"
      echo "[launch] gpu=${gpu_id} ${CONFIGS[$cfg_idx]} -> ${NAMES[$cfg_idx]}"
      (
        run_one "${CONFIGS[$cfg_idx]}" "${NAMES[$cfg_idx]}" 1 "${gpu_id}"
      ) &
      pids+=("$!")
      jobs_started=$((jobs_started + 1))
    done
    if [ "${jobs_started}" -gt 0 ]; then
      for pid in "${pids[@]}"; do
        wait "${pid}"
      done
      write_result_summary
    else
      echo "[pair] indices ${idx}-$((idx + 1)) already completed"
    fi
    idx=$((idx + 2))
  done
else
  for idx in "${!CONFIGS[@]}"; do
    if [ "${idx}" -lt "${START_AT_INDEX}" ]; then
      continue
    fi
    run_one "${CONFIGS[$idx]}" "${NAMES[$idx]}" "${NUM_GPU}"
    write_result_summary
  done
fi

if [ "${RUN_STRESS}" = "1" ]; then
  STRESS_CSV="tools/concerto_projection_shortcut/results_arkit_full_stress.csv"
  STRESS_MD="tools/concerto_projection_shortcut/results_arkit_full_stress.md"
  : > "${STRESS_CSV}"
  echo "checkpoint,stress,batches,enc2d_loss_mean" > "${STRESS_CSV}"

  STRESS_CHECKPOINTS=(
    "${EXP_PREFIX}-baseline"
    "${EXP_PREFIX}-coord-mlp"
  )
  if [ "${INCLUDE_FIX}" = "1" ]; then
    STRESS_CHECKPOINTS+=("${EXP_PREFIX}-coord-residual-target")
  fi

  for exp_name in "${STRESS_CHECKPOINTS[@]}"; do
    checkpoint="exp/${DATASET_NAME}/${exp_name}/model/model_last.pth"
    result_tmp="$(mktemp)"
    "${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_enc2d_stress.py \
      --config pretrain-concerto-v1m1-0-probe-enc2d-full-baseline \
      --weight "${checkpoint}" \
      --data-root "${ARKIT_FULL_META_ROOT}" \
      --max-batches "${MAX_STRESS_BATCHES}" > "${result_tmp}"
    tail -n +2 "${result_tmp}" | sed "s#^#${exp_name},#" >> "${STRESS_CSV}"
    rm -f "${result_tmp}"
  done

  "${PYTHON_BIN}" - "${STRESS_CSV}" "${STRESS_MD}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))
lines = [
    "# ARKit Full Stress",
    "",
    "| checkpoint | stress | batches | enc2d loss mean |",
    "| --- | --- | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| {row['checkpoint']} | {row['stress']} | {row['batches']} | {row['enc2d_loss_mean']} |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
fi

write_result_summary
date '+%F %T' > "${DONE_STAMP}"
echo "[done] wrote ${RESULT_CSV} and ${RESULT_MD}"
