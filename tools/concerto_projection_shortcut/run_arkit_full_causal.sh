#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPU="${NUM_GPU:-2}"
DATASET_NAME="${DATASET_NAME:-concerto}"
EXP_PREFIX="${EXP_PREFIX:-arkit-full-causal}"
INCLUDE_FIX="${INCLUDE_FIX:-0}"
RUN_STRESS="${RUN_STRESS:-0}"
MAX_STRESS_BATCHES="${MAX_STRESS_BATCHES:-20}"
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

for idx in "${!CONFIGS[@]}"; do
  echo "[run] ${CONFIGS[$idx]} -> ${NAMES[$idx]}"
  bash scripts/train.sh \
    -p "${PYTHON_BIN}" \
    -d "${DATASET_NAME}" \
    -g "${NUM_GPU}" \
    -c "${CONFIGS[$idx]}" \
    -n "${NAMES[$idx]}"
done

RESULT_CSV="tools/concerto_projection_shortcut/results_arkit_full_causal.csv"
RESULT_MD="tools/concerto_projection_shortcut/results_arkit_full_causal.md"
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
    exp_name = Path(row["log"]).parents[1].name
    lines.append(
        f"| {exp_name} | {row['enc2d_loss_first']} | {row['enc2d_loss_last']} | "
        f"{row['enc2d_loss_min']} | {row['enc2d_loss_count']} |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

if [ "${RUN_STRESS}" = "1" ]; then
  STRESS_CSV="tools/concerto_projection_shortcut/results_arkit_full_stress.csv"
  : > "${STRESS_CSV}"
  echo "checkpoint,stress,batches,enc2d_loss_mean" > "${STRESS_CSV}"

  STRESS_CHECKPOINTS=("${EXP_PREFIX}-baseline")
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
fi

echo "[done] wrote ${RESULT_CSV} and ${RESULT_MD}"
