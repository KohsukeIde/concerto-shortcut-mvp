#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPU="${NUM_GPU:-2}"
DATASET_NAME="${DATASET_NAME:-concerto}"
MODE="${1:-all}"
OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${REPO_ROOT}/weights/concerto/concerto_base_origin.pth}"

CONTINUE_CONFIGS=(
  "pretrain-concerto-v1m1-0-arkit-full-continue"
  "pretrain-concerto-v1m1-0-arkit-full-no-enc2d-continue"
  "pretrain-concerto-v1m1-0-arkit-full-coord-mlp-continue"
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
  bash scripts/train.sh \
    -p "${PYTHON_BIN}" \
    -d "${DATASET_NAME}" \
    -g "${NUM_GPU}" \
    -c "${config_name}" \
    -n "${exp_name}" \
    -w "${weight_path}"
}

run_official_gate() {
  echo "[gate] official ScanNet linear probe"
  run_train "semseg-ptv3-base-v1m1-0a-scannet-lin-proxy" "scannet-proxy-official-origin-lin" "${OFFICIAL_WEIGHT}"
  echo "[gate] official ScanNet fine-tune"
  run_train "semseg-ptv3-base-v1m1-0c-scannet-ft-proxy" "scannet-proxy-official-origin-ft" "${OFFICIAL_WEIGHT}"
}

run_continuations() {
  for idx in "${!CONTINUE_CONFIGS[@]}"; do
    echo "[continue] ${CONTINUE_CONFIGS[$idx]} -> ${CONTINUE_NAMES[$idx]}"
    run_train "${CONTINUE_CONFIGS[$idx]}" "${CONTINUE_NAMES[$idx]}" "${OFFICIAL_WEIGHT}"
  done
}

run_downstream_from_checkpoints() {
  local task_config="$1"
  local task_suffix="$2"
  for idx in "${!CONTINUE_NAMES[@]}"; do
    checkpoint="exp/${DATASET_NAME}/${CONTINUE_NAMES[$idx]}/model/model_last.pth"
    exp_name="scannet-proxy-${PROBE_LABELS[$idx]}-${task_suffix}"
    echo "[downstream] ${task_config} <- ${checkpoint}"
    run_train "${task_config}" "${exp_name}" "${checkpoint}"
  done
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
    exp_name = Path(row["log"]).parents[1].name
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
  pretrain)
    run_continuations
    ;;
  lin)
    run_downstream_from_checkpoints "semseg-ptv3-base-v1m1-0a-scannet-lin-proxy" "lin"
    ;;
  ft)
    run_downstream_from_checkpoints "semseg-ptv3-base-v1m1-0c-scannet-ft-proxy" "ft"
    ;;
  all)
    run_official_gate
    run_continuations
    run_downstream_from_checkpoints "semseg-ptv3-base-v1m1-0a-scannet-lin-proxy" "lin"
    run_downstream_from_checkpoints "semseg-ptv3-base-v1m1-0c-scannet-ft-proxy" "ft"
    ;;
  *)
    echo "usage: $0 [gate|pretrain|lin|ft|all]" >&2
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
