#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

DATASET_NAME="${DATASET_NAME:-concerto}"
LIN_CONFIG="${LIN_CONFIG:-configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-safe.py}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
GPU_IDS_CSV="${GPU_IDS_CSV:-2,3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/repcomp_linear_seed_repeats}"
SUMMARY_CSV="${SUMMARY_CSV:-tools/concerto_projection_shortcut/results_scannet_repcomp_linear_seed_repeats.csv}"
SUMMARY_MD="${SUMMARY_MD:-tools/concerto_projection_shortcut/results_scannet_repcomp_linear_seed_repeats.md}"

mkdir -p "${LOG_DIR}"

declare -a VARIANT_SPECS=(
  "coord-mlp|exp/${DATASET_NAME}/arkit-full-continue-coord-mlp/model/model_last.pth|scannet-proxy-coord-mlp-continue-lin-seed"
  "no-enc2d|exp/${DATASET_NAME}/arkit-full-continue-no-enc2d/model/model_last.pth|scannet-proxy-no-enc2d-continue-lin-seed"
)

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

run_one() {
  local variant="$1"
  local weight_path="$2"
  local exp_prefix="$3"
  local seed="$4"
  local gpu_id="$5"

  local exp_name="${exp_prefix}${seed}"
  local save_path="exp/${DATASET_NAME}/${exp_name}"
  local log_path="${LOG_DIR}/${exp_name}.log"

  if [ ! -f "${weight_path}" ]; then
    echo "[error] missing checkpoint for ${variant}: ${weight_path}" >&2
    return 2
  fi
  if [ -f "${save_path}/model/model_last.pth" ] && grep -q "Val result: mIoU/mAcc/allAcc" "${save_path}/train.log" 2>/dev/null; then
    echo "[skip] ${exp_name} already finished"
    return 0
  fi

  echo "[run] gpu=${gpu_id} variant=${variant} seed=${seed} exp=${exp_name}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "${PYTHON_BIN}" tools/train.py \
    --config-file "${LIN_CONFIG}" \
    --num-gpus 1 \
    --num-machines 1 \
    --machine-rank 0 \
    --dist-url "tcp://127.0.0.1:$((17000 + seed * 101 + gpu_id))" \
    --options \
      save_path="${save_path}" \
      resume=False \
      weight="${weight_path}" \
      seed="${seed}" \
    > "${log_path}" 2>&1
}

summarize() {
  local -a files=()
  while IFS= read -r file; do
    files+=("${file}")
  done < <(find "exp/${DATASET_NAME}" -path "*/scannet-proxy-*-continue-lin-seed*/train.log" | sort)
  if [ "${#files[@]}" -eq 0 ]; then
    echo "[summary] no seed-repeat logs found yet"
    return 0
  fi
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_semseg_logs.py "${files[@]}" > "${SUMMARY_CSV}"
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))

def variant_and_seed(log: str):
    name = Path(log).parent.name
    seed_match = re.search(r"seed(\d+)$", name)
    seed = seed_match.group(1) if seed_match else ""
    if "coord-mlp" in name:
        variant = "coord-mlp"
    elif "no-enc2d" in name:
        variant = "no-enc2d"
    elif "concerto" in name:
        variant = "concerto"
    else:
        variant = "unknown"
    return variant, seed, name

lines = [
    "# ScanNet Linear Proxy Seed Repeats",
    "",
    "| variant | seed | experiment | status | final mIoU | final mAcc | final allAcc | best metric | best value | eval count |",
    "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
]
by_variant = defaultdict(list)
for row in rows:
    variant, seed, name = variant_and_seed(row["log"])
    try:
        miou = float(row["val_miou_last"])
        macc = float(row["val_macc_last"])
        allacc = float(row["val_allacc_last"])
        best = float(row["best_metric_value"])
        eval_count = row["val_eval_count"]
    except Exception:
        continue
    by_variant[variant].append((miou, macc, allacc, best))
    lines.append(
        f"| {variant} | {seed} | `{name}` | {row['status']} | "
        f"{miou:.4f} | {macc:.4f} | {allacc:.4f} | {row['best_metric_name']} | {best:.4f} | {eval_count} |"
    )

lines += ["", "## Aggregate", ""]
lines.append("| variant | n | final mIoU mean | final mIoU std | final mAcc mean | final allAcc mean |")
lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
for variant in sorted(by_variant):
    vals = by_variant[variant]
    n = len(vals)
    miou = [v[0] for v in vals]
    macc = [v[1] for v in vals]
    allacc = [v[2] for v in vals]
    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    lines.append(f"| {variant} | {n} | {mean(miou):.4f} | {std(miou):.4f} | {mean(macc):.4f} | {mean(allacc):.4f} |")

if "coord-mlp" in by_variant and "no-enc2d" in by_variant:
    diffs = [a[0] - b[0] for a, b in zip(sorted(by_variant["coord-mlp"]), sorted(by_variant["no-enc2d"]))]
    if diffs:
        m = sum(diffs) / len(diffs)
        s = 0.0 if len(diffs) < 2 else (sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
        lines += ["", f"Coord-MLP minus no-enc2d paired final mIoU: mean {m:.4f}, std {s:.4f} over {len(diffs)} paired rows."]

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  echo "[summary] wrote ${SUMMARY_CSV} and ${SUMMARY_MD}"
}

declare -a TASKS=()
for spec in "${VARIANT_SPECS[@]}"; do
  IFS='|' read -r variant weight exp_prefix <<< "${spec}"
  for seed in "${SEEDS[@]}"; do
    TASKS+=("${variant}|${weight}|${exp_prefix}|${seed}")
  done
done

declare -a SLOT_PID=()
declare -a SLOT_LABEL=()
next_idx=0
active=0

while [ "${next_idx}" -lt "${#TASKS[@]}" ] || [ "${active}" -gt 0 ]; do
  for slot in "${!GPU_IDS[@]}"; do
    pid="${SLOT_PID[$slot]:-}"
    if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
      if ! wait "${pid}"; then
        echo "[warn] ${SLOT_LABEL[$slot]} exited non-zero" >&2
      fi
      SLOT_PID[$slot]=""
      SLOT_LABEL[$slot]=""
      summarize || true
    fi
    if [ -z "${SLOT_PID[$slot]:-}" ] && [ "${next_idx}" -lt "${#TASKS[@]}" ]; then
      IFS='|' read -r variant weight exp_prefix seed <<< "${TASKS[$next_idx]}"
      next_idx=$((next_idx + 1))
      gpu_id="${GPU_IDS[$slot]}"
      label="${exp_prefix}${seed}"
      ( run_one "${variant}" "${weight}" "${exp_prefix}" "${seed}" "${gpu_id}" ) &
      SLOT_PID[$slot]="$!"
      SLOT_LABEL[$slot]="${label}"
      sleep 5
    fi
  done
  active=0
  for slot in "${!GPU_IDS[@]}"; do
    if [ -n "${SLOT_PID[$slot]:-}" ]; then
      active=$((active + 1))
    fi
  done
  if [ "${active}" -gt 0 ]; then
    sleep 30
  fi
done

summarize
