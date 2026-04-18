#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

if [ "${SKIP_VENV_ACTIVATE:-0}" != "1" ]; then
  ensure_venv_active
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-semseg-ptv3-large-v1m1-0a-scannet-lin-proxy-valonly}"
WEIGHT="${WEIGHT:-${REPO_ROOT}/exp/concerto/scannet-proxy-large-video-official-lin/model/model_last.pth}"
OUT_ROOT="${OUT_ROOT:-${POINTCEPT_DATA_ROOT}/runs/scannet_counterfactual_downstream/large_video_official}"
OUTPUT="${OUTPUT:-${OUT_ROOT}/counterfactual_stress.csv}"
STRESS_LIST="${STRESS_LIST:-clean z_shift_p025 z_shift_p050 z_shift_p100 z_scale_050 z_scale_150 xy_shift_post_p050}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKER="${NUM_WORKER:-4}"
MAX_BATCHES="${MAX_BATCHES:--1}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${OUT_ROOT}"

echo "[run] config=${CONFIG}"
echo "[run] weight=${WEIGHT}"
echo "[run] output=${OUTPUT}"
echo "[run] stress=${STRESS_LIST}"
echo "[run] max_batches=${MAX_BATCHES}"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/eval_scannet_semseg_stress.py" \
    --config "${CONFIG}" \
    --weight "${WEIGHT}" \
    --output "${OUTPUT}.tmp" \
    --stress ${STRESS_LIST} \
    --batch-size "${BATCH_SIZE}" \
    --num-worker "${NUM_WORKER}" \
    --max-batches "${MAX_BATCHES}"

mv "${OUTPUT}.tmp" "${OUTPUT}"
class_tmp="${OUTPUT%.csv}_classwise.csv.tmp"
class_out="${OUTPUT%.csv}_classwise.csv"
if [ -f "${class_tmp}" ]; then
  mv "${class_tmp}" "${class_out}"
fi

"${PYTHON_BIN}" - "${OUTPUT}" "${class_out}" "${OUT_ROOT}/counterfactual_summary.md" <<'PY'
import csv
import sys
from pathlib import Path

stress_csv = Path(sys.argv[1])
class_csv = Path(sys.argv[2])
out_md = Path(sys.argv[3])

rows = list(csv.DictReader(stress_csv.open()))
class_rows = list(csv.DictReader(class_csv.open())) if class_csv.exists() else []
clean = next(row for row in rows if row["stress"] == "clean")
clean_miou = float(clean["mIoU"])

lines = [
    "# ScanNet Counterfactual Downstream Stress",
    "",
    "## Overall",
    "",
    "| stress | mIoU | delta vs clean | mAcc | allAcc | batches |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    miou = float(row["mIoU"])
    lines.append(
        f"| {row['stress']} | {miou:.4f} | {miou - clean_miou:+.4f} | "
        f"{float(row['mAcc']):.4f} | {float(row['allAcc']):.4f} | {int(row['batches'])} |"
    )

if class_rows:
    clean_by_class = {
        row["class_name"]: float(row["iou"])
        for row in class_rows
        if row["stress"] == "clean"
    }
    focus = [
        "picture",
        "wall",
        "counter",
        "desk",
        "sink",
        "cabinet",
        "shower curtain",
        "door",
        "window",
    ]
    lines += [
        "",
        "## Focus Class Deltas",
        "",
        "| stress | class | IoU | delta vs clean |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in class_rows:
        cls = row["class_name"]
        if row["stress"] == "clean" or cls not in focus:
            continue
        iou = float(row["iou"])
        lines.append(
            f"| {row['stress']} | {cls} | {iou:.4f} | {iou - clean_by_class[cls]:+.4f} |"
        )

lines += [
    "",
    "## Readout",
    "",
    "- `z_shift_*` changes floor-relative height after the first centering step.",
    "- `z_scale_*` scales floor-relative height.",
    "- `xy_shift_post_*` is inserted after final xy recentering, so it tests residual coordinate-feature dependence rather than being canceled by the standard transform.",
]
out_md.write_text("\n".join(lines) + "\n")
print(out_md)
PY

cat "${OUT_ROOT}/counterfactual_summary.md"
