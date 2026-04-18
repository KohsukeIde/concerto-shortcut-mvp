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
OUT_ROOT="${OUT_ROOT:-${POINTCEPT_DATA_ROOT}/runs/scannet_classwise_diagnosis/large_video_sr_lora}"
NUM_WORKER="${NUM_WORKER:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCHES="${MAX_BATCHES:--1}"

BASELINE_WEIGHT="${BASELINE_WEIGHT:-${REPO_ROOT}/exp/concerto/scannet-proxy-large-video-official-lin/model/model_last.pth}"
SR_M01_WEIGHT="${SR_M01_WEIGHT:-${REPO_ROOT}/exp/concerto/scannet-proxy-sr-lora-v5-r4-d0p3-i256-qf4-lin/model/model_last.pth}"
SR_M02_WEIGHT="${SR_M02_WEIGHT:-${REPO_ROOT}/exp/concerto/scannet-proxy-sr-lora-v5-r4-d0p3-m0p2-i256-qf4-lin/model/model_last.pth}"

BASELINE_LOG="${BASELINE_LOG:-${REPO_ROOT}/exp/concerto/scannet-proxy-large-video-official-lin/train.log}"
SR_M01_LOG="${SR_M01_LOG:-${REPO_ROOT}/exp/concerto/scannet-proxy-sr-lora-v5-r4-d0p3-i256-qf4-lin/train.log}"
SR_M02_LOG="${SR_M02_LOG:-${REPO_ROOT}/exp/concerto/scannet-proxy-sr-lora-v5-r4-d0p3-m0p2-i256-qf4-lin/train.log}"

mkdir -p "${OUT_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/summarize_scannet_classwise.py" \
  --baseline baseline \
  --spec "baseline=${BASELINE_LOG}" \
  --spec "sr_lora_m01=${SR_M01_LOG}" \
  --spec "sr_lora_m02=${SR_M02_LOG}" \
  --output-csv "${OUT_ROOT}/classwise_from_logs.csv" \
  --output-md "${OUT_ROOT}/classwise_from_logs.md"

run_one() {
  local label="$1"
  local weight="$2"
  local gpu="$3"
  local summary="${OUT_ROOT}/${label}_summary.json"
  if [ -f "${summary}" ]; then
    echo "[skip] ${label}: ${summary}"
    return 0
  fi
  echo "[eval] label=${label} gpu=${gpu} weight=${weight}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/eval_scannet_semseg_confusion.py" \
      --config "${CONFIG}" \
      --weight "${weight}" \
      --label "${label}" \
      --output-dir "${OUT_ROOT}" \
      --batch-size "${BATCH_SIZE}" \
      --num-worker "${NUM_WORKER}" \
      --max-batches "${MAX_BATCHES}"
}

run_one baseline "${BASELINE_WEIGHT}" 0 &
p0="$!"
run_one sr_lora_m01 "${SR_M01_WEIGHT}" 1 &
p1="$!"
run_one sr_lora_m02 "${SR_M02_WEIGHT}" 2 &
p2="$!"

wait "${p0}"
wait "${p1}"
wait "${p2}"

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = ["baseline", "sr_lora_m01", "sr_lora_m02"]
summaries = {label: json.loads((root / f"{label}_summary.json").read_text()) for label in labels}
metrics = {
    label: {row["class_name"]: row for row in csv.DictReader((root / f"{label}_class_metrics.csv").open())}
    for label in labels
}
confusions = {
    label: list(csv.DictReader((root / f"{label}_top_confusions.csv").open()))
    for label in labels
}

lines = [
    "# ScanNet Class-wise and Confusion Diagnosis",
    "",
    "## Overall From Confusion Evaluation",
    "",
    "| experiment | mIoU | mAcc | allAcc | batches |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for label in labels:
    s = summaries[label]
    lines.append(f"| {label} | {s['mIoU']:.4f} | {s['mAcc']:.4f} | {s['allAcc']:.4f} | {s['batches']} |")

base = metrics["baseline"]
weak = sorted(base.values(), key=lambda row: float(row["iou"]))[:8]
lines += [
    "",
    "## Weakest Baseline Classes",
    "",
    "| rank | class | IoU | accuracy | target share | top non-self confusions |",
    "| ---: | --- | ---: | ---: | ---: | --- |",
]
for rank, row in enumerate(weak, start=1):
    cls = row["class_name"]
    tops = [
        f"{r['pred_name']} {float(r['fraction_of_target']):.3f}"
        for r in confusions["baseline"]
        if r["target_name"] == cls
    ][:3]
    lines.append(
        f"| {rank} | {cls} | {float(row['iou']):.4f} | {float(row['accuracy']):.4f} | "
        f"{float(row['target_share']):.4f} | {', '.join(tops)} |"
    )

for label in ["sr_lora_m01", "sr_lora_m02"]:
    rows = []
    for cls, row in metrics[label].items():
        rows.append(
            {
                "class": cls,
                "base_iou": float(base[cls]["iou"]),
                "iou": float(row["iou"]),
                "delta": float(row["iou"]) - float(base[cls]["iou"]),
                "base_acc": float(base[cls]["accuracy"]),
                "acc": float(row["accuracy"]),
                "delta_acc": float(row["accuracy"]) - float(base[cls]["accuracy"]),
            }
        )
    lines += [
        "",
        f"## Class Delta: {label} - baseline",
        "",
        "| class | baseline IoU | SR IoU | delta IoU | baseline acc | SR acc | delta acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda x: x["delta"]):
        lines.append(
            f"| {row['class']} | {row['base_iou']:.4f} | {row['iou']:.4f} | {row['delta']:+.4f} | "
            f"{row['base_acc']:.4f} | {row['acc']:.4f} | {row['delta_acc']:+.4f} |"
        )

lines += [
    "",
    "## Readout",
    "",
    "- The first intervention target should be chosen from low-IoU classes and dominant confusions, not from aggregate mIoU alone.",
    "- If SR-LoRA only changes high-IoU layout classes or creates mixed small deltas, it is not solving the downstream bottleneck.",
    "- Use this table to choose class-aware or confusion-aware pressure before spending more points on broad SR-LoRA sweeps.",
]
(root / "classwise_confusion_summary.md").write_text("\n".join(lines) + "\n")
print(root / "classwise_confusion_summary.md")
PY

echo "[done] ${OUT_ROOT}"
