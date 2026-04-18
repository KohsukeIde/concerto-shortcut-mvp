#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

if [ "${SKIP_VENV_ACTIVATE:-0}" != "1" ]; then
  ensure_venv_active
fi

DATASET_NAME="${DATASET_NAME:-concerto}"
LINEAR_CONFIG="${LINEAR_CONFIG:-semseg-ptv3-large-v1m1-0a-scannet-lin-proxy-valonly}"
STRESS_CONFIG="${STRESS_CONFIG:-${LINEAR_CONFIG}}"
BASELINE_WEIGHT="${BASELINE_WEIGHT:-${WEIGHT_DIR}/pretrain-concerto-v1m1-2-large-video.pth}"
SR_WEIGHT="${SR_WEIGHT:-${REPO_ROOT}/exp/concerto/sr-lora-v5-r4-d0p3-i256-qf4-matrix/model/model_last_merged_lora.pth}"
BASELINE_EXP="${BASELINE_EXP:-scannet-proxy-large-video-official-lin}"
SR_EXP="${SR_EXP:-scannet-proxy-sr-lora-v5-r4-d0p3-i256-qf4-lin}"
OUT_ROOT="${OUT_ROOT:-${POINTCEPT_DATA_ROOT}/runs/sr_lora_phasea/followup/r4-d0p3-i256}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_ROOT}/linear_summary.csv}"
SUMMARY_MD="${SUMMARY_MD:-${OUT_ROOT}/linear_summary.md}"
COMPARE_JSON="${COMPARE_JSON:-${OUT_ROOT}/followup_compare.json}"
COMPARE_MD="${COMPARE_MD:-${OUT_ROOT}/followup_compare.md}"

RUN_LINEAR="${RUN_LINEAR:-1}"
RUN_STRESS="${RUN_STRESS:-1}"
BASELINE_GPU_IDS="${BASELINE_GPU_IDS:-0,1}"
SR_GPU_IDS="${SR_GPU_IDS:-2,3}"
BASELINE_NUM_GPU="${BASELINE_NUM_GPU:-$(awk -F',' '{print NF}' <<< "${BASELINE_GPU_IDS}")}"
SR_NUM_GPU="${SR_NUM_GPU:-$(awk -F',' '{print NF}' <<< "${SR_GPU_IDS}")}"
BASELINE_STRESS_GPU="${BASELINE_STRESS_GPU:-0}"
SR_STRESS_GPU="${SR_STRESS_GPU:-1}"
STRESS_BATCH_SIZE="${STRESS_BATCH_SIZE:-1}"
STRESS_NUM_WORKER="${STRESS_NUM_WORKER:-4}"
STRESS_MAX_BATCHES="${STRESS_MAX_BATCHES:--1}"
STRESS_LIST="${STRESS_LIST:-clean local_surface_destroy z_flip xy_swap roll_90_x}"

mkdir -p "${OUT_ROOT}"

run_train_one() {
  local exp_name="$1"
  local weight_path="$2"
  local gpu_ids="$3"
  local num_gpu="$4"
  local checkpoint="${REPO_ROOT}/exp/${DATASET_NAME}/${exp_name}/model/model_last.pth"
  if [ -f "${checkpoint}" ]; then
    echo "[$(timestamp)] skip linear: ${checkpoint}"
    return 0
  fi
  echo "[$(timestamp)] linear: exp=${exp_name} gpu_ids=${gpu_ids} num_gpu=${num_gpu} weight=${weight_path}"
  CUDA_VISIBLE_DEVICES="${gpu_ids}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    bash "${REPO_ROOT}/scripts/train.sh" \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g "${num_gpu}" \
      -c "${LINEAR_CONFIG}" \
      -n "${exp_name}" \
      -w "${weight_path}"
}

run_stress_one() {
  local label="$1"
  local exp_name="$2"
  local gpu_id="$3"
  local checkpoint="${REPO_ROOT}/exp/${DATASET_NAME}/${exp_name}/model/model_last.pth"
  local out_csv="${OUT_ROOT}/${label}_scannet_stress.csv"
  if [ -f "${out_csv}" ]; then
    echo "[$(timestamp)] skip stress: ${out_csv}"
    cat "${out_csv}"
    return 0
  fi
  if [ ! -f "${checkpoint}" ]; then
    echo "[$(timestamp)] error: missing linear checkpoint: ${checkpoint}" >&2
    return 2
  fi
  echo "[$(timestamp)] stress: label=${label} gpu=${gpu_id} checkpoint=${checkpoint}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/eval_scannet_semseg_stress.py" \
      --config "${STRESS_CONFIG}" \
      --weight "${checkpoint}" \
      --output "${out_csv}.tmp" \
      --stress ${STRESS_LIST} \
      --batch-size "${STRESS_BATCH_SIZE}" \
      --num-worker "${STRESS_NUM_WORKER}" \
      --max-batches "${STRESS_MAX_BATCHES}"
  mv "${out_csv}.tmp" "${out_csv}"
  cat "${out_csv}"
}

write_summaries() {
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_semseg_logs.py \
    "exp/${DATASET_NAME}/${BASELINE_EXP}/train.log" \
    "exp/${DATASET_NAME}/${SR_EXP}/train.log" \
    > "${SUMMARY_CSV}"
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${SUMMARY_MD}" "${OUT_ROOT}/baseline_scannet_stress.csv" "${OUT_ROOT}/sr_lora_scannet_stress.csv" "${COMPARE_JSON}" "${COMPARE_MD}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_csv, summary_md, baseline_stress, sr_stress, compare_json, compare_md = map(Path, sys.argv[1:])
rows = list(csv.DictReader(summary_csv.open()))

def exp_name(row):
    return Path(row["log"]).parent.name

def f(row, key):
    return float(str(row[key]).rstrip("."))

payload = {"linear": {}, "stress": {}}
for row in rows:
    name = exp_name(row)
    payload["linear"][name] = {
        "last_miou": f(row, "val_miou_last"),
        "last_macc": f(row, "val_macc_last"),
        "last_allacc": f(row, "val_allacc_last"),
        "best_metric_name": row["best_metric_name"],
        "best_metric_value": f(row, "best_metric_value"),
        "eval_count": int(float(row["val_eval_count"])),
    }

def read_stress(path):
    if not path.exists():
        return {}
    return {row["stress"]: row for row in csv.DictReader(path.open())}

base_s = read_stress(baseline_stress)
sr_s = read_stress(sr_stress)
for stress, sr_row in sr_s.items():
    if stress not in base_s:
        continue
    b = base_s[stress]
    payload["stress"][stress] = {
        "baseline_miou": float(b["mIoU"]),
        "sr_lora_miou": float(sr_row["mIoU"]),
        "delta_miou": float(sr_row["mIoU"]) - float(b["mIoU"]),
        "baseline_allacc": float(b["allAcc"]),
        "sr_lora_allacc": float(sr_row["allAcc"]),
        "delta_allacc": float(sr_row["allAcc"]) - float(b["allAcc"]),
        "batches": int(sr_row["batches"]),
    }

names = list(payload["linear"])
if len(names) >= 2:
    baseline_name = next((n for n in names if "large-video-official" in n), names[0])
    sr_name = next((n for n in names if "sr-lora" in n), names[-1])
    payload["linear_delta"] = {
        "baseline": baseline_name,
        "sr_lora": sr_name,
        "last_miou": payload["linear"][sr_name]["last_miou"] - payload["linear"][baseline_name]["last_miou"],
        "best_miou": payload["linear"][sr_name]["best_metric_value"] - payload["linear"][baseline_name]["best_metric_value"],
    }

compare_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

lines = [
    "# SR-LoRA r4 d0.3 Follow-up",
    "",
    "## Linear",
    "",
    "| experiment | last mIoU | best mIoU | eval count |",
    "| --- | ---: | ---: | ---: |",
]
for name, row in payload["linear"].items():
    lines.append(f"| {name} | {row['last_miou']:.4f} | {row['best_metric_value']:.4f} | {row['eval_count']} |")
if "linear_delta" in payload:
    d = payload["linear_delta"]
    lines += [
        "",
        f"Linear delta SR-LoRA - baseline: last `{d['last_miou']:+.4f}`, best `{d['best_miou']:+.4f}`.",
    ]
if payload["stress"]:
    lines += [
        "",
        "## Stress",
        "",
        "| stress | baseline mIoU | SR-LoRA mIoU | delta | batches |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stress, row in payload["stress"].items():
        lines.append(
            f"| {stress} | {row['baseline_miou']:.4f} | {row['sr_lora_miou']:.4f} | "
            f"{row['delta_miou']:+.4f} | {row['batches']} |"
        )
compare_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

summary_lines = [
    "# Linear Summary",
    "",
    "| experiment | last mIoU | last mAcc | last allAcc | best metric | best value | eval count |",
    "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
]
for row in rows:
    summary_lines.append(
        f"| {exp_name(row)} | {row['val_miou_last']} | {row['val_macc_last']} | {row['val_allacc_last']} | "
        f"{row['best_metric_name']} | {row['best_metric_value']} | {row['val_eval_count']} |"
    )
summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
  cat "${COMPARE_MD}"
}

echo "[$(timestamp)] start SR-LoRA r4 d0.3 follow-up"
echo "linear_config=${LINEAR_CONFIG}"
echo "stress_config=${STRESS_CONFIG}"
echo "baseline_weight=${BASELINE_WEIGHT}"
echo "sr_weight=${SR_WEIGHT}"
echo "baseline_exp=${BASELINE_EXP}"
echo "sr_exp=${SR_EXP}"
echo "out_root=${OUT_ROOT}"
echo "run_linear=${RUN_LINEAR}"
echo "run_stress=${RUN_STRESS}"
echo "baseline_gpu_ids=${BASELINE_GPU_IDS}"
echo "sr_gpu_ids=${SR_GPU_IDS}"
echo "baseline_num_gpu=${BASELINE_NUM_GPU}"
echo "sr_num_gpu=${SR_NUM_GPU}"

if [ "${RUN_LINEAR}" = "1" ]; then
  run_train_one "${BASELINE_EXP}" "${BASELINE_WEIGHT}" "${BASELINE_GPU_IDS}" "${BASELINE_NUM_GPU}" &
  baseline_pid="$!"
  run_train_one "${SR_EXP}" "${SR_WEIGHT}" "${SR_GPU_IDS}" "${SR_NUM_GPU}" &
  sr_pid="$!"
  wait "${baseline_pid}"
  wait "${sr_pid}"
fi

if [ "${RUN_STRESS}" = "1" ]; then
  run_stress_one baseline "${BASELINE_EXP}" "${BASELINE_STRESS_GPU}" &
  baseline_stress_pid="$!"
  run_stress_one sr_lora "${SR_EXP}" "${SR_STRESS_GPU}" &
  sr_stress_pid="$!"
  wait "${baseline_stress_pid}"
  wait "${sr_stress_pid}"
fi

write_summaries
echo "[$(timestamp)] done SR-LoRA r4 d0.3 follow-up"
