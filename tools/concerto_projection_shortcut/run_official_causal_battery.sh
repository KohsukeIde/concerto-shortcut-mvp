#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${WEIGHT_DIR}/pretrain-concerto-v1m1-2-large-video.pth}"
OFFICIAL_CONFIG_STEM="${OFFICIAL_CONFIG_STEM:-pretrain-concerto-v1m1-2-large-video-probe-enc2d}"
EXP_TAG="${EXP_TAG:-official-step0}"
MAX_BATCHES="${MAX_BATCHES:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKER="${NUM_WORKER:-2}"
STRESS_LIST="${STRESS_LIST:-clean}"
INCLUDE_SCANNET="${INCLUDE_SCANNET:-1}"
REQUIRE_SCANNET="${REQUIRE_SCANNET:-1}"
RESULT_ROOT="${RESULT_ROOT:-${POINTCEPT_DATA_ROOT}/runs/official_causal_battery/${EXP_TAG}}"
RESULT_CSV="${RESULT_CSV:-tools/concerto_projection_shortcut/results_official_causal_battery.csv}"
RESULT_MD="${RESULT_MD:-tools/concerto_projection_shortcut/results_official_causal_battery.md}"

if [ ! -f "${OFFICIAL_WEIGHT}" ]; then
  echo "[error] official weight not found: ${OFFICIAL_WEIGHT}" >&2
  echo "        download it from Pointcept/Concerto, e.g. pretrain-concerto-v1m1-2-large-video.pth" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${OFFICIAL_WEIGHT}" <<'PY'
import sys
import torch
weight = sys.argv[1]
checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
if not any("enc2d_head" in key for key in state_dict):
    raise SystemExit(
        f"[error] {weight} does not contain enc2d_head weights; "
        "released backbone-only Concerto weights are not valid for enc2d causal evaluation."
    )
print(f"[ok] official full pretraining weight has enc2d_head keys: {weight}")
PY

mkdir -p "${RESULT_ROOT}" "$(dirname "${RESULT_CSV}")" "$(dirname "${RESULT_MD}")"

"${PYTHON_BIN}" tools/concerto_projection_shortcut/prepare_arkit_full_splits.py \
  --source-root "${ARKIT_FULL_SOURCE_ROOT}" \
  --output-root "${ARKIT_FULL_META_ROOT}"

resolve_scannet_imagepoint_root() {
  local root="${SCANNET_IMAGEPOINT_META_ROOT}"
  if [ -f "${root}/splits/val.json" ] || [ -f "${root}/splits/train.json" ]; then
    printf '%s\n' "${root}"
    return 0
  fi
  root="${SCANNET_IMAGEPOINT_ROOT}"
  if [ -f "${root}/splits/val.json" ] || [ -f "${root}/splits/train.json" ]; then
    printf '%s\n' "${root}"
    return 0
  fi
  if [ -f "${root}/scannet/splits/val.json" ] || [ -f "${root}/scannet/splits/train.json" ]; then
    printf '%s\n' "${root}/scannet"
    return 0
  fi
  find "${root}" -maxdepth 3 -type f \( -path '*/splits/val.json' -o -path '*/splits/train.json' \) \
    | sed 's#/splits/[^/]*$##' \
    | head -n 1
}

SCANNET_CAUSAL_ROOT="$(resolve_scannet_imagepoint_root || true)"
if [ "${INCLUDE_SCANNET}" = "1" ] && [ -z "${SCANNET_CAUSAL_ROOT}" ]; then
  if [ "${REQUIRE_SCANNET}" = "1" ]; then
    echo "[error] Concerto ScanNet image-point root is not prepared under ${SCANNET_IMAGEPOINT_META_ROOT} or ${SCANNET_IMAGEPOINT_ROOT}" >&2
    echo "        run: bash tools/concerto_projection_shortcut/setup_concerto_scannet_imagepoint.sh" >&2
    exit 3
  fi
  echo "[warn] skipping ScanNet because image-point root is not prepared"
fi

declare -a DATASETS=(
  "arkit|${ARKIT_FULL_META_ROOT}|Validation"
)
if [ "${INCLUDE_SCANNET}" = "1" ] && [ -n "${SCANNET_CAUSAL_ROOT}" ]; then
  DATASETS+=("scannet|${SCANNET_CAUSAL_ROOT}|val")
fi

declare -a MODES=(
  "baseline|${OFFICIAL_CONFIG_STEM}-baseline"
  "global_target_permutation|${OFFICIAL_CONFIG_STEM}-global-target-permutation"
  "cross_image_target_swap|${OFFICIAL_CONFIG_STEM}-cross-image-target-swap"
  "cross_scene_target_swap|${OFFICIAL_CONFIG_STEM}-cross-scene-target-swap"
)

RAW_CSV="${RESULT_ROOT}/raw.csv"
: > "${RAW_CSV}"
echo "dataset,mode,stress,batches,enc2d_loss_mean,config,data_root,split,weight" > "${RAW_CSV}"

run_eval() {
  local dataset="$1"
  local data_root="$2"
  local split="$3"
  local mode="$4"
  local config="$5"
  local log_path="${RESULT_ROOT}/${dataset}_${mode}.log"

  echo "[eval] dataset=${dataset} mode=${mode} split=${split} root=${data_root}"
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_enc2d_stress.py \
    --config "${config}" \
    --weight "${OFFICIAL_WEIGHT}" \
    --data-root "${data_root}" \
    --split "${split}" \
    --stress ${STRESS_LIST} \
    --max-batches "${MAX_BATCHES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-worker "${NUM_WORKER}" | tee "${log_path}"

  "${PYTHON_BIN}" - "${log_path}" "${RAW_CSV}" "${dataset}" "${mode}" "${config}" "${data_root}" "${split}" "${OFFICIAL_WEIGHT}" <<'PY'
import csv
import sys
from pathlib import Path

log_path, csv_path, dataset, mode, config, data_root, split, weight = sys.argv[1:]
lines = log_path and Path(log_path).read_text(errors="ignore").splitlines()
rows = []
for line in lines:
    parts = [item.strip() for item in line.split(",")]
    if len(parts) != 3 or parts[0] in {"stress", ""}:
        continue
    try:
        batches = int(parts[1])
        mean = float(parts[2])
    except ValueError:
        continue
    rows.append(
        {
            "dataset": dataset,
            "mode": mode,
            "stress": parts[0],
            "batches": batches,
            "enc2d_loss_mean": f"{mean:.6f}",
            "config": config,
            "data_root": data_root,
            "split": split,
            "weight": weight,
        }
    )
with open(csv_path, "a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "dataset",
            "mode",
            "stress",
            "batches",
            "enc2d_loss_mean",
            "config",
            "data_root",
            "split",
            "weight",
        ],
    )
    writer.writerows(rows)
PY
}

for dataset_spec in "${DATASETS[@]}"; do
  IFS='|' read -r dataset_name data_root split <<< "${dataset_spec}"
  for mode_spec in "${MODES[@]}"; do
    IFS='|' read -r mode config <<< "${mode_spec}"
    run_eval "${dataset_name}" "${data_root}" "${split}" "${mode}" "${config}"
  done
done

"${PYTHON_BIN}" - "${RAW_CSV}" "${RESULT_CSV}" "${RESULT_MD}" <<'PY'
import csv
import sys
from pathlib import Path

raw_path, out_csv_path, out_md_path = map(Path, sys.argv[1:])
rows = list(csv.DictReader(raw_path.open()))
baselines = {
    (row["dataset"], row["stress"]): float(row["enc2d_loss_mean"])
    for row in rows
    if row["mode"] == "baseline"
}
out_rows = []
for row in rows:
    value = float(row["enc2d_loss_mean"])
    base = baselines.get((row["dataset"], row["stress"]))
    delta = value - base if base is not None else None
    out = dict(row)
    out["delta_vs_baseline"] = "" if delta is None else f"{delta:.6f}"
    out_rows.append(out)

fieldnames = list(out_rows[0].keys()) if out_rows else []
with out_csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

lines = [
    "# Official Concerto Causal Battery",
    "",
    f"- raw logs: `{raw_path}`",
    "",
    "| dataset | mode | stress | batches | enc2d loss mean | delta vs baseline |",
    "| --- | --- | --- | ---: | ---: | ---: |",
]
for row in out_rows:
    lines.append(
        f"| {row['dataset']} | {row['mode']} | {row['stress']} | {row['batches']} | "
        f"{row['enc2d_loss_mean']} | {row['delta_vs_baseline']} |"
    )
out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "[done] wrote ${RESULT_CSV}"
echo "[done] wrote ${RESULT_MD}"
