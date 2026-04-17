#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
POSTHOC_STRESS_ROOT=${POSTHOC_STRESS_ROOT:-${POINTCEPT_DATA_ROOT}/runs/posthoc_stress_e025pilot}
BACKBONE_CKPT=${BACKBONE_CKPT:-${POINTCEPT_DATA_ROOT}/runs/projres_long/exp/arkit-full-original-long-e025-qf32-continue/model/model_last.pth}
ORIGINAL_LINEAR_CKPT=${ORIGINAL_LINEAR_CKPT:-${POINTCEPT_DATA_ROOT}/runs/projres_long/exp/scannet-proxy-arkit-full-original-long-e025-qf32-continue-lin/model/model_last.pth}
SPLICE_HEIGHT_EDITOR=${SPLICE_HEIGHT_EDITOR:-${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery_e025pilot/original-long-e025-qf32/splice3d_height_g1.0/splice3d_editor.pth}
SPLICE_HEIGHT_LINEAR_CKPT=${SPLICE_HEIGHT_LINEAR_CKPT:-${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery_e025pilot/exp/posthoc-original-long-e025-qf32-splice3d-height-g1.0-lin/model/model_last.pth}
SPLICE_HEIGHT_XYZ_EDITOR=${SPLICE_HEIGHT_XYZ_EDITOR:-${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery_e025pilot/original-long-e025-qf32/splice3d_height_xyz_g1.0/splice3d_editor.pth}
SPLICE_HEIGHT_XYZ_LINEAR_CKPT=${SPLICE_HEIGHT_XYZ_LINEAR_CKPT:-${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery_e025pilot/exp/posthoc-original-long-e025-qf32-splice3d-height_xyz-g1.0-lin/model/model_last.pth}
RECYCLE_EDITOR=${RECYCLE_EDITOR:-${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery_e025pilot/original-long-e025-qf32/recycle_height_xyz_coord9_g1.0_r1.0/residual_recycling_editor.pth}
RECYCLE_LINEAR_CKPT=${RECYCLE_LINEAR_CKPT:-${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery_e025pilot/exp/posthoc-original-long-e025-qf32-recycle_height_xyz_coord9_g1.0_r1.0-lin/model/model_last.pth}
RECYCLE_GEOM=${RECYCLE_GEOM:-coord9}
RECYCLE_SCALE=${RECYCLE_SCALE:-1.0}
RECYCLE_MAX_RANK=${RECYCLE_MAX_RANK:-8}
RECYCLE_COEFF_CLIP=${RECYCLE_COEFF_CLIP:-0.0}
STRESSES=${STRESSES:-"clean local_surface_destroy z_flip xy_swap roll_90_x"}
MAX_BATCHES=${MAX_BATCHES:--1}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKER=${NUM_WORKER:-4}
VOXEL_SIZE=${VOXEL_SIZE:-0.2}
RUN_PARALLEL=${RUN_PARALLEL:-1}
RUN_ORIGINAL=${RUN_ORIGINAL:-1}
RUN_SPLICE_HEIGHT=${RUN_SPLICE_HEIGHT:-1}
RUN_SPLICE_HEIGHT_XYZ=${RUN_SPLICE_HEIGHT_XYZ:-1}
RUN_RECYCLE=${RUN_RECYCLE:-0}

mkdir -p "${POSTHOC_STRESS_ROOT}"

run_eval() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local weight="$4"
  local editor="${5:-}"
  local out="${POSTHOC_STRESS_ROOT}/${name}_stress.csv"
  echo "[stress-suite] name=${name} gpu=${gpu} config=${config} weight=${weight} editor=${editor} out=${out}"
  if [ -n "${editor}" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
    POSTHOC_BACKBONE_CKPT="${BACKBONE_CKPT}" \
    POSTHOC_EDITOR_CKPT="${editor}" \
    POSTHOC_RECYCLE_GEOM_SPEC="${RECYCLE_GEOM}" \
    POSTHOC_RECYCLE_SCALE="${RECYCLE_SCALE}" \
    POSTHOC_RECYCLE_MAX_RANK="${RECYCLE_MAX_RANK}" \
    POSTHOC_RECYCLE_COEFF_CLIP="${RECYCLE_COEFF_CLIP}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/eval_scannet_semseg_stress.py" \
      --repo-root "${REPO_ROOT}" \
      --config "${config}" \
      --weight "${weight}" \
      --output "${out}" \
      --stress ${STRESSES} \
      --voxel-size "${VOXEL_SIZE}" \
      --batch-size "${BATCH_SIZE}" \
      --num-worker "${NUM_WORKER}" \
      --max-batches "${MAX_BATCHES}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/eval_scannet_semseg_stress.py" \
      --repo-root "${REPO_ROOT}" \
      --config "${config}" \
      --weight "${weight}" \
      --output "${out}" \
      --stress ${STRESSES} \
      --voxel-size "${VOXEL_SIZE}" \
      --batch-size "${BATCH_SIZE}" \
      --num-worker "${NUM_WORKER}" \
      --max-batches "${MAX_BATCHES}"
  fi
}

if [ "${RUN_PARALLEL}" = "1" ]; then
  pids=()
  if [ "${RUN_ORIGINAL}" = "1" ]; then
    run_eval original 0 semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly "${ORIGINAL_LINEAR_CKPT}" "" &
    pids+=("$!")
  fi
  if [ "${RUN_SPLICE_HEIGHT}" = "1" ]; then
    run_eval splice3d_height 1 semseg-ptv3-base-v1m1-0a-scannet-lin-splice3d-frozen "${SPLICE_HEIGHT_LINEAR_CKPT}" "${SPLICE_HEIGHT_EDITOR}" &
    pids+=("$!")
  fi
  if [ "${RUN_SPLICE_HEIGHT_XYZ}" = "1" ]; then
    run_eval splice3d_height_xyz 2 semseg-ptv3-base-v1m1-0a-scannet-lin-splice3d-frozen "${SPLICE_HEIGHT_XYZ_LINEAR_CKPT}" "${SPLICE_HEIGHT_XYZ_EDITOR}" &
    pids+=("$!")
  fi
  if [ "${RUN_RECYCLE}" = "1" ]; then
    run_eval recycle 3 semseg-ptv3-base-v1m1-0a-scannet-lin-recycle-frozen "${RECYCLE_LINEAR_CKPT}" "${RECYCLE_EDITOR}" &
    pids+=("$!")
  fi
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  if [ "${status}" -ne 0 ]; then
    echo "[stress-suite] at least one eval failed" >&2
    exit "${status}"
  fi
else
  [ "${RUN_ORIGINAL}" = "1" ] && run_eval original 0 semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly "${ORIGINAL_LINEAR_CKPT}" ""
  [ "${RUN_SPLICE_HEIGHT}" = "1" ] && run_eval splice3d_height 0 semseg-ptv3-base-v1m1-0a-scannet-lin-splice3d-frozen "${SPLICE_HEIGHT_LINEAR_CKPT}" "${SPLICE_HEIGHT_EDITOR}"
  [ "${RUN_SPLICE_HEIGHT_XYZ}" = "1" ] && run_eval splice3d_height_xyz 0 semseg-ptv3-base-v1m1-0a-scannet-lin-splice3d-frozen "${SPLICE_HEIGHT_XYZ_LINEAR_CKPT}" "${SPLICE_HEIGHT_XYZ_EDITOR}"
  [ "${RUN_RECYCLE}" = "1" ] && run_eval recycle 0 semseg-ptv3-base-v1m1-0a-scannet-lin-recycle-frozen "${RECYCLE_LINEAR_CKPT}" "${RECYCLE_EDITOR}"
fi

"${PYTHON_BIN}" - <<PY
import csv
from pathlib import Path
root = Path("${POSTHOC_STRESS_ROOT}")
rows = []
for path in sorted(root.glob("*_stress.csv")):
    method = path.name[:-len("_stress.csv")]
    with path.open() as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["method"] = method
            rows.append(row)
out = root / "posthoc_stress_suite.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "stress", "batches", "mIoU", "mAcc", "allAcc", "loss"])
    writer.writeheader()
    writer.writerows(rows)
print(f"[stress-suite] summary={out}")
PY
