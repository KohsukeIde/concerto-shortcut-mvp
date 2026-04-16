#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

BACKBONE_CKPT=${BACKBONE_CKPT:?set BACKBONE_CKPT}
BACKBONE_TAG=${BACKBONE_TAG:-}
POSTHOC_SPECS=${POSTHOC_SPECS:-"splice3d:height+xyz splice3d:height hlns:height+xyz"}
POSTHOC_ROOT=${POSTHOC_ROOT:-${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery}
ROWS_PER_SCENE=${ROWS_PER_SCENE:-1024}
MAX_BATCHES_TRAIN=${MAX_BATCHES_TRAIN:--1}
MAX_BATCHES_VAL=${MAX_BATCHES_VAL:--1}
GAMMA=${GAMMA:-1.0}
GPUS=${GPUS:-4}
GPU_IDS_CSV=${GPU_IDS_CSV:-0,1,2,3}
LINEAR_TRAIN_LAUNCHER=${LINEAR_TRAIN_LAUNCHER:-torchrun}
RUN_LINEAR=${RUN_LINEAR:-1}

echo "=== Posthoc nuisance surgery suite ==="
echo "date=$(date -Is)"
echo "repo_root=${REPO_ROOT}"
echo "backbone_ckpt=${BACKBONE_CKPT}"
echo "backbone_tag=${BACKBONE_TAG}"
echo "posthoc_specs=${POSTHOC_SPECS}"
echo "posthoc_root=${POSTHOC_ROOT}"
echo "rows_per_scene=${ROWS_PER_SCENE}"
echo "max_batches_train=${MAX_BATCHES_TRAIN}"
echo "max_batches_val=${MAX_BATCHES_VAL}"
echo "gamma=${GAMMA}"
echo "gpus=${GPUS}"
echo "gpu_ids_csv=${GPU_IDS_CSV}"
echo "linear_train_launcher=${LINEAR_TRAIN_LAUNCHER}"
echo "run_linear=${RUN_LINEAR}"

for spec in ${POSTHOC_SPECS}; do
  method="${spec%%:*}"
  nuisance="${spec#*:}"
  echo "=== Run posthoc spec: method=${method} nuisance=${nuisance} ==="
  METHOD="${method}" \
  NUISANCE="${nuisance}" \
  BACKBONE_CKPT="${BACKBONE_CKPT}" \
  BACKBONE_TAG="${BACKBONE_TAG}" \
  POSTHOC_ROOT="${POSTHOC_ROOT}" \
  ROWS_PER_SCENE="${ROWS_PER_SCENE}" \
  MAX_BATCHES_TRAIN="${MAX_BATCHES_TRAIN}" \
  MAX_BATCHES_VAL="${MAX_BATCHES_VAL}" \
  GAMMA="${GAMMA}" \
  GPUS="${GPUS}" \
  GPU_IDS_CSV="${GPU_IDS_CSV}" \
  LINEAR_TRAIN_LAUNCHER="${LINEAR_TRAIN_LAUNCHER}" \
  RUN_LINEAR="${RUN_LINEAR}" \
  bash "${REPO_ROOT}/tools/concerto_projection_shortcut/run_posthoc_surgery_chain.sh"
done

echo "[done] posthoc nuisance surgery suite"
