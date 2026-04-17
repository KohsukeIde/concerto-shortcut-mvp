#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
if [ -f "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
fi

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy}
BACKBONE_CKPT=${BACKBONE_CKPT:?set BACKBONE_CKPT}
ROWS_PER_SCENE=${ROWS_PER_SCENE:-1024}
MAX_BATCHES_TRAIN=${MAX_BATCHES_TRAIN:--1}
MAX_BATCHES_VAL=${MAX_BATCHES_VAL:--1}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKER=${NUM_WORKER:-4}
GEOMETRY_KNN=${GEOMETRY_KNN:-32}
GEOMETRY_QUERY_CHUNK=${GEOMETRY_QUERY_CHUNK:-512}
GEOMETRY_KEY_CHUNK=${GEOMETRY_KEY_CHUNK:-32768}
GEOMETRY_UP_AXIS=${GEOMETRY_UP_AXIS:-z}
RIDGE=${RIDGE:-1e-2}
RESIDUAL_RIDGE=${RESIDUAL_RIDGE:-1e-2}
PASS_THRESHOLD=${PASS_THRESHOLD:-0.003}
POINTCEPT_DATA_ROOT=${POINTCEPT_DATA_ROOT:-${REPO_ROOT}/data}
STEP1_ROOT=${STEP1_ROOT:-${POINTCEPT_DATA_ROOT}/runs/step1_geometry_smoke}

if [ -z "${BACKBONE_TAG:-}" ]; then
  if [ "$(basename "$(dirname "${BACKBONE_CKPT}")")" = "model" ]; then
    BACKBONE_TAG="$(basename "$(dirname "$(dirname "${BACKBONE_CKPT}")")")"
  else
    BACKBONE_TAG="$(basename "${BACKBONE_CKPT}" .pth)"
  fi
fi
BACKBONE_TAG="$(printf '%s' "${BACKBONE_TAG}" | tr -c 'A-Za-z0-9_.-' '_')"
CACHE_ROOT=${CACHE_ROOT:-${STEP1_ROOT}/${BACKBONE_TAG}/cache/${CONFIG}_r${ROWS_PER_SCENE}_mt${MAX_BATCHES_TRAIN}_mv${MAX_BATCHES_VAL}_geomk${GEOMETRY_KNN}}
OUTDIR=${OUTDIR:-${STEP1_ROOT}/${BACKBONE_TAG}/geom_smoke}
TRAIN_CACHE="${CACHE_ROOT}/train_features.pt"
VAL_CACHE="${CACHE_ROOT}/val_features.pt"

mkdir -p "${CACHE_ROOT}" "${OUTDIR}"

echo "[step1] repo_root=${REPO_ROOT}"
echo "[step1] backbone_ckpt=${BACKBONE_CKPT}"
echo "[step1] cache_root=${CACHE_ROOT}"
echo "[step1] outdir=${OUTDIR}"

needs_extract=1
if [ -f "${TRAIN_CACHE}" ] && [ -f "${VAL_CACHE}" ]; then
  if "${PYTHON_BIN}" - <<'PY' "${TRAIN_CACHE}" "${VAL_CACHE}"
import sys
import torch
for path in sys.argv[1:]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "geom_local9" not in payload:
        raise SystemExit(1)
raise SystemExit(0)
PY
  then
    needs_extract=0
  fi
fi

if [ "${needs_extract}" = "1" ]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/extract_frozen_backbone_features_step1_geom.py" \
    --repo-root "${REPO_ROOT}" \
    --config "${CONFIG}" \
    --weight "${BACKBONE_CKPT}" \
    --output-root "${CACHE_ROOT}" \
    --rows-per-scene "${ROWS_PER_SCENE}" \
    --max-batches-train "${MAX_BATCHES_TRAIN}" \
    --max-batches-val "${MAX_BATCHES_VAL}" \
    --batch-size "${BATCH_SIZE}" \
    --num-worker "${NUM_WORKER}" \
    --geometry-knn "${GEOMETRY_KNN}" \
    --geometry-query-chunk "${GEOMETRY_QUERY_CHUNK}" \
    --geometry-key-chunk "${GEOMETRY_KEY_CHUNK}" \
    --geometry-up-axis "${GEOMETRY_UP_AXIS}"
else
  echo "[step1] reuse geometry cache: ${CACHE_ROOT}"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/fit_geometry_step1_smoke.py" \
  --train-cache "${TRAIN_CACHE}" \
  --val-cache "${VAL_CACHE}" \
  --output-root "${OUTDIR}" \
  --geometry-key geom_local9 \
  --ridge "${RIDGE}" \
  --residual-ridge "${RESIDUAL_RIDGE}" \
  --pass-threshold "${PASS_THRESHOLD}"

echo "[step1] wrote: ${OUTDIR}/step1_geometry_smoke.md"
