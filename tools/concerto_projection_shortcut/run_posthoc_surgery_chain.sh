#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end helper for frozen-feature post-hoc editors.
# Usage example:
#   BACKBONE_CKPT=/path/to/model_last.pth \
#   REPO_ROOT=/path/to/concerto-shortcut-mvp \
#   METHOD=splice3d \
#   bash tools/concerto_projection_shortcut/run_posthoc_surgery_chain.sh

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

CONFIG=${CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy}
BACKBONE_CKPT=${BACKBONE_CKPT:?set BACKBONE_CKPT}
METHOD=${METHOD:-splice3d}
NUISANCE=${NUISANCE:-height+xyz}
GAMMA=${GAMMA:-1.0}
RIDGE=${RIDGE:-1e-2}
CLASSIFIER_RIDGE=${CLASSIFIER_RIDGE:-1e-2}
RECYCLE_GEOM=${RECYCLE_GEOM:-coord9}
RECYCLE_SCALE=${RECYCLE_SCALE:-1.0}
RECYCLE_RIDGE=${RECYCLE_RIDGE:-1e-2}
RECYCLE_COEFF_RIDGE=${RECYCLE_COEFF_RIDGE:-1e-2}
RECYCLE_MAX_RANK=${RECYCLE_MAX_RANK:-8}
RECYCLE_COEFF_CLIP=${RECYCLE_COEFF_CLIP:-0.0}
POSTHOC_ROOT=${POSTHOC_ROOT:-${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery}
if [ -z "${BACKBONE_TAG:-}" ]; then
  if [ "$(basename "$(dirname "${BACKBONE_CKPT}")")" = "model" ]; then
    BACKBONE_TAG="$(basename "$(dirname "$(dirname "${BACKBONE_CKPT}")")")"
  else
    BACKBONE_TAG="$(basename "${BACKBONE_CKPT}" .pth)"
  fi
fi
BACKBONE_TAG="$(printf '%s' "${BACKBONE_TAG}" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_NUISANCE="$(printf '%s' "${NUISANCE}" | tr '+/' '__')"
SAFE_RECYCLE_GEOM="$(printf '%s' "${RECYCLE_GEOM}" | tr '+/' '__')"
if [[ "${METHOD}" == "recycle" || "${METHOD}" == "residual_recycling" ]]; then
  DEFAULT_POSTHOC_RUN="${METHOD}_${SAFE_NUISANCE}_${SAFE_RECYCLE_GEOM}_g${GAMMA}_r${RECYCLE_SCALE}"
else
  DEFAULT_POSTHOC_RUN="${METHOD}_${SAFE_NUISANCE}_g${GAMMA}"
fi
OUTDIR=${OUTDIR:-${POSTHOC_ROOT}/${BACKBONE_TAG}/${DEFAULT_POSTHOC_RUN}}
ROWS_PER_SCENE=${ROWS_PER_SCENE:-1024}
MAX_BATCHES_TRAIN=${MAX_BATCHES_TRAIN:--1}
MAX_BATCHES_VAL=${MAX_BATCHES_VAL:--1}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKER=${NUM_WORKER:-4}
CACHE_ROOT=${CACHE_ROOT:-${POSTHOC_ROOT}/${BACKBONE_TAG}/cache/${CONFIG}_r${ROWS_PER_SCENE}_mt${MAX_BATCHES_TRAIN}_mv${MAX_BATCHES_VAL}}
DATASET_NAME=${DATASET_NAME:-concerto}
EXP_MIRROR_ROOT=${EXP_MIRROR_ROOT:-${POSTHOC_ROOT}/exp}
LINEAR_CONFIG=${LINEAR_CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-${METHOD}-frozen}
LINEAR_EXP=${LINEAR_EXP:-posthoc-${BACKBONE_TAG}-${DEFAULT_POSTHOC_RUN}-lin}
RUN_LINEAR=${RUN_LINEAR:-1}
GPUS=${GPUS:-4}
GPU_IDS_CSV=${GPU_IDS_CSV:-0,1,2,3}
LINEAR_TRAIN_LAUNCHER=${LINEAR_TRAIN_LAUNCHER:-torchrun}
PYTHON_BIN=${PYTHON_BIN:-python}

ensure_exp_link() {
  local exp_name="$1"
  local link_path="${REPO_ROOT}/exp/${DATASET_NAME}/${exp_name}"
  local target_path="${EXP_MIRROR_ROOT}/${exp_name}"
  mkdir -p "${REPO_ROOT}/exp/${DATASET_NAME}" "${target_path}"
  if [ -L "${link_path}" ] || [ -e "${link_path}" ]; then
    return 0
  fi
  ln -s "${target_path}" "${link_path}"
}

mkdir -p "${OUTDIR}" "${CACHE_ROOT}"

echo "[posthoc] repo_root=${REPO_ROOT}"
echo "[posthoc] backbone_ckpt=${BACKBONE_CKPT}"
echo "[posthoc] backbone_tag=${BACKBONE_TAG}"
echo "[posthoc] method=${METHOD}"
echo "[posthoc] nuisance=${NUISANCE}"
echo "[posthoc] recycle_geom=${RECYCLE_GEOM}"
echo "[posthoc] recycle_scale=${RECYCLE_SCALE}"
echo "[posthoc] outdir=${OUTDIR}"
echo "[posthoc] cache_root=${CACHE_ROOT}"
echo "[posthoc] rows_per_scene=${ROWS_PER_SCENE}"
echo "[posthoc] max_batches_train=${MAX_BATCHES_TRAIN}"
echo "[posthoc] max_batches_val=${MAX_BATCHES_VAL}"
echo "[posthoc] linear_config=${LINEAR_CONFIG}"
echo "[posthoc] linear_exp=${LINEAR_EXP}"
echo "[posthoc] run_linear=${RUN_LINEAR}"
echo "[posthoc] gpus=${GPUS}"

if [ ! -f "${CACHE_ROOT}/train_features.pt" ] || [ ! -f "${CACHE_ROOT}/val_features.pt" ]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/extract_frozen_backbone_features.py" \
    --repo-root "${REPO_ROOT}" \
    --config "${CONFIG}" \
    --weight "${BACKBONE_CKPT}" \
    --output-root "${CACHE_ROOT}" \
    --rows-per-scene "${ROWS_PER_SCENE}" \
    --max-batches-train "${MAX_BATCHES_TRAIN}" \
    --max-batches-val "${MAX_BATCHES_VAL}" \
    --batch-size "${BATCH_SIZE}" \
    --num-worker "${NUM_WORKER}"
else
  echo "[posthoc] reuse feature cache: ${CACHE_ROOT}"
fi

if [[ "${METHOD}" == "splice3d" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/fit_splice3d_frozen.py" \
    --train-cache "${CACHE_ROOT}/train_features.pt" \
    --val-cache "${CACHE_ROOT}/val_features.pt" \
    --output "${OUTDIR}/splice3d_editor.pth" \
    --nuisance "${NUISANCE}" \
    --gamma "${GAMMA}" \
    --ridge "${RIDGE}" \
    --classifier-ridge "${CLASSIFIER_RIDGE}"
  export POSTHOC_EDITOR_CKPT="${OUTDIR}/splice3d_editor.pth"
elif [[ "${METHOD}" == "recycle" || "${METHOD}" == "residual_recycling" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/fit_residual_recycling_frozen.py" \
    --train-cache "${CACHE_ROOT}/train_features.pt" \
    --val-cache "${CACHE_ROOT}/val_features.pt" \
    --output "${OUTDIR}/residual_recycling_editor.pth" \
    --nuisance "${NUISANCE}" \
    --geometry "${RECYCLE_GEOM}" \
    --gamma "${GAMMA}" \
    --recycle-scale "${RECYCLE_SCALE}" \
    --ridge "${RIDGE}" \
    --classifier-ridge "${CLASSIFIER_RIDGE}" \
    --recycle-ridge "${RECYCLE_RIDGE}" \
    --coeff-ridge "${RECYCLE_COEFF_RIDGE}" \
    --max-rank "${RECYCLE_MAX_RANK}"
  export POSTHOC_EDITOR_CKPT="${OUTDIR}/residual_recycling_editor.pth"
  export POSTHOC_RECYCLE_GEOM_SPEC="${RECYCLE_GEOM}"
  export POSTHOC_RECYCLE_MAX_RANK="${RECYCLE_MAX_RANK}"
  export POSTHOC_RECYCLE_SCALE="${RECYCLE_SCALE}"
  export POSTHOC_RECYCLE_COEFF_CLIP="${RECYCLE_COEFF_CLIP}"
else
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/fit_hlns_frozen.py" \
    --train-cache "${CACHE_ROOT}/train_features.pt" \
    --val-cache "${CACHE_ROOT}/val_features.pt" \
    --output "${OUTDIR}/hlns_editor.pth" \
    --nuisance "${NUISANCE}" \
    --gamma "${GAMMA}" \
    --ridge "${RIDGE}" \
    --classifier-ridge "${CLASSIFIER_RIDGE}" \
    --num-groups "${POSTHOC_HLNS_GROUPS:-16}" \
    --topk "${POSTHOC_HLNS_TOPK:-4}"
  export POSTHOC_EDITOR_CKPT="${OUTDIR}/hlns_editor.pth"
fi
export POSTHOC_BACKBONE_CKPT="${BACKBONE_CKPT}"

if [ "${RUN_LINEAR}" != "1" ]; then
  echo "[posthoc] skip linear train: RUN_LINEAR=${RUN_LINEAR}"
  exit 0
fi

ensure_exp_link "${LINEAR_EXP}"
if [ -f "${REPO_ROOT}/exp/${DATASET_NAME}/${LINEAR_EXP}/model/model_last.pth" ]; then
  echo "[posthoc] skip linear train; checkpoint exists: exp/${DATASET_NAME}/${LINEAR_EXP}/model/model_last.pth"
else
  CUDA_VISIBLE_DEVICES="${GPU_IDS_CSV}" \
  POINTCEPT_TRAIN_LAUNCHER="${LINEAR_TRAIN_LAUNCHER}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  bash "${REPO_ROOT}/scripts/train.sh" \
    -p "${PYTHON_BIN}" \
    -d "${DATASET_NAME}" \
    -g "${GPUS}" \
    -c "${LINEAR_CONFIG}" \
    -n "${LINEAR_EXP}" \
    -w None
fi

"${PYTHON_BIN}" "${REPO_ROOT}/tools/concerto_projection_shortcut/summarize_semseg_logs.py" \
  "${REPO_ROOT}/exp/${DATASET_NAME}/${LINEAR_EXP}/train.log" \
  > "${OUTDIR}/${LINEAR_EXP}_summary.csv"
echo "[posthoc] summary=${OUTDIR}/${LINEAR_EXP}_summary.csv"
