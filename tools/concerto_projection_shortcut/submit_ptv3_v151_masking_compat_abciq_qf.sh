#!/bin/bash
#PBS -W group_list=qgah50055
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/data/logs/abciq/

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${WORKDIR}"

mkdir -p data/logs/abciq

module purge
module load python/3.11 cuda/12.6

VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/data/venv/pointcept-concerto-py311-cu124/bin/activate}"
source "${VENV_ACTIVATE}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OFFICIAL_ROOT="${OFFICIAL_ROOT:-data/tmp/Pointcept-v1.5.1}"
CONFIG="${CONFIG:-configs/scannet/semseg-pt-v3m1-0-base.py}"
WEIGHT="${WEIGHT:-data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth}"
METHOD_NAME="${METHOD_NAME:-ptv3_supervised_v151_compat}"
DATA_ROOT="${DATA_ROOT:-data/scannet}"
SPLIT="${SPLIT:-val}"
SEGMENT_KEY="${SEGMENT_KEY:-segment20}"
OUTPUT_DIR="${OUTPUT_DIR:-data/runs/masking_ranking/ptv3_v151_compat}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
MASK_PRESET="${MASK_PRESET:-}"
if [[ "${MASK_PRESET}" == "full" ]]; then
  RANDOM_KEEP_RATIOS="${RANDOM_KEEP_RATIOS-0.5,0.3,0.2,0.1}"
  CLASSWISE_KEEP_RATIOS="${CLASSWISE_KEEP_RATIOS-0.2}"
  STRUCTURED_KEEP_RATIOS="${STRUCTURED_KEEP_RATIOS-0.5,0.2}"
  FEATURE_ZERO_RATIOS="${FEATURE_ZERO_RATIOS-1.0}"
else
  RANDOM_KEEP_RATIOS="${RANDOM_KEEP_RATIOS-0.2}"
  CLASSWISE_KEEP_RATIOS="${CLASSWISE_KEEP_RATIOS-}"
  STRUCTURED_KEEP_RATIOS="${STRUCTURED_KEEP_RATIOS-0.2}"
  FEATURE_ZERO_RATIOS="${FEATURE_ZERO_RATIOS-1.0}"
fi
STRUCTURED_BLOCK_SIZE="${STRUCTURED_BLOCK_SIZE:-64}"
REPEATS="${REPEATS:-1}"
NUM_CLASSES="${NUM_CLASSES:-20}"
CLASS_NAMES="${CLASS_NAMES:-}"
FOCUS_CLASS="${FOCUS_CLASS:-picture}"
CONFUSION_CLASS="${CONFUSION_CLASS:-wall}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_ptv3_v151_masking_compat}"

echo "=== PTv3 v1.5.1 masking compatibility eval ==="
date +"date=%Y-%m-%dT%H:%M:%S%z"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "official_root=${OFFICIAL_ROOT}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "method_name=${METHOD_NAME}"
echo "data_root=${DATA_ROOT}"
echo "split=${SPLIT}"
echo "segment_key=${SEGMENT_KEY}"
echo "output_dir=${OUTPUT_DIR}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_ptv3_v151_masking_compat.py \
  --official-root "${OFFICIAL_ROOT}" \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --method-name "${METHOD_NAME}" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --segment-key "${SEGMENT_KEY}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --random-keep-ratios "${RANDOM_KEEP_RATIOS}" \
  --classwise-keep-ratios "${CLASSWISE_KEEP_RATIOS}" \
  --structured-keep-ratios "${STRUCTURED_KEEP_RATIOS}" \
  --feature-zero-ratios "${FEATURE_ZERO_RATIOS}" \
  --structured-block-size "${STRUCTURED_BLOCK_SIZE}" \
  --repeats "${REPEATS}" \
  --num-classes "${NUM_CLASSES}" \
  --class-names "${CLASS_NAMES}" \
  --focus-class "${FOCUS_CLASS}" \
  --confusion-class "${CONFUSION_CLASS}" \
  --summary-prefix "${SUMMARY_PREFIX}"

echo "[done] ptv3 v1.5.1 masking compatibility eval"
