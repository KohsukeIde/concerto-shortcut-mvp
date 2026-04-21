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
  FIXED_POINT_COUNTS="${FIXED_POINT_COUNTS-4000}"
  CLASSWISE_KEEP_RATIOS="${CLASSWISE_KEEP_RATIOS-0.2}"
  STRUCTURED_KEEP_RATIOS="${STRUCTURED_KEEP_RATIOS-0.5,0.2}"
  MASKED_MODEL_KEEP_RATIOS="${MASKED_MODEL_KEEP_RATIOS-0.2}"
  FEATURE_ZERO_RATIOS="${FEATURE_ZERO_RATIOS-1.0}"
else
  RANDOM_KEEP_RATIOS="${RANDOM_KEEP_RATIOS-0.2}"
  FIXED_POINT_COUNTS="${FIXED_POINT_COUNTS-}"
  CLASSWISE_KEEP_RATIOS="${CLASSWISE_KEEP_RATIOS-}"
  STRUCTURED_KEEP_RATIOS="${STRUCTURED_KEEP_RATIOS-0.2}"
  MASKED_MODEL_KEEP_RATIOS="${MASKED_MODEL_KEEP_RATIOS-}"
  FEATURE_ZERO_RATIOS="${FEATURE_ZERO_RATIOS-1.0}"
fi
STRUCTURED_BLOCK_SIZE="${STRUCTURED_BLOCK_SIZE:-64}"
REPEATS="${REPEATS:-1}"
NUM_CLASSES="${NUM_CLASSES:-20}"
CLASS_NAMES="${CLASS_NAMES:-}"
FOCUS_CLASS="${FOCUS_CLASS:-picture}"
CONFUSION_CLASS="${CONFUSION_CLASS:-wall}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_ptv3_v151_masking_compat}"
FULL_SCENE_SCORING="${FULL_SCENE_SCORING:-0}"
FULL_SCENE_CHUNK_SIZE="${FULL_SCENE_CHUNK_SIZE:-2048}"
DATASET_TAG="${DATASET_TAG:-}"
SAVE_EXAMPLE_SCENES="${SAVE_EXAMPLE_SCENES:-0}"
EXAMPLE_OUTPUT_DIR="${EXAMPLE_OUTPUT_DIR:-data/runs/masking_examples}"
EXAMPLE_MAX_EXPORT_POINTS="${EXAMPLE_MAX_EXPORT_POINTS:-200000}"

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
echo "random_keep_ratios=${RANDOM_KEEP_RATIOS}"
echo "fixed_point_counts=${FIXED_POINT_COUNTS}"
echo "structured_keep_ratios=${STRUCTURED_KEEP_RATIOS}"
echo "masked_model_keep_ratios=${MASKED_MODEL_KEEP_RATIOS}"
echo "feature_zero_ratios=${FEATURE_ZERO_RATIOS}"
echo "full_scene_scoring=${FULL_SCENE_SCORING}"
nvidia-smi -L || true

FULL_SCENE_ARGS=()
if [[ "${FULL_SCENE_SCORING}" == "1" ]]; then
  FULL_SCENE_ARGS=(--full-scene-scoring --full-scene-chunk-size "${FULL_SCENE_CHUNK_SIZE}")
fi

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
  --fixed-point-counts "${FIXED_POINT_COUNTS}" \
  --classwise-keep-ratios "${CLASSWISE_KEEP_RATIOS}" \
  --structured-keep-ratios "${STRUCTURED_KEEP_RATIOS}" \
  --masked-model-keep-ratios "${MASKED_MODEL_KEEP_RATIOS}" \
  --feature-zero-ratios "${FEATURE_ZERO_RATIOS}" \
  --structured-block-size "${STRUCTURED_BLOCK_SIZE}" \
  --repeats "${REPEATS}" \
  --num-classes "${NUM_CLASSES}" \
  --class-names "${CLASS_NAMES}" \
  --focus-class "${FOCUS_CLASS}" \
  --confusion-class "${CONFUSION_CLASS}" \
  --dataset-tag "${DATASET_TAG}" \
  --save-example-scenes "${SAVE_EXAMPLE_SCENES}" \
  --example-output-dir "${EXAMPLE_OUTPUT_DIR}" \
  --example-max-export-points "${EXAMPLE_MAX_EXPORT_POINTS}" \
  --summary-prefix "${SUMMARY_PREFIX}" \
  "${FULL_SCENE_ARGS[@]}"

echo "[done] ptv3 v1.5.1 masking compatibility eval"
