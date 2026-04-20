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

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG="${CONFIG:-configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py}"
DATA_ROOT="${DATA_ROOT:-data/scannet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/runs/scannet_masking_baselines/coord_majority}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-256}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
MASK_PRESET="${MASK_PRESET:-full}"
if [[ "${MASK_PRESET}" == "full" ]]; then
  RANDOM_KEEP_RATIOS="${RANDOM_KEEP_RATIOS:-0.5,0.3,0.2,0.1}"
  STRUCTURED_KEEP_RATIOS="${STRUCTURED_KEEP_RATIOS:-0.5,0.2}"
  FEATURE_ZERO_RATIOS="${FEATURE_ZERO_RATIOS:-1.0}"
else
  RANDOM_KEEP_RATIOS="${RANDOM_KEEP_RATIOS:-0.2}"
  STRUCTURED_KEEP_RATIOS="${STRUCTURED_KEEP_RATIOS:-}"
  FEATURE_ZERO_RATIOS="${FEATURE_ZERO_RATIOS:-}"
fi
EPOCHS="${EPOCHS:-20}"
CLASS_BALANCED="${CLASS_BALANCED:-0}"
NUM_WORKER="${NUM_WORKER:-8}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_masking_coord_baselines}"

echo "=== Masking coord/majority baselines ==="
date +"date=%Y-%m-%dT%H:%M:%S%z"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "output_dir=${OUTPUT_DIR}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "random_keep_ratios=${RANDOM_KEEP_RATIOS}"
echo "structured_keep_ratios=${STRUCTURED_KEEP_RATIOS}"
echo "feature_zero_ratios=${FEATURE_ZERO_RATIOS}"
echo "epochs=${EPOCHS}"
echo "class_balanced=${CLASS_BALANCED}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

BALANCED_FLAG=()
if [[ "${CLASS_BALANCED}" == "1" ]]; then
  BALANCED_FLAG=(--class-balanced)
fi

python tools/concerto_projection_shortcut/eval_masking_coord_baselines.py \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --random-keep-ratios "${RANDOM_KEEP_RATIOS}" \
  --structured-keep-ratios "${STRUCTURED_KEEP_RATIOS}" \
  --feature-zero-ratios "${FEATURE_ZERO_RATIOS}" \
  --epochs "${EPOCHS}" \
  --num-worker "${NUM_WORKER}" \
  --summary-prefix "${SUMMARY_PREFIX}" \
  "${BALANCED_FLAG[@]}"

echo "[done] masking coord/majority baselines"
