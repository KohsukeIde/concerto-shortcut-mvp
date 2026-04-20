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
WEIGHT="${WEIGHT:-data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth}"
METHOD_NAME="${METHOD_NAME:-concerto_decoder_origin}"
DATA_ROOT="${DATA_ROOT:-data/scannet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/runs/scannet_decoder_probe_origin/masking_battery}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
MASK_PRESET="${MASK_PRESET:-}"
if [[ "${MASK_PRESET}" == "full" ]]; then
  RANDOM_KEEP_RATIOS="${RANDOM_KEEP_RATIOS:-0.5,0.3,0.2,0.1}"
  STRUCTURED_KEEP_RATIOS="${STRUCTURED_KEEP_RATIOS:-0.5,0.2}"
  FEATURE_ZERO_RATIOS="${FEATURE_ZERO_RATIOS:-1.0}"
else
  RANDOM_KEEP_RATIOS="${RANDOM_KEEP_RATIOS:-0.2}"
  STRUCTURED_KEEP_RATIOS="${STRUCTURED_KEEP_RATIOS:-}"
  FEATURE_ZERO_RATIOS="${FEATURE_ZERO_RATIOS:-}"
fi
STRUCTURED_BLOCK_SIZE="${STRUCTURED_BLOCK_SIZE:-64}"
REPEATS="${REPEATS:-1}"
NUM_WORKER="${NUM_WORKER:-8}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_masking_battery}"

echo "=== Masking battery ==="
date +"date=%Y-%m-%dT%H:%M:%S%z"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "method_name=${METHOD_NAME}"
echo "output_dir=${OUTPUT_DIR}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "random_keep_ratios=${RANDOM_KEEP_RATIOS}"
echo "structured_keep_ratios=${STRUCTURED_KEEP_RATIOS}"
echo "feature_zero_ratios=${FEATURE_ZERO_RATIOS}"
echo "repeats=${REPEATS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_masking_battery.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --method-name "${METHOD_NAME}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --random-keep-ratios "${RANDOM_KEEP_RATIOS}" \
  --structured-keep-ratios "${STRUCTURED_KEEP_RATIOS}" \
  --feature-zero-ratios "${FEATURE_ZERO_RATIOS}" \
  --structured-block-size "${STRUCTURED_BLOCK_SIZE}" \
  --repeats "${REPEATS}" \
  --num-worker "${NUM_WORKER}" \
  --summary-prefix "${SUMMARY_PREFIX}"

echo "[done] masking battery"
