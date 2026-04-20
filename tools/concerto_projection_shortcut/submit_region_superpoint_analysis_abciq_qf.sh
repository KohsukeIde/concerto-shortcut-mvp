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
DATA_ROOT="${DATA_ROOT:-data/scannet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/runs/scannet_decoder_probe_origin/region_superpoint_analysis}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
REGION_VOXEL_SIZES="${REGION_VOXEL_SIZES:-4,8,16,32}"
NUM_WORKER="${NUM_WORKER:-8}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_region_superpoint_analysis}"

echo "=== Region / superpoint analysis ==="
date +"date=%Y-%m-%dT%H:%M:%S%z"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "region_voxel_sizes=${REGION_VOXEL_SIZES}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_region_superpoint_analysis.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --region-voxel-sizes "${REGION_VOXEL_SIZES}" \
  --num-worker "${NUM_WORKER}" \
  --summary-prefix "${SUMMARY_PREFIX}"

echo "[done] region / superpoint analysis"
