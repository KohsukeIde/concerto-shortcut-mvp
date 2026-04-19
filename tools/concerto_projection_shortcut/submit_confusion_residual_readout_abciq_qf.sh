#!/usr/bin/env bash
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/data/logs/abciq/

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${WORKDIR}"

module load python/3.11/3.11.14
module load cuda/12.6/12.6.2

VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/data/venv/pointcept-concerto-py311-cu124/bin/activate}"
source "${VENV_ACTIVATE}"

export HF_HOME="${HF_HOME:-${WORKDIR}/data/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${WORKDIR}/data/hf-home/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-${WORKDIR}/data/hf-home/xet}"
export TORCH_HOME="${TORCH_HOME:-${WORKDIR}/data/torch-home}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS_CSV:-0}"

CONFIG="${CONFIG:?CONFIG is required}"
WEIGHT="${WEIGHT:?WEIGHT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
DATA_ROOT="${DATA_ROOT:-data/scannet}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-256}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
MAX_PER_CLASS="${MAX_PER_CLASS:-60000}"
EXPERT_STEPS="${EXPERT_STEPS:-800}"
EXPERT_LR="${EXPERT_LR:-0.05}"
LAMBDAS="${LAMBDAS:-0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0}"
GATES="${GATES:-none,top1_pair,top2_any,top2_both,uncertain_top2_any}"
UNCERTAINTY_MARGIN="${UNCERTAINTY_MARGIN:-1.0}"
NUM_WORKER="${NUM_WORKER:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CLASS_PAIRS="${CLASS_PAIRS:-picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall}"

echo "=== Confusion residual readout ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/fit_confusion_residual_readout.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --class-pairs "${CLASS_PAIRS}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --expert-steps "${EXPERT_STEPS}" \
  --expert-lr "${EXPERT_LR}" \
  --lambdas "${LAMBDAS}" \
  --gates "${GATES}" \
  --uncertainty-margin "${UNCERTAINTY_MARGIN}" \
  --num-worker "${NUM_WORKER}" \
  --batch-size "${BATCH_SIZE}"

echo "[done] confusion residual readout"
