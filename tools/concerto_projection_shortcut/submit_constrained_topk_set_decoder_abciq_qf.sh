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
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-320}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
MAX_RERANK_TRAIN_POINTS="${MAX_RERANK_TRAIN_POINTS:-500000}"
MAX_HELDOUT_POINTS="${MAX_HELDOUT_POINTS:-250000}"
MAX_PER_CLASS="${MAX_PER_CLASS:-60000}"
STEPS="${STEPS:-2500}"
LR="${LR:-0.0003}"
RESIDUAL_L2="${RESIDUAL_L2:-0.005}"
KL_WEIGHT="${KL_WEIGHT:-0.02}"
WEAK_CLASS_WEIGHT="${WEAK_CLASS_WEIGHT:-2.0}"
TRAIN_TOP_K="${TRAIN_TOP_K:-5}"
NUM_WORKER="${NUM_WORKER:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"

echo "=== Constrained top-K set decoder ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/fit_constrained_topk_set_decoder.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-rerank-train-points "${MAX_RERANK_TRAIN_POINTS}" \
  --max-heldout-points "${MAX_HELDOUT_POINTS}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --residual-l2 "${RESIDUAL_L2}" \
  --kl-weight "${KL_WEIGHT}" \
  --weak-class-weight "${WEAK_CLASS_WEIGHT}" \
  --train-top-k "${TRAIN_TOP_K}" \
  --num-worker "${NUM_WORKER}" \
  --batch-size "${BATCH_SIZE}"

echo "[done] constrained top-K set decoder"
