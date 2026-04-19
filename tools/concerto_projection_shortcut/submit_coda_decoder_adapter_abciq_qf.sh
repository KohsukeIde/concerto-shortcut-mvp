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
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-420}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
MAX_ADAPTER_TRAIN_POINTS="${MAX_ADAPTER_TRAIN_POINTS:-1200000}"
MAX_HELDOUT_POINTS="${MAX_HELDOUT_POINTS:-1200000}"
MAX_PER_CLASS="${MAX_PER_CLASS:-60000}"
STEPS="${STEPS:-3000}"
LR="${LR:-0.0003}"
WEAK_CLASS_WEIGHT="${WEAK_CLASS_WEIGHT:-3.0}"
PAIR_WEIGHT="${PAIR_WEIGHT:-0.5}"
KL_WEIGHT="${KL_WEIGHT:-0.05}"
RESIDUAL_L2="${RESIDUAL_L2:-0.01}"
TRAIN_TAU="${TRAIN_TAU:-1.0}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
NUM_WORKER="${NUM_WORKER:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_LAMBDAS="${EVAL_LAMBDAS:-0.1,0.2,0.5,1.0}"
EVAL_TAUS="${EVAL_TAUS:-0.1,0.2,0.5,1.0}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
EVAL_ALL_VAL="${EVAL_ALL_VAL:-0}"

echo "=== CoDA decoder adapter ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "adapter_path=${ADAPTER_PATH}"
echo "eval_all_val=${EVAL_ALL_VAL}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

CMD=(python tools/concerto_projection_shortcut/fit_coda_decoder_adapter.py
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-adapter-train-points "${MAX_ADAPTER_TRAIN_POINTS}" \
  --max-heldout-points "${MAX_HELDOUT_POINTS}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --weak-class-weight "${WEAK_CLASS_WEIGHT}" \
  --pair-weight "${PAIR_WEIGHT}" \
  --kl-weight "${KL_WEIGHT}" \
  --residual-l2 "${RESIDUAL_L2}" \
  --train-tau "${TRAIN_TAU}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --eval-lambdas "${EVAL_LAMBDAS}" \
  --eval-taus "${EVAL_TAUS}" \
  --num-worker "${NUM_WORKER}" \
  --batch-size "${BATCH_SIZE}")

if [[ -n "${ADAPTER_PATH}" ]]; then
  CMD+=(--adapter-path "${ADAPTER_PATH}")
fi
if [[ "${EVAL_ALL_VAL}" == "1" ]]; then
  CMD+=(--eval-all-val)
fi

"${CMD[@]}"

echo "[done] CoDA decoder adapter"
