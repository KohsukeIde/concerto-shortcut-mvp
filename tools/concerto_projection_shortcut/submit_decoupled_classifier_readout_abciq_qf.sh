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
MAX_BANK_POINTS="${MAX_BANK_POINTS:-200000}"
MAX_PER_CLASS="${MAX_PER_CLASS:-10000}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4096}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
TAU_VALUES="${TAU_VALUES:-0,0.25,0.5,0.75,1}"
LOGIT_ADJUST_ALPHAS="${LOGIT_ADJUST_ALPHAS:-0.25,0.5,1}"
MIX_LAMBDAS="${MIX_LAMBDAS:-0.05,0.1,0.2,0.4}"
NUM_WORKER="${NUM_WORKER:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_decoupled_classifier_readout}"
BALANCED_SAMPLER="${BALANCED_SAMPLER:-1}"

echo "=== Decoupled classifier readout ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

sampler_arg=()
if [[ "${BALANCED_SAMPLER}" != "1" ]]; then
  sampler_arg+=(--no-balanced-sampler)
fi

python tools/concerto_projection_shortcut/fit_decoupled_classifier_readout.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-bank-points "${MAX_BANK_POINTS}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --train-steps "${TRAIN_STEPS}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --tau-values "${TAU_VALUES}" \
  --logit-adjust-alphas "${LOGIT_ADJUST_ALPHAS}" \
  --mix-lambdas "${MIX_LAMBDAS}" \
  --num-worker "${NUM_WORKER}" \
  --batch-size "${BATCH_SIZE}" \
  --summary-prefix "${SUMMARY_PREFIX}" \
  "${sampler_arg[@]}"

echo "[done] decoupled classifier readout"
