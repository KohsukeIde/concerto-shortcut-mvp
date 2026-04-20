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
MAX_PROTO_POINTS_PER_CLASS="${MAX_PROTO_POINTS_PER_CLASS:-20000}"
VAL_CHUNK_SIZE="${VAL_CHUNK_SIZE:-2048}"
KNN_KS="${KNN_KS:-5,10,20,50}"
KNN_LAMBDAS="${KNN_LAMBDAS:-0.05,0.1,0.2,0.4}"
KNN_TAUS="${KNN_TAUS:-0.05,0.1}"
RUN_KNN="${RUN_KNN:-1}"
PROTOTYPE_COUNTS="${PROTOTYPE_COUNTS:-1,4,8}"
PROTOTYPE_LAMBDAS="${PROTOTYPE_LAMBDAS:-0.05,0.1,0.2,0.4}"
PROTOTYPE_TAUS="${PROTOTYPE_TAUS:-0.05,0.1,0.2}"
RUN_PROTOTYPE="${RUN_PROTOTYPE:-1}"
KMEANS_ITERS="${KMEANS_ITERS:-10}"
NUM_WORKER="${NUM_WORKER:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_retrieval_prototype_readout}"

echo "=== Retrieval / prototype readout ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

if [[ "${RUN_KNN}" != "1" ]]; then
  KNN_KS=""
  KNN_LAMBDAS=""
  KNN_TAUS=""
fi
if [[ "${RUN_PROTOTYPE}" != "1" ]]; then
  PROTOTYPE_COUNTS=""
  PROTOTYPE_LAMBDAS=""
  PROTOTYPE_TAUS=""
fi

python tools/concerto_projection_shortcut/eval_retrieval_prototype_readout.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-bank-points "${MAX_BANK_POINTS}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --max-proto-points-per-class "${MAX_PROTO_POINTS_PER_CLASS}" \
  --val-chunk-size "${VAL_CHUNK_SIZE}" \
  --knn-ks "${KNN_KS}" \
  --knn-lambdas "${KNN_LAMBDAS}" \
  --knn-taus "${KNN_TAUS}" \
  --prototype-counts "${PROTOTYPE_COUNTS}" \
  --prototype-lambdas "${PROTOTYPE_LAMBDAS}" \
  --prototype-taus "${PROTOTYPE_TAUS}" \
  --kmeans-iters "${KMEANS_ITERS}" \
  --num-worker "${NUM_WORKER}" \
  --batch-size "${BATCH_SIZE}" \
  --summary-prefix "${SUMMARY_PREFIX}"

echo "[done] retrieval / prototype readout"
