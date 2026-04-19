#!/usr/bin/env bash
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/data/logs/abciq/

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${WORKDIR}"

mkdir -p data/logs/abciq

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

LINEAR_CONFIG="${LINEAR_CONFIG:?LINEAR_CONFIG is required}"
LINEAR_WEIGHT="${LINEAR_WEIGHT:?LINEAR_WEIGHT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

DATA_ROOT="${DATA_ROOT:-data/scannet}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-256}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-128}"
MAX_PER_CLASS="${MAX_PER_CLASS:-60000}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-100}"
LOGREG_STEPS="${LOGREG_STEPS:-600}"
NUM_WORKER="${NUM_WORKER:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CLASS_PAIRS="${CLASS_PAIRS:-picture:wall,counter:cabinet,desk:wall,desk:table,sink:cabinet,door:wall,shower curtain:wall}"

echo "=== ScanNet point stagewise trace ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "linear_config=${LINEAR_CONFIG}"
echo "linear_weight=${LINEAR_WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_scannet_point_stagewise_trace.py \
  --linear-config "${LINEAR_CONFIG}" \
  --linear-weight "${LINEAR_WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --bootstrap-iters "${BOOTSTRAP_ITERS}" \
  --logreg-steps "${LOGREG_STEPS}" \
  --num-worker "${NUM_WORKER}" \
  --batch-size "${BATCH_SIZE}" \
  --class-pairs "${CLASS_PAIRS}"

echo "[done] ScanNet point stagewise trace"
