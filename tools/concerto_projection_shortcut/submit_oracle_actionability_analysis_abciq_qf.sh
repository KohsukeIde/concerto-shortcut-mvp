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
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-600000}"
MAX_PER_CLASS="${MAX_PER_CLASS:-60000}"
MAX_GEOMETRY_PER_CLASS="${MAX_GEOMETRY_PER_CLASS:-60000}"
PAIR_PROBE_STEPS="${PAIR_PROBE_STEPS:-800}"
BIAS_STEPS="${BIAS_STEPS:-1000}"
NUM_WORKER="${NUM_WORKER:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WEAK_CLASSES="${WEAK_CLASSES:-picture,counter,desk,sink,cabinet,shower curtain,door}"
CLASS_PAIRS="${CLASS_PAIRS:-picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall}"

echo "=== Oracle actionability analysis ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_oracle_actionability_analysis.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --weak-classes "${WEAK_CLASSES}" \
  --class-pairs "${CLASS_PAIRS}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-train-points "${MAX_TRAIN_POINTS}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --max-geometry-per-class "${MAX_GEOMETRY_PER_CLASS}" \
  --pair-probe-steps "${PAIR_PROBE_STEPS}" \
  --bias-steps "${BIAS_STEPS}" \
  --num-worker "${NUM_WORKER}" \
  --batch-size "${BATCH_SIZE}"

echo "[done] oracle actionability analysis"
