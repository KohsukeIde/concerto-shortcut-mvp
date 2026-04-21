#!/usr/bin/env bash
#PBS -W group_list=qgah50055
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

OFFICIAL_ROOT="${OFFICIAL_ROOT:-data/tmp/Pointcept-v1.5.1}"
CONFIG="${CONFIG:-configs/scannet/semseg-pt-v3m1-0-base.py}"
WEIGHT="${WEIGHT:?WEIGHT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_ptv3_v151_point_stagewise_trace}"

DATA_ROOT="${DATA_ROOT:-data/scannet}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
SEGMENT_KEY="${SEGMENT_KEY:-segment20}"
NUM_CLASSES="${NUM_CLASSES:-20}"
MAX_TRAIN_SCENES="${MAX_TRAIN_SCENES:-256}"
MAX_VAL_SCENES="${MAX_VAL_SCENES:-128}"
MAX_PER_CLASS="${MAX_PER_CLASS:-60000}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-100}"
LOGREG_STEPS="${LOGREG_STEPS:-600}"
CLASS_PAIRS="${CLASS_PAIRS:-picture:wall,counter:cabinet,door:wall}"

echo "=== PTv3 v1.5.1 point stagewise trace ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "official_root=${OFFICIAL_ROOT}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "summary_prefix=${SUMMARY_PREFIX}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_ptv3_v151_point_stagewise_trace.py \
  --official-root "${OFFICIAL_ROOT}" \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --train-split "${TRAIN_SPLIT}" \
  --val-split "${VAL_SPLIT}" \
  --segment-key "${SEGMENT_KEY}" \
  --num-classes "${NUM_CLASSES}" \
  --output-dir "${OUTPUT_DIR}" \
  --summary-prefix "${SUMMARY_PREFIX}" \
  --max-train-scenes "${MAX_TRAIN_SCENES}" \
  --max-val-scenes "${MAX_VAL_SCENES}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --bootstrap-iters "${BOOTSTRAP_ITERS}" \
  --logreg-steps "${LOGREG_STEPS}" \
  --class-pairs "${CLASS_PAIRS}"

echo "[done] PTv3 v1.5.1 point stagewise trace"
