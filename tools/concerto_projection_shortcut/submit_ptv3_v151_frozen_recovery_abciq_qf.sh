#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=06:00:00
#PBS -N ptv3_rec
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/data/logs/abciq/

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${WORKDIR}"

mkdir -p data/logs/abciq

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}"
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}"

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
WEIGHT="${WEIGHT:-data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-tools/concerto_projection_shortcut/results_ptv3_scannet20_frozen_recovery}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_ptv3_scannet20_frozen_recovery}"

DATA_ROOT="${DATA_ROOT:-data/scannet}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
SEGMENT_KEY="${SEGMENT_KEY:-segment20}"
NUM_CLASSES="${NUM_CLASSES:-20}"
MAX_TRAIN_SCENES="${MAX_TRAIN_SCENES:-256}"
MAX_VAL_SCENES="${MAX_VAL_SCENES:--1}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-600000}"
MAX_PER_CLASS="${MAX_PER_CLASS:-60000}"
MAX_GEOMETRY_PER_CLASS="${MAX_GEOMETRY_PER_CLASS:-60000}"
PAIR_PROBE_STEPS="${PAIR_PROBE_STEPS:-800}"
BIAS_STEPS="${BIAS_STEPS:-1000}"
WEAK_CLASSES="${WEAK_CLASSES:-picture,counter,desk,sink,cabinet,shower curtain,door}"
CLASS_PAIRS="${CLASS_PAIRS:-picture:wall,counter:cabinet,door:wall}"

echo "=== PTv3 v1.5.1 frozen recovery suite ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "official_root=${OFFICIAL_ROOT}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "summary_prefix=${SUMMARY_PREFIX}"
echo "max_val_scenes=${MAX_VAL_SCENES}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_ptv3_v151_oracle_actionability_analysis.py \
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
  --weak-classes "${WEAK_CLASSES}" \
  --class-pairs "${CLASS_PAIRS}" \
  --max-train-scenes "${MAX_TRAIN_SCENES}" \
  --max-val-scenes "${MAX_VAL_SCENES}" \
  --max-train-points "${MAX_TRAIN_POINTS}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --max-geometry-per-class "${MAX_GEOMETRY_PER_CLASS}" \
  --prototype-counts "${PROTOTYPE_COUNTS:-1,4}" \
  --prototype-lambdas "${PROTOTYPE_LAMBDAS:-0.05,0.1,0.2}" \
  --prototype-taus "${PROTOTYPE_TAUS:-0.05,0.1,0.2}" \
  --pair-probe-steps "${PAIR_PROBE_STEPS}" \
  --bias-steps "${BIAS_STEPS}"

echo "[done] PTv3 v1.5.1 frozen recovery suite"
