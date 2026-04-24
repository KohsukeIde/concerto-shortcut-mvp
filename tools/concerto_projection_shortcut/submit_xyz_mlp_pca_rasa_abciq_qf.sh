#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=06:00:00
#PBS -N xyzpca_rasa
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/data/logs/abciq/

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${WORKDIR}"

source /etc/profile.d/modules.sh 2>/dev/null || true
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

CONFIG="${CONFIG:-configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py}"
WEIGHT="${WEIGHT:-data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth}"
DATA_ROOT="${DATA_ROOT:-data/scannet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_rasa_reservoir}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_xyz_mlp_pca_rasa_reservoir}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:--1}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-1200000}"
MAX_VAL_POINTS="${MAX_VAL_POINTS:-2000000}"
MAX_TRAIN_PER_CLASS="${MAX_TRAIN_PER_CLASS:-60000}"
VAL_SAMPLING="${VAL_SAMPLING:-reservoir}"
XYZ_EPOCHS="${XYZ_EPOCHS:-30}"
NUM_WORKER="${NUM_WORKER:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"

echo "=== XYZ-MLP PCA RASA pilot ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_xyz_mlp_pca_rasa.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --summary-prefix "${SUMMARY_PREFIX}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --max-train-points "${MAX_TRAIN_POINTS}" \
  --max-val-points "${MAX_VAL_POINTS}" \
  --max-train-per-class "${MAX_TRAIN_PER_CLASS}" \
  --val-sampling "${VAL_SAMPLING}" \
  --xyz-epochs "${XYZ_EPOCHS}" \
  --num-worker "${NUM_WORKER}" \
  --batch-size "${BATCH_SIZE}"

echo "[done] XYZ-MLP PCA RASA pilot"
