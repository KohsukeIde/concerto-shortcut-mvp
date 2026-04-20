#!/bin/bash
#PBS -W group_list=qgah50055
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/data/logs/abciq/

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${WORKDIR}"

mkdir -p data/logs/abciq

module purge
module load python/3.11 cuda/12.6

VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/data/venv/pointcept-concerto-py311-cu124/bin/activate}"
source "${VENV_ACTIVATE}"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG="${CONFIG:-configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py}"
WEIGHT="${WEIGHT:-data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth}"
DATA_ROOT="${DATA_ROOT:-data/scannet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/runs/scannet_decoder_probe_origin/proposal_verify_decoder}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-256}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
REGION_VOXEL_SIZE="${REGION_VOXEL_SIZE:-4}"
POSITIVE_PURITY="${POSITIVE_PURITY:-0.8}"
PROPOSAL_CLASSES="${PROPOSAL_CLASSES:-picture,counter,desk,sink,door,shower curtain,cabinet,table,wall}"
WEAK_CLASSES="${WEAK_CLASSES:-picture,counter,desk,sink,cabinet,shower curtain,door}"
THRESHOLDS="${THRESHOLDS:-0.5,0.7,0.9}"
BETAS="${BETAS:-0.25,0.5,1.0}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKER="${NUM_WORKER:-8}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_proposal_verify_decoder}"

echo "=== Proposal-then-Verify Decoder pilot ==="
date +"date=%Y-%m-%dT%H:%M:%S%z"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "region_voxel_size=${REGION_VOXEL_SIZE}"
echo "positive_purity=${POSITIVE_PURITY}"
echo "proposal_classes=${PROPOSAL_CLASSES}"
echo "thresholds=${THRESHOLDS}"
echo "betas=${BETAS}"
echo "epochs=${EPOCHS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/fit_proposal_verify_decoder.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --region-voxel-size "${REGION_VOXEL_SIZE}" \
  --positive-purity "${POSITIVE_PURITY}" \
  --proposal-classes "${PROPOSAL_CLASSES}" \
  --weak-classes "${WEAK_CLASSES}" \
  --thresholds "${THRESHOLDS}" \
  --betas "${BETAS}" \
  --epochs "${EPOCHS}" \
  --num-worker "${NUM_WORKER}" \
  --summary-prefix "${SUMMARY_PREFIX}"

echo "[done] proposal-then-verify decoder pilot"
