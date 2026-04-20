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
OUTPUT_DIR="${OUTPUT_DIR:-data/runs/scannet_decoder_probe_origin/proposal_recall_analysis}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
REGION_VOXEL_SIZES="${REGION_VOXEL_SIZES:-4,8,16}"
PURITY_THRESHOLDS="${PURITY_THRESHOLDS:-0.5,0.7,0.8,0.9}"
PROPOSAL_SOURCES="${PROPOSAL_SOURCES:-voxel,pred_cc}"
CLASS_PAIRS="${CLASS_PAIRS:-picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall}"
NUM_WORKER="${NUM_WORKER:-8}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_proposal_recall_analysis}"

echo "=== Proposal recall analysis ==="
date +"date=%Y-%m-%dT%H:%M:%S%z"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "region_voxel_sizes=${REGION_VOXEL_SIZES}"
echo "purity_thresholds=${PURITY_THRESHOLDS}"
echo "proposal_sources=${PROPOSAL_SOURCES}"
echo "class_pairs=${CLASS_PAIRS}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_proposal_recall_analysis.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --region-voxel-sizes "${REGION_VOXEL_SIZES}" \
  --purity-thresholds "${PURITY_THRESHOLDS}" \
  --proposal-sources "${PROPOSAL_SOURCES}" \
  --class-pairs "${CLASS_PAIRS}" \
  --num-worker "${NUM_WORKER}" \
  --summary-prefix "${SUMMARY_PREFIX}"

echo "[done] proposal recall analysis"
