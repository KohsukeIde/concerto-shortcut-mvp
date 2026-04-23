#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=01:00:00
#PBS -N sonft_orc
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/data/logs/abciq/

set -euo pipefail

WORKDIR="/groups/qgah50055/ide/concerto-shortcut-mvp"
cd "${WORKDIR}"

module load python/3.11/3.11.14
module load cuda/12.6/12.6.2

source "${WORKDIR}/data/venv/pointcept-concerto-py311-cu124/bin/activate"

export HF_HOME="${WORKDIR}/data/hf-home"
export HF_HUB_CACHE="${WORKDIR}/data/hf-home/hub"
export HF_XET_CACHE="${WORKDIR}/data/hf-home/xet"
export TORCH_HOME="${WORKDIR}/data/torch-home"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

CONFIG="${WORKDIR}/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/config.py"
WEIGHT="${WORKDIR}/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/model/model_best.pth"
OUTPUT_DIR="${WORKDIR}/tools/concerto_projection_shortcut/results_sonata_fullft_oracle_actionability"

echo "=== Sonata full-FT oracle actionability ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_oracle_actionability_analysis.py \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --data-root data/scannet \
  --output-dir "${OUTPUT_DIR}" \
  --weak-classes "picture,counter,door" \
  --class-pairs "picture:wall,door:wall,counter:cabinet" \
  --max-train-batches 256 \
  --max-val-batches -1 \
  --max-train-points 600000 \
  --max-per-class 60000 \
  --max-geometry-per-class 60000 \
  --pair-probe-steps 800 \
  --bias-steps 1000 \
  --num-worker 8 \
  --batch-size 1

echo "[done] Sonata full-FT oracle actionability"
