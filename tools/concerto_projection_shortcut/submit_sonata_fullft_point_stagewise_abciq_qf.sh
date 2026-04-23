#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=00:50:00
#PBS -N sonft_stg
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/data/logs/abciq/

set -euo pipefail

WORKDIR="/groups/qgah50055/ide/concerto-shortcut-mvp"
cd "${WORKDIR}"

mkdir -p data/logs/abciq

module load python/3.11/3.11.14
module load cuda/12.6/12.6.2

source "${WORKDIR}/data/venv/pointcept-concerto-py311-cu124/bin/activate"

export HF_HOME="${WORKDIR}/data/hf-home"
export HF_HUB_CACHE="${WORKDIR}/data/hf-home/hub"
export HF_XET_CACHE="${WORKDIR}/data/hf-home/xet"
export TORCH_HOME="${WORKDIR}/data/torch-home"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

LINEAR_CONFIG="${WORKDIR}/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/config.py"
LINEAR_WEIGHT="${WORKDIR}/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/model/model_best.pth"
OUTPUT_DIR="${WORKDIR}/tools/concerto_projection_shortcut/results_sonata_fullft_point_stagewise_trace"

echo "=== Sonata full-FT point stagewise trace ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "linear_config=${LINEAR_CONFIG}"
echo "linear_weight=${LINEAR_WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi -L || true

python tools/concerto_projection_shortcut/eval_scannet_point_stagewise_trace.py \
  --linear-config "${LINEAR_CONFIG}" \
  --linear-weight "${LINEAR_WEIGHT}" \
  --data-root data/scannet \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-batches 256 \
  --max-val-batches 128 \
  --max-per-class 60000 \
  --bootstrap-iters 100 \
  --logreg-steps 600 \
  --num-worker 8 \
  --batch-size 1 \
  --class-pairs "picture:wall,door:wall,counter:cabinet"

echo "[done] Sonata full-FT point stagewise trace"
