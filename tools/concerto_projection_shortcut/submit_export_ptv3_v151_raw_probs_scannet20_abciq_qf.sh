#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=03:00:00
#PBS -N ptv3_probs
#PBS -j oe

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}" 2>/dev/null || module load python/3.11
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}" 2>/dev/null || module load cuda/12.6

source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/export_ptv3_v151_raw_probs_scannet20_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Export PTv3 v1.5.1 raw ScanNet20 probabilities ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "official_root=${OFFICIAL_ROOT:-data/tmp/Pointcept-v1.5.1}"
echo "weight=${WEIGHT:-data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth}"
echo "data_root=${SCANNET_DATA_ROOT:-${SCANNET_EXTRACT_DIR}}"
echo "output_dir=${OUTPUT_DIR:-data/runs/ptv3_v151_raw_probs_scannet20/full}"
echo "max_val_batches=${MAX_VAL_BATCHES:--1}"
nvidia-smi -L || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/export_ptv3_v151_raw_probs_scannet20.py \
  --repo-root "${REPO_ROOT}" \
  --official-root "${OFFICIAL_ROOT:-data/tmp/Pointcept-v1.5.1}" \
  --config "${CONFIG:-configs/scannet/semseg-pt-v3m1-0-base.py}" \
  --weight "${WEIGHT:-data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth}" \
  --data-root "${SCANNET_DATA_ROOT:-${SCANNET_EXTRACT_DIR}}" \
  --split "${SPLIT:-val}" \
  --segment-key "${SEGMENT_KEY:-segment20}" \
  --output-dir "${OUTPUT_DIR:-data/runs/ptv3_v151_raw_probs_scannet20/full}" \
  --max-val-batches "${MAX_VAL_BATCHES:--1}" \
  ${EXTRA_ARGS:-}

echo "[done] export PTv3 v1.5.1 raw probabilities"
echo "[log] ${LOG_FILE}"
