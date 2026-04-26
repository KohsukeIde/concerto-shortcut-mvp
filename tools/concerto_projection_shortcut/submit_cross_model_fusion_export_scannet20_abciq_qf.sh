#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=04:00:00
#PBS -N xmodel_export
#PBS -j oe

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}" 2>/dev/null || module load python/3.11
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}" 2>/dev/null || module load cuda/12.6

source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export PYTHONPATH="${REPO_ROOT}/external/Utonia:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/cross_model_fusion_export_scannet20_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Cross-model fusion export ScanNet20 ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "data_root=${SCANNET_DATA_ROOT:-${SCANNET_EXTRACT_DIR}}"
echo "output_dir=${OUTPUT_DIR:-data/runs/cross_model_fusion_export_scannet20/avgprob5_fullft_ptv3}"
echo "max_val_batches=${MAX_VAL_BATCHES:--1}"
nvidia-smi -L || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_cross_model_fusion_export_scannet20.py \
  --repo-root "${REPO_ROOT}" \
  --data-root "${SCANNET_DATA_ROOT:-${SCANNET_EXTRACT_DIR}}" \
  --output-dir "${OUTPUT_DIR:-data/runs/cross_model_fusion_export_scannet20/avgprob5_fullft_ptv3}" \
  --summary-prefix "${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_cross_model_fusion_export_scannet20_avgprob5_fullft_ptv3}" \
  --max-val-batches "${MAX_VAL_BATCHES:--1}" \
  --num-worker "${NUM_WORKER:-4}" \
  --full-scene-chunk-size "${FULL_SCENE_CHUNK_SIZE:-2048}" \
  --include-utonia \
  --cached-expert "Concerto fullFT::data/runs/concerto_fullft_raw_probs_scannet20/full" \
  --cached-expert "PTv3_supervised::data/runs/ptv3_v151_raw_probs_scannet20/full" \
  ${EXTRA_ARGS:-}

echo "[done] cross-model fusion export ScanNet20"
echo "[log] ${LOG_FILE}"
