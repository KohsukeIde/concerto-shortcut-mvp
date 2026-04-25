#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=06:00:00
#PBS -N xmodel_stack
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
LOG_FILE="${LOG_DIR}/cross_model_fusion_cv_stacker_scannet20_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Cross-model fusion CV stacker ScanNet20 ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "data_root=${SCANNET_DATA_ROOT:-${SCANNET_EXTRACT_DIR}}"
echo "output_dir=${OUTPUT_DIR:-data/runs/cross_model_fusion_scannet20/cv_stacker}"
echo "max_val_batches=${MAX_VAL_BATCHES:--1}"
echo "sample_points_per_scene=${SAMPLE_POINTS_PER_SCENE:-4096}"
echo "epochs=${STACKER_EPOCHS:-60}"
echo "class_weight_powers=${CLASS_WEIGHT_POWERS:-0.0,0.5}"
nvidia-smi -L || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_cross_model_fusion_cv_stacker_scannet20.py \
  --repo-root "${REPO_ROOT}" \
  --data-root "${SCANNET_DATA_ROOT:-${SCANNET_EXTRACT_DIR}}" \
  --output-dir "${OUTPUT_DIR:-data/runs/cross_model_fusion_scannet20/cv_stacker}" \
  --max-val-batches "${MAX_VAL_BATCHES:--1}" \
  --num-worker "${NUM_WORKER:-4}" \
  --full-scene-chunk-size "${FULL_SCENE_CHUNK_SIZE:-2048}" \
  --sample-points-per-scene "${SAMPLE_POINTS_PER_SCENE:-4096}" \
  --stacker-epochs "${STACKER_EPOCHS:-60}" \
  --class-weight-powers "${CLASS_WEIGHT_POWERS:-0.0,0.5}" \
  --summary-prefix "${SUMMARY_PREFIX:-tools/concerto_projection_shortcut/results_cross_model_fusion_cv_stacker_scannet20}" \
  --include-utonia \
  ${EXTRA_ARGS:-}

echo "[done] cross-model fusion CV stacker ScanNet20"
echo "[log] ${LOG_FILE}"
