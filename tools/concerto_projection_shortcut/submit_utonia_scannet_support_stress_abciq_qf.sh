#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=06:00:00
#PBS -N utonia_stress
#PBS -j oe

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}"
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}"

source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export PYTHONPATH="${REPO_ROOT}/external/Utonia:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/utonia_support_stress_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ABCI-Q Utonia ScanNet support stress ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "output_dir=${OUTPUT_DIR:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_utonia_scannet_support_stress}"
nvidia-smi -L || true
nvcc --version || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_utonia_scannet_support_stress.py \
  --repo-root "${REPO_ROOT}" \
  --data-root "${UTONIA_DATA_ROOT:-data/scannet}" \
  --utonia-weight "${UTONIA_WEIGHT:-${REPO_ROOT}/data/weights/utonia/utonia.pth}" \
  --seg-head-weight "${UTONIA_SEG_HEAD_WEIGHT:-${REPO_ROOT}/data/weights/utonia/utonia_linear_prob_head_sc.pth}" \
  --output-dir "${OUTPUT_DIR:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_utonia_scannet_support_stress}" \
  --random-keep-ratios "${RANDOM_KEEP_RATIOS:-0.2}" \
  --structured-keep-ratios "${STRUCTURED_KEEP_RATIOS:-0.2}" \
  --masked-model-keep-ratios "${MASKED_MODEL_KEEP_RATIOS:-0.2}" \
  --fixed-point-counts "${FIXED_POINT_COUNTS:-4000}" \
  --max-val-scenes "${MAX_VAL_SCENES:--1}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-worker "${NUM_WORKER:-2}" \
  --feature-zero \
  --disable-flash

echo "[done] Utonia ScanNet support stress"
echo "[log] ${LOG_FILE}"
