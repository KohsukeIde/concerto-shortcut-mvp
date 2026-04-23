#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=01:30:00
#PBS -N utonia_orc
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

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/utonia_oracle_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ABCI-Q Utonia ScanNet oracle/actionability ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "output_dir=${OUTPUT_DIR:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_utonia_scannet_oracle_actionability}"
echo "class_pairs=${CLASS_PAIRS:-picture:wall,door:wall,counter:cabinet}"
echo "weak_classes=${WEAK_CLASSES:-picture,counter,door}"
echo "max_train_batches=${MAX_TRAIN_BATCHES:-128}"
echo "max_val_batches=${MAX_VAL_BATCHES:-64}"
nvidia-smi -L || true
nvcc --version || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_utonia_scannet_oracle_actionability.py \
  --repo-root "${REPO_ROOT}" \
  --data-root "${UTONIA_DATA_ROOT:-data/scannet}" \
  --utonia-weight "${UTONIA_WEIGHT:-${REPO_ROOT}/data/weights/utonia/utonia.pth}" \
  --seg-head-weight "${UTONIA_SEG_HEAD_WEIGHT:-${REPO_ROOT}/data/weights/utonia/utonia_linear_prob_head_sc.pth}" \
  --output-dir "${OUTPUT_DIR:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_utonia_scannet_oracle_actionability}" \
  --weak-classes "${WEAK_CLASSES:-picture,counter,door}" \
  --class-pairs "${CLASS_PAIRS:-picture:wall,door:wall,counter:cabinet}" \
  --max-train-batches "${MAX_TRAIN_BATCHES:-128}" \
  --max-val-batches "${MAX_VAL_BATCHES:-64}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-worker "${NUM_WORKER:-2}" \
  --disable-flash

echo "[done] Utonia ScanNet oracle/actionability"
echo "[log] ${LOG_FILE}"
