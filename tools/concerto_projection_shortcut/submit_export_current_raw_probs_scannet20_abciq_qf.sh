#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=04:00:00
#PBS -N export_rawprob
#PBS -j oe

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}" 2>/dev/null || module load python/3.11
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}" 2>/dev/null || module load cuda/12.6

source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/export_current_raw_probs_scannet20_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -z "${CONFIG:-}" || -z "${WEIGHT:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  echo "[error] CONFIG, WEIGHT, and OUTPUT_DIR must be set" >&2
  exit 2
fi

echo "=== Export current raw probabilities ScanNet20 ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "model_name=${MODEL_NAME:-current_model}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "data_root=${SCANNET_DATA_ROOT:-${SCANNET_EXTRACT_DIR}}"
echo "output_dir=${OUTPUT_DIR}"
echo "max_val_batches=${MAX_VAL_BATCHES:--1}"
nvidia-smi -L || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/export_current_raw_probs_scannet20.py \
  --repo-root "${REPO_ROOT}" \
  --config "${CONFIG}" \
  --weight "${WEIGHT}" \
  --model-name "${MODEL_NAME:-current_model}" \
  --data-root "${SCANNET_DATA_ROOT:-${SCANNET_EXTRACT_DIR}}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-val-batches "${MAX_VAL_BATCHES:--1}" \
  --num-worker "${NUM_WORKER:-4}" \
  --full-scene-chunk-size "${FULL_SCENE_CHUNK_SIZE:-2048}" \
  ${EXTRA_ARGS:-}

echo "[done] export current raw probabilities ScanNet20"
echo "[log] ${LOG_FILE}"
