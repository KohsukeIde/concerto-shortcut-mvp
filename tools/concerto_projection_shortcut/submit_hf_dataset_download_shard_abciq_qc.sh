#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QC=1
#PBS -l walltime=08:00:00
#PBS -N hf_shard
#PBS -j oe

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}
cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}" 2>/dev/null || true
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}" 2>/dev/null || true

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
export PYTHON_BIN="${VENV_DIR}/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME

: "${HF_REPO_ID:?HF_REPO_ID is required}"
: "${HF_LOCAL_DIR:?HF_LOCAL_DIR is required}"
: "${SHARD_COUNT:?SHARD_COUNT is required}"
: "${SHARD_INDEX:?SHARD_INDEX is required}"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/hf_dataset_download_${PBS_JOBID:-manual}_shard${SHARD_INDEX}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== HF dataset shard download ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "hf_repo_id=${HF_REPO_ID}"
echo "hf_local_dir=${HF_LOCAL_DIR}"
echo "hf_include_pattern=${HF_INCLUDE_PATTERN:-}"
echo "shard=${SHARD_INDEX}/${SHARD_COUNT}"
echo "log=${LOG_FILE}"

"${PYTHON_BIN}" tools/concerto_projection_shortcut/download_hf_dataset_shard.py \
  --repo-id "${HF_REPO_ID}" \
  --repo-type dataset \
  --local-dir "${HF_LOCAL_DIR}" \
  --shard-count "${SHARD_COUNT}" \
  --shard-index "${SHARD_INDEX}" \
  --include-pattern "${HF_INCLUDE_PATTERN:-}"

echo "[done] HF dataset shard download"
echo "[log] ${LOG_FILE}"
