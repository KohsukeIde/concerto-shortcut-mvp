#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QC=1
#PBS -l walltime=04:00:00
#PBS -N conc_data
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

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/concerto_six_data_prep_${DATASET_FILTER:-all}_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Concerto six-dataset image-point data prep ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "dataset_filter=${DATASET_FILTER:-arkit,scannet,scannetpp,s3dis,hm3d,structured3d}"
echo "download_missing=${DOWNLOAD_MISSING:-1}"
echo "extract_missing=${EXTRACT_MISSING:-1}"
echo "log=${LOG_FILE}"

bash "${REPO_ROOT}/tools/concerto_projection_shortcut/setup_concerto_six_imagepoint.sh"

echo "[done] Concerto six-dataset data prep"
echo "[log] ${LOG_FILE}"
