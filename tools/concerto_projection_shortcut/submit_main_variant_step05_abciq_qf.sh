#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=02:00:00
#PBS -N main_step05
#PBS -j oe

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}
cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}"
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
export PYTHON_BIN="${VENV_DIR}/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/main_variant_step05_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ABCI-Q main-variant Step 0/0.5 ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "main_origin_weight=${MAIN_ORIGIN_WEIGHT:-${WEIGHT_DIR}/concerto_base_origin.pth}"
echo "datasets=${DATASETS:-arkit,scannet,scannetpp,s3dis,hm3d,structured3d}"
echo "dry_run=${DRY_RUN:-0}"
echo "headfit_max_train_batches=${HEADFIT_MAX_TRAIN_BATCHES:-128}"
echo "coord_max_train_batches=${COORD_MAX_TRAIN_BATCHES:-256}"
echo "walltime=${PBS_WALLTIME:-}"
nvidia-smi -L || true
nvcc --version || true
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
PY

bash "${REPO_ROOT}/tools/concerto_projection_shortcut/run_main_variant_step05.sh"

echo "[done] main-variant Step 0/0.5"
echo "[log] ${LOG_FILE}"
