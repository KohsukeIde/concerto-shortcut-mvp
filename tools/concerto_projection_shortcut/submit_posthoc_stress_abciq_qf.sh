#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=01:20:00
#PBS -j oe
#PBS -N posthoc_stress

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/posthoc_stress_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ABCI-Q posthoc stress suite ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "posthoc_stress_root=${POSTHOC_STRESS_ROOT:-${POINTCEPT_DATA_ROOT}/runs/posthoc_stress_e025pilot}"
echo "stresses=${STRESSES:-clean local_surface_destroy z_flip xy_swap roll_90_x}"
echo "max_batches=${MAX_BATCHES:--1}"
echo "run_parallel=${RUN_PARALLEL:-1}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.11/3.11.14
module load cuda/12.6/12.6.2

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
export PYTHON_BIN="${VENV_DIR}/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nvidia-smi -L || true
nvcc --version || true
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
PY

bash "${REPO_ROOT}/tools/concerto_projection_shortcut/run_posthoc_stress_suite.sh"
