#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=00:45:00
#PBS -N c3d_ctrl
#PBS -j oe

set -euo pipefail

cd "${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}" || exit 1

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source tools/concerto_projection_shortcut/device_defaults.sh
ensure_venv_active

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/concerto3d_dino_exact_controls_stepA_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

echo "=== ABCI-Q Concerto 3D / DINO exact-patch controls Step A ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "out_dir=${OUT_DIR:-${POINTCEPT_DATA_ROOT}/runs/concerto3d_dino_exact_controls_stepA/scannet_medium}"
nvidia-smi -L || true
nvcc --version || true

bash tools/concerto_projection_shortcut/run_concerto3d_dino_exact_controls_stepA.sh

echo "[done] Concerto 3D / DINO exact-patch controls Step A"
echo "[log] ${LOG_PATH}"
