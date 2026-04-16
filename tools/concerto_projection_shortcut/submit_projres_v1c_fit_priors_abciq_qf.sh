#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l select=1
#PBS -l walltime=00:25:00
#PBS -N projres_v1c_priors
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

EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1c}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXP_MIRROR_ROOT}/priors}"
CACHE_ROOT="${CACHE_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1/priors/cache}"
PRIOR_ARCHS="${PRIOR_ARCHS:-linear_z,mlp_z}"
PRIOR_EPOCHS="${PRIOR_EPOCHS:-20}"
PRIOR_BATCH_SIZE="${PRIOR_BATCH_SIZE:-8192}"
PRIOR_HIDDEN_CHANNELS="${PRIOR_HIDDEN_CHANNELS:-512}"

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq" "${OUTPUT_ROOT}"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/projres_v1c_fit_priors_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

ensure_venv_active
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

echo "=== ABCI-Q ProjRes v1c fit coordinate priors ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "python_module=${PYTHON_MODULE}"
echo "cuda_module=${CUDA_MODULE}"
echo "output_root=${OUTPUT_ROOT}"
echo "cache_root=${CACHE_ROOT}"
echo "prior_archs=${PRIOR_ARCHS}"
echo "prior_epochs=${PRIOR_EPOCHS}"
echo "prior_batch_size=${PRIOR_BATCH_SIZE}"
echo "prior_hidden_channels=${PRIOR_HIDDEN_CHANNELS}"
nvidia-smi -L || true
nvcc --version || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/fit_coord_prior.py \
  --output-root "${OUTPUT_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --prior-archs "${PRIOR_ARCHS}" \
  --prior-epochs "${PRIOR_EPOCHS}" \
  --prior-batch-size "${PRIOR_BATCH_SIZE}" \
  --prior-hidden-channels "${PRIOR_HIDDEN_CHANNELS}"

echo "[done] projres v1c prior fitting completed"
echo "[log] ${LOG_PATH}"
