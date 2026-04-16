#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=03:00:00
#PBS -N posthoc_surg
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

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq" "${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/posthoc_surgery_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

ensure_venv_active
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

BACKBONE_CKPT="${BACKBONE_CKPT:-${WEIGHT_DIR}/concerto_base_origin.pth}"
BACKBONE_TAG="${BACKBONE_TAG:-}"
RUN_SUITE="${RUN_SUITE:-1}"
METHOD="${METHOD:-splice3d}"
NUISANCE="${NUISANCE:-height+xyz}"
POSTHOC_SPECS="${POSTHOC_SPECS:-splice3d:height+xyz splice3d:height hlns:height+xyz}"
POSTHOC_ROOT="${POSTHOC_ROOT:-${POINTCEPT_DATA_ROOT}/runs/posthoc_surgery}"
ROWS_PER_SCENE="${ROWS_PER_SCENE:-1024}"
MAX_BATCHES_TRAIN="${MAX_BATCHES_TRAIN:--1}"
MAX_BATCHES_VAL="${MAX_BATCHES_VAL:--1}"
GAMMA="${GAMMA:-1.0}"
GPUS="${GPUS:-4}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
LINEAR_TRAIN_LAUNCHER="${LINEAR_TRAIN_LAUNCHER:-torchrun}"
RUN_LINEAR="${RUN_LINEAR:-1}"

export BACKBONE_CKPT BACKBONE_TAG RUN_SUITE METHOD NUISANCE POSTHOC_SPECS POSTHOC_ROOT
export ROWS_PER_SCENE MAX_BATCHES_TRAIN MAX_BATCHES_VAL GAMMA
export GPUS GPU_IDS_CSV LINEAR_TRAIN_LAUNCHER RUN_LINEAR

echo "=== ABCI-Q posthoc nuisance surgery ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "python_module=${PYTHON_MODULE}"
echo "cuda_module=${CUDA_MODULE}"
echo "backbone_ckpt=${BACKBONE_CKPT}"
echo "backbone_tag=${BACKBONE_TAG}"
echo "run_suite=${RUN_SUITE}"
echo "method=${METHOD}"
echo "nuisance=${NUISANCE}"
echo "posthoc_specs=${POSTHOC_SPECS}"
echo "posthoc_root=${POSTHOC_ROOT}"
echo "rows_per_scene=${ROWS_PER_SCENE}"
echo "max_batches_train=${MAX_BATCHES_TRAIN}"
echo "max_batches_val=${MAX_BATCHES_VAL}"
echo "gamma=${GAMMA}"
echo "gpus=${GPUS}"
echo "gpu_ids_csv=${GPU_IDS_CSV}"
echo "linear_train_launcher=${LINEAR_TRAIN_LAUNCHER}"
echo "run_linear=${RUN_LINEAR}"
nvidia-smi -L || true
nvcc --version || true

if [ ! -f "${BACKBONE_CKPT}" ]; then
  echo "[error] missing BACKBONE_CKPT=${BACKBONE_CKPT}" >&2
  exit 2
fi

if [ "${RUN_SUITE}" = "1" ]; then
  bash tools/concerto_projection_shortcut/run_posthoc_surgery_suite.sh
else
  bash tools/concerto_projection_shortcut/run_posthoc_surgery_chain.sh
fi

echo "[done] posthoc nuisance surgery"
echo "[log] ${LOG_PATH}"
