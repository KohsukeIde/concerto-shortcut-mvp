#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l select=1
#PBS -l walltime=01:30:00
#PBS -N projres_follow
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

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq" "${POINTCEPT_DATA_ROOT}/runs/projres_v1"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/projres_v1_followup_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

ensure_venv_active
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1}"
EXP_TAG="${EXP_TAG:--h10032-qf32}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}}"
LOG_DIR="${LOG_DIR:-${EXP_MIRROR_ROOT}/logs/followup/${PBS_JOBID:-manual}_${EXP_TAG#-}}"
MAX_STRESS_BATCHES="${MAX_STRESS_BATCHES:-20}"
FOLLOWUP_PARALLEL="${FOLLOWUP_PARALLEL:-1}"
RUN_STRESS="${RUN_STRESS:-1}"
RUN_LINEAR="${RUN_LINEAR:-1}"
STRESS_GPU="${STRESS_GPU:-0}"
LINEAR_GPU="${LINEAR_GPU:-1}"
LINEAR_GPU_IDS_CSV="${LINEAR_GPU_IDS_CSV:-${LINEAR_GPU}}"
LINEAR_NUM_GPU="${LINEAR_NUM_GPU:-$(awk -F',' '{print NF}' <<< "${LINEAR_GPU_IDS_CSV}")}"
LINEAR_TRAIN_LAUNCHER="${LINEAR_TRAIN_LAUNCHER:-pointcept}"

export EXP_MIRROR_ROOT EXP_TAG SUMMARY_ROOT LOG_DIR
export MAX_STRESS_BATCHES FOLLOWUP_PARALLEL RUN_STRESS RUN_LINEAR
export STRESS_GPU LINEAR_GPU LINEAR_GPU_IDS_CSV LINEAR_NUM_GPU LINEAR_TRAIN_LAUNCHER

echo "=== ABCI-Q ProjRes v1 follow-up ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "python_module=${PYTHON_MODULE}"
echo "cuda_module=${CUDA_MODULE}"
echo "exp_mirror_root=${EXP_MIRROR_ROOT}"
echo "exp_tag=${EXP_TAG}"
echo "summary_root=${SUMMARY_ROOT}"
echo "log_dir=${LOG_DIR}"
echo "max_stress_batches=${MAX_STRESS_BATCHES}"
echo "followup_parallel=${FOLLOWUP_PARALLEL}"
echo "run_stress=${RUN_STRESS}"
echo "run_linear=${RUN_LINEAR}"
echo "stress_gpu=${STRESS_GPU}"
echo "linear_gpu=${LINEAR_GPU}"
echo "linear_gpu_ids_csv=${LINEAR_GPU_IDS_CSV}"
echo "linear_num_gpu=${LINEAR_NUM_GPU}"
echo "linear_train_launcher=${LINEAR_TRAIN_LAUNCHER}"
nvidia-smi -L || true
nvcc --version || true

bash tools/concerto_projection_shortcut/check_setup_status.sh
bash tools/concerto_projection_shortcut/run_projres_v1_followup.sh

echo "[done] projres v1 follow-up completed"
echo "[log] ${LOG_PATH}"
