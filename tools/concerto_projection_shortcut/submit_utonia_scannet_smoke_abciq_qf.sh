#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=00:20:00
#PBS -N utonia_smoke
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
LOG_FILE="${LOG_DIR}/utonia_smoke_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ABCI-Q Utonia ScanNet smoke ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "scene_dir=${UTONIA_SCENE_DIR:-${REPO_ROOT}/data/scannet/val/scene0685_00}"
echo "utonia_weight=${UTONIA_WEIGHT:-${REPO_ROOT}/data/weights/utonia/utonia.pth}"
echo "seg_head_weight=${UTONIA_SEG_HEAD_WEIGHT:-${REPO_ROOT}/data/weights/utonia/utonia_linear_prob_head_sc.pth}"
nvidia-smi -L || true
nvcc --version || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/smoke_utonia_scannet_scene.py \
  --scene-dir "${UTONIA_SCENE_DIR:-${REPO_ROOT}/data/scannet/val/scene0685_00}" \
  --utonia-weight "${UTONIA_WEIGHT:-${REPO_ROOT}/data/weights/utonia/utonia.pth}" \
  --seg-head-weight "${UTONIA_SEG_HEAD_WEIGHT:-${REPO_ROOT}/data/weights/utonia/utonia_linear_prob_head_sc.pth}" \
  --disable-flash

echo "[done] Utonia ScanNet smoke"
echo "[log] ${LOG_FILE}"
