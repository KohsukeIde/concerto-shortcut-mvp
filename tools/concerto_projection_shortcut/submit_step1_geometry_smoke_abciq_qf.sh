#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=01:30:00
#PBS -N step1_geom
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

BACKBONE_CKPT="${BACKBONE_CKPT:-${POINTCEPT_DATA_ROOT}/runs/projres_long/exp/arkit-full-original-long-e025-qf32-continue/model/model_last.pth}"
STEP1_ROOT="${STEP1_ROOT:-${POINTCEPT_DATA_ROOT}/runs/step1_geometry_smoke}"
LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}" "${STEP1_ROOT}"
LOG_FILE="${LOG_DIR}/step1_geometry_smoke_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export REPO_ROOT BACKBONE_CKPT STEP1_ROOT

echo "=== ABCI-Q Step 1 geometry smoke ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "repo_root=${REPO_ROOT}"
echo "backbone_ckpt=${BACKBONE_CKPT}"
echo "step1_root=${STEP1_ROOT}"
echo "rows_per_scene=${ROWS_PER_SCENE:-1024}"
echo "max_batches_train=${MAX_BATCHES_TRAIN:--1}"
echo "max_batches_val=${MAX_BATCHES_VAL:--1}"
echo "geometry_knn=${GEOMETRY_KNN:-32}"
echo "geometry_query_chunk=${GEOMETRY_QUERY_CHUNK:-512}"
echo "geometry_key_chunk=${GEOMETRY_KEY_CHUNK:-32768}"
echo "pass_threshold=${PASS_THRESHOLD:-0.003}"
nvidia-smi -L || true
nvcc --version || true
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
PY

bash "${REPO_ROOT}/tools/concerto_projection_shortcut/run_step1_geometry_smoke.sh"
echo "[done] Step 1 geometry smoke"
echo "[log] ${LOG_FILE}"
