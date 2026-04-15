#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=4
#PBS -l walltime=06:00:00
#PBS -N projres_cont_qf16
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
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/projres_v1_continue_qf16_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

ensure_venv_active
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export POINTCEPT_TRAIN_LAUNCHER="${POINTCEPT_TRAIN_LAUNCHER:-torchrun}"
export NCCL_STABLE_MODE="${NCCL_STABLE_MODE:-1}"
if [ "${NCCL_STABLE_MODE}" = "1" ]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
fi

DATASET_NAME="${DATASET_NAME:-concerto}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1}"
LOG_DIR="${LOG_DIR:-${EXP_MIRROR_ROOT}/logs}"
EXP_TAG="${EXP_TAG:--h10016-qf16}"
SMOKE_SUMMARY_ROOT="${SMOKE_SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/h10016-qf1fixed64}"
SELECTED_SMOKE_JSON="${SELECTED_SMOKE_JSON:-${SMOKE_SUMMARY_ROOT}/selected_smoke.json}"
SELECTED_PRIOR_JSON="${SELECTED_PRIOR_JSON:-${EXP_MIRROR_ROOT}/priors/selected_prior.json}"
CONTINUE_CONFIG="${CONTINUE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-projres-v1a-continue-h10016}"
OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${WEIGHT_DIR}/concerto_base_origin.pth}"
CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-32}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-3}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-64}"
CONCERTO_MAX_TRAIN_ITER="${CONCERTO_MAX_TRAIN_ITER:-0}"
CONCERTO_EPOCH="${CONCERTO_EPOCH:-0}"
CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"
PREFLIGHT_CONCERTO_GLOBAL_BATCH_SIZE="${PREFLIGHT_CONCERTO_GLOBAL_BATCH_SIZE:-1}"
PREFLIGHT_CONCERTO_NUM_WORKER="${PREFLIGHT_CONCERTO_NUM_WORKER:-0}"

if [ ! -f "${SELECTED_SMOKE_JSON}" ]; then
  echo "[error] missing selected smoke: ${SELECTED_SMOKE_JSON}" >&2
  exit 2
fi
if [ ! -f "${SELECTED_PRIOR_JSON}" ]; then
  echo "[error] missing selected prior: ${SELECTED_PRIOR_JSON}" >&2
  exit 2
fi
if [ ! -f "${OFFICIAL_WEIGHT}" ]; then
  echo "[error] missing official weight: ${OFFICIAL_WEIGHT}" >&2
  exit 2
fi

SELECTED_ALPHA="$("${PYTHON_BIN}" - "${SELECTED_SMOKE_JSON}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
if not payload.get("pass"):
    raise SystemExit("selected smoke did not pass")
print(payload["selected"]["alpha"])
PY
)"
SELECTED_TAG="$(printf '%s' "${SELECTED_ALPHA}" | tr -d '.')"
COORD_PRIOR_PATH="$("${PYTHON_BIN}" - "${SELECTED_PRIOR_JSON}" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["selected_path"])
PY
)"
CONTINUE_EXP="${CONTINUE_EXP:-arkit-full-projres-v1a-alpha${SELECTED_TAG}${EXP_TAG}-continue}"

export DATASET_NAME GPU_IDS_CSV EXP_MIRROR_ROOT LOG_DIR CONTINUE_CONFIG OFFICIAL_WEIGHT
export CONCERTO_GLOBAL_BATCH_SIZE CONCERTO_GRAD_ACCUM CONCERTO_NUM_WORKER
export CONCERTO_MAX_TRAIN_ITER CONCERTO_EPOCH CONCERTO_ENABLE_FLASH
export COORD_PRIOR_PATH COORD_PROJECTION_ALPHA="${SELECTED_ALPHA}"

echo "=== ABCI-Q ProjRes v1 H100 multi-node continuation ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "pbs_nodefile=${PBS_NODEFILE:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python=$(command -v python)"
echo "gpu_ids=${GPU_IDS_CSV}"
echo "exp_mirror_root=${EXP_MIRROR_ROOT}"
echo "smoke_summary=${SELECTED_SMOKE_JSON}"
echo "selected_alpha=${SELECTED_ALPHA}"
echo "coord_prior_path=${COORD_PRIOR_PATH}"
echo "continue_config=${CONTINUE_CONFIG}"
echo "continue_exp=${CONTINUE_EXP}"
echo "concerto_global_batch_size=${CONCERTO_GLOBAL_BATCH_SIZE}"
echo "concerto_grad_accum=${CONCERTO_GRAD_ACCUM}"
echo "concerto_num_worker=${CONCERTO_NUM_WORKER}"
echo "concerto_max_train_iter=${CONCERTO_MAX_TRAIN_ITER}"
echo "concerto_epoch=${CONCERTO_EPOCH}"
echo "concerto_enable_flash=${CONCERTO_ENABLE_FLASH}"
echo "preflight_concerto_global_batch_size=${PREFLIGHT_CONCERTO_GLOBAL_BATCH_SIZE}"
echo "preflight_concerto_num_worker=${PREFLIGHT_CONCERTO_NUM_WORKER}"
echo "pointcept_train_launcher=${POINTCEPT_TRAIN_LAUNCHER}"
echo "nccl_stable_mode=${NCCL_STABLE_MODE}"
echo "nccl_p2p_disable=${NCCL_P2P_DISABLE:-}"
echo "nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL:-}"
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
  echo "nodes:"
  awk '!seen[$0]++{print}' "${PBS_NODEFILE}" | nl -ba
fi
nvidia-smi -L || true

EXP_LINK="exp/${DATASET_NAME}/${CONTINUE_EXP}"
EXP_TARGET="${EXP_MIRROR_ROOT}/exp/${CONTINUE_EXP}"
mkdir -p "exp/${DATASET_NAME}" "${EXP_TARGET}"
if [ ! -e "${EXP_LINK}" ]; then
  ln -s "${EXP_TARGET}" "${EXP_LINK}"
fi

if [ -f "${EXP_LINK}/model/model_last.pth" ]; then
  echo "[done] continuation checkpoint already exists: ${EXP_LINK}/model/model_last.pth"
  echo "[log] ${LOG_PATH}"
  exit 0
fi

env CUDA_VISIBLE_DEVICES="$(awk -F',' '{print $1}' <<< "${GPU_IDS_CSV}")" \
  CONCERTO_GLOBAL_BATCH_SIZE="${PREFLIGHT_CONCERTO_GLOBAL_BATCH_SIZE}" \
  CONCERTO_NUM_WORKER="${PREFLIGHT_CONCERTO_NUM_WORKER}" \
  COORD_PRIOR_PATH="${COORD_PRIOR_PATH}" \
  COORD_PROJECTION_ALPHA="${SELECTED_ALPHA}" \
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/preflight.py \
  --check-data --check-batch --check-forward --config "${CONTINUE_CONFIG}" \
  --data-root "${ARKIT_FULL_META_ROOT}"

DATASET_NAME="${DATASET_NAME}" \
CONFIG_NAME="${CONTINUE_CONFIG}" \
EXP_NAME="${CONTINUE_EXP}" \
WEIGHT_PATH="${OFFICIAL_WEIGHT}" \
TRAIN_GPU_IDS_CSV="${GPU_IDS_CSV}" \
COORD_PRIOR_PATH="${COORD_PRIOR_PATH}" \
COORD_PROJECTION_ALPHA="${SELECTED_ALPHA}" \
LOG_DIR="${LOG_DIR}" \
bash tools/concerto_projection_shortcut/run_pointcept_train_multinode_pbsdsh.sh

echo "[done] projres v1 H100 multi-node continuation completed"
echo "[log] ${LOG_PATH}"
