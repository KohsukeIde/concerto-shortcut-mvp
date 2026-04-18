#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

DATASET_NAME="${DATASET_NAME:-concerto}"
CONFIG_NAME="${CONFIG_NAME:?CONFIG_NAME is required}"
EXP_NAME="${EXP_NAME:?EXP_NAME is required}"
WEIGHT_PATH="${WEIGHT_PATH:-None}"
TRAIN_RESUME="${TRAIN_RESUME:-false}"
TRAIN_GPU_IDS_CSV="${TRAIN_GPU_IDS_CSV:-${GPU_IDS_CSV:-0,1,2,3}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F',' '{print NF}' <<< "${TRAIN_GPU_IDS_CSV}")}"
LOG_DIR="${LOG_DIR:-${POINTCEPT_DATA_ROOT}/runs/projres_v1/logs}"
WORKDIR="${WORKDIR:-${REPO_ROOT}}"
PREFERRED_IFNAME="${PREFERRED_IFNAME:-}"
NCCL_STABLE_MODE="${NCCL_STABLE_MODE:-1}"
POINTCEPT_TRAIN_LAUNCHER="${POINTCEPT_TRAIN_LAUNCHER:-torchrun}"
NODE_START_TIMEOUT_SEC="${NODE_START_TIMEOUT_SEC:-300}"

if [ -z "${PBS_NODEFILE:-}" ] || [ ! -f "${PBS_NODEFILE:-}" ]; then
  echo "[error] PBS_NODEFILE is required for multi-node training." >&2
  exit 2
fi
if ! command -v pbsdsh >/dev/null 2>&1; then
  echo "[error] pbsdsh not found in PATH." >&2
  exit 2
fi

safe_exp="$(printf '%s' "${EXP_NAME}" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="${MULTINODE_RUN_DIR:-${LOG_DIR}/multinode/${PBS_JOBID:-manual}_${safe_exp}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_DIR}/logs"

HOSTS_FILE="${RUN_DIR}/hosts.txt"
awk '!seen[$0]++{print}' "${PBS_NODEFILE}" > "${HOSTS_FILE}"
NNODES="$(wc -l < "${HOSTS_FILE}" | tr -d ' ')"
MASTER_HOST="$(head -n 1 "${HOSTS_FILE}")"
MASTER_ADDR="${MASTER_ADDR:-$(getent ahostsv4 "${MASTER_HOST}" 2>/dev/null | awk '{print $1; exit}' || true)}"
if [ -z "${MASTER_ADDR}" ]; then
  echo "[error] cannot resolve MASTER_HOST=${MASTER_HOST} to IPv4" >&2
  exit 2
fi
if [ -z "${MASTER_PORT:-}" ]; then
  checksum="$(printf '%s' "${DATASET_NAME}/${EXP_NAME}" | cksum | awk '{print $1}')"
  MASTER_PORT="$((20000 + checksum % 20000))"
fi
DIST_URL="tcp://${MASTER_ADDR}:${MASTER_PORT}"
DONE_MARKER="${RUN_DIR}/rank0_done.marker"

echo "=== Pointcept multi-node train ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "dataset=${DATASET_NAME}"
echo "config=${CONFIG_NAME}"
echo "exp=${EXP_NAME}"
echo "weight=${WEIGHT_PATH}"
echo "resume=${TRAIN_RESUME}"
echo "train_gpu_ids=${TRAIN_GPU_IDS_CSV}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "nnodes=${NNODES}"
echo "master_host=${MASTER_HOST}"
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"
echo "dist_url=${DIST_URL}"
echo "run_dir=${RUN_DIR}"
nl -ba "${HOSTS_FILE}"

ENVCONF="${RUN_DIR}/env.conf"
: > "${ENVCONF}"
write_env() {
  local name="$1"
  printf '%s=%q\n' "${name}" "${!name}" >> "${ENVCONF}"
}

export WORKDIR REPO_ROOT DATASET_NAME CONFIG_NAME EXP_NAME WEIGHT_PATH
export TRAIN_RESUME
export TRAIN_GPU_IDS_CSV NPROC_PER_NODE NNODES MASTER_ADDR MASTER_PORT DIST_URL
export RUN_DIR HOSTS_FILE DONE_MARKER VENV_ACTIVATE PYTHON_BIN PYTHON_MODULE CUDA_MODULE
export NODE_START_TIMEOUT_SEC
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export POINTCEPT_TRAIN_LAUNCHER
export COORD_PRIOR_PATH="${COORD_PRIOR_PATH:-}"
export COORD_PROJECTION_ALPHA="${COORD_PROJECTION_ALPHA:-0.05}"
export COORD_PROJECTION_BETA="${COORD_PROJECTION_BETA:-1.0}"
export COORD_RIVAL_PATH="${COORD_RIVAL_PATH:-}"
export SR_LORA_RANK="${SR_LORA_RANK:-}"
export SR_LORA_ALPHA="${SR_LORA_ALPHA:-}"
export SR_LORA_DROPOUT="${SR_LORA_DROPOUT:-}"
export SR_MARGIN_ALPHA="${SR_MARGIN_ALPHA:-}"
export SR_MARGIN_VALUE="${SR_MARGIN_VALUE:-}"
export SR_DISTILL_WEIGHT="${SR_DISTILL_WEIGHT:-}"
export SR_TRAIN_PATCH_PROJ="${SR_TRAIN_PATCH_PROJ:-}"
export SR_TRAIN_STUDENT_HEADS="${SR_TRAIN_STUDENT_HEADS:-}"
export CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-}"
export CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-}"
export CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-}"
export CONCERTO_MAX_TRAIN_ITER="${CONCERTO_MAX_TRAIN_ITER:-}"
export CONCERTO_EPOCH="${CONCERTO_EPOCH:-}"
export CONCERTO_STOP_EPOCH="${CONCERTO_STOP_EPOCH:-}"
export CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_STABLE_MODE PREFERRED_IFNAME

for var in \
  WORKDIR REPO_ROOT DATASET_NAME CONFIG_NAME EXP_NAME WEIGHT_PATH TRAIN_RESUME \
  TRAIN_GPU_IDS_CSV NPROC_PER_NODE NNODES MASTER_ADDR MASTER_PORT DIST_URL \
  RUN_DIR HOSTS_FILE DONE_MARKER VENV_ACTIVATE PYTHON_BIN PYTHON_MODULE CUDA_MODULE \
  NODE_START_TIMEOUT_SEC \
  HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME \
  COORD_PRIOR_PATH COORD_PROJECTION_ALPHA COORD_PROJECTION_BETA COORD_RIVAL_PATH \
  SR_LORA_RANK SR_LORA_ALPHA SR_LORA_DROPOUT SR_MARGIN_ALPHA SR_MARGIN_VALUE \
  SR_DISTILL_WEIGHT SR_TRAIN_PATCH_PROJ SR_TRAIN_STUDENT_HEADS \
  PYTORCH_CUDA_ALLOC_CONF \
  CONCERTO_GLOBAL_BATCH_SIZE CONCERTO_GRAD_ACCUM CONCERTO_NUM_WORKER \
  CONCERTO_MAX_TRAIN_ITER CONCERTO_EPOCH CONCERTO_STOP_EPOCH CONCERTO_ENABLE_FLASH \
  OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NCCL_DEBUG \
  NCCL_STABLE_MODE PREFERRED_IFNAME POINTCEPT_TRAIN_LAUNCHER; do
  write_env "${var}"
done

NODE_ENTRY="${RUN_DIR}/node_entry.sh"
cat > "${NODE_ENTRY}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
set -a
source "$(dirname "$0")/env.conf"
set +a

host="$(hostname)"
NODE_RANK="$(awk -v h="${host}" '$0==h{print NR-1; exit}' "${HOSTS_FILE}")"
if [ -z "${NODE_RANK}" ]; then
  host_s="$(hostname -s)"
  NODE_RANK="$(awk -v h="${host_s}" '$0==h{print NR-1; exit}' "${HOSTS_FILE}")"
fi
if [ -z "${NODE_RANK}" ]; then
  echo "ERROR: cannot determine NODE_RANK for ${host}"
  cat "${HOSTS_FILE}"
  exit 41
fi

log="${RUN_DIR}/logs/${host}.rank${NODE_RANK}.${EXP_NAME}.log"
exec > "${log}" 2>&1

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

cd "${WORKDIR}"
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

if [ -n "${PREFERRED_IFNAME:-}" ] && [ -d "/sys/class/net/${PREFERRED_IFNAME}" ]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${PREFERRED_IFNAME}}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${PREFERRED_IFNAME}}"
else
  ifname="$(ls /sys/class/net 2>/dev/null | grep -E '^ibp' | head -n 1 || true)"
  if [ -n "${ifname}" ]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${ifname}}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${ifname}}"
  fi
fi
if [ "${NCCL_STABLE_MODE:-0}" = "1" ]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
fi
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NCCL_DEBUG
export POINTCEPT_TRAIN_LAUNCHER

echo "=== Pointcept node entry ==="
echo "date=$(date -Is)"
echo "host=${host}"
echo "node_rank=${NODE_RANK}"
echo "nnodes=${NNODES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"
echo "dist_url=${DIST_URL}"
echo "cuda_visible_devices=${TRAIN_GPU_IDS_CSV}"
echo "config=${CONFIG_NAME}"
echo "exp=${EXP_NAME}"
echo "resume=${TRAIN_RESUME}"
echo "concerto_global_batch_size=${CONCERTO_GLOBAL_BATCH_SIZE:-}"
echo "concerto_grad_accum=${CONCERTO_GRAD_ACCUM:-}"
echo "concerto_num_worker=${CONCERTO_NUM_WORKER:-}"
echo "concerto_max_train_iter=${CONCERTO_MAX_TRAIN_ITER:-}"
echo "concerto_epoch=${CONCERTO_EPOCH:-}"
echo "concerto_stop_epoch=${CONCERTO_STOP_EPOCH:-}"
echo "concerto_enable_flash=${CONCERTO_ENABLE_FLASH:-}"
echo "coord_rival_path=${COORD_RIVAL_PATH:-}"
echo "sr_lora_rank=${SR_LORA_RANK:-}"
echo "sr_distill_weight=${SR_DISTILL_WEIGHT:-}"
echo "sr_margin_alpha=${SR_MARGIN_ALPHA:-}"
echo "sr_margin_value=${SR_MARGIN_VALUE:-}"
echo "python=${PYTHON_BIN}"
echo "pointcept_train_launcher=${POINTCEPT_TRAIN_LAUNCHER}"
echo "nccl_stable_mode=${NCCL_STABLE_MODE}"
echo "nccl_p2p_disable=${NCCL_P2P_DISABLE:-}"
echo "nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL:-}"
nvidia-smi -L || true
"${PYTHON_BIN}" - <<'PY'
import importlib
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
print("flash_attn", getattr(importlib.import_module("flash_attn"), "__version__", "OK"))
PY

set +e
CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS_CSV}" \
MACHINE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
DIST_URL="${DIST_URL}" \
COORD_PRIOR_PATH="${COORD_PRIOR_PATH}" \
COORD_PROJECTION_ALPHA="${COORD_PROJECTION_ALPHA}" \
COORD_PROJECTION_BETA="${COORD_PROJECTION_BETA}" \
COORD_RIVAL_PATH="${COORD_RIVAL_PATH}" \
SR_LORA_RANK="${SR_LORA_RANK}" \
SR_LORA_ALPHA="${SR_LORA_ALPHA}" \
SR_LORA_DROPOUT="${SR_LORA_DROPOUT}" \
SR_MARGIN_ALPHA="${SR_MARGIN_ALPHA}" \
SR_MARGIN_VALUE="${SR_MARGIN_VALUE}" \
SR_DISTILL_WEIGHT="${SR_DISTILL_WEIGHT}" \
SR_TRAIN_PATCH_PROJ="${SR_TRAIN_PATCH_PROJ}" \
SR_TRAIN_STUDENT_HEADS="${SR_TRAIN_STUDENT_HEADS}" \
bash "${WORKDIR}/scripts/train.sh" \
  -p "${PYTHON_BIN}" \
  -d "${DATASET_NAME}" \
  -g "${NPROC_PER_NODE}" \
  -m "${NNODES}" \
  -c "${CONFIG_NAME}" \
  -n "${EXP_NAME}" \
  -w "${WEIGHT_PATH}" \
  -r "${TRAIN_RESUME}"
status=$?
set -e

if [ "${status}" -eq 0 ] && [ "${NODE_RANK}" = "0" ]; then
  touch "${DONE_MARKER}"
fi
exit "${status}"
SH
chmod +x "${NODE_ENTRY}"

echo "=== LAUNCH via pbsdsh ==="
echo "+ pbsdsh -c ${NNODES} -- bash ${NODE_ENTRY}"
rm -f "${DONE_MARKER}"
rm -f "${RUN_DIR}/logs"/*.log 2>/dev/null || true
set +e
pbsdsh -c "${NNODES}" -- bash "${NODE_ENTRY}" &
pbsdsh_pid=$!
startup_deadline=$((SECONDS + NODE_START_TIMEOUT_SEC))
startup_timed_out=0
while kill -0 "${pbsdsh_pid}" 2>/dev/null; do
  started_nodes="$(find "${RUN_DIR}/logs" -maxdepth 1 -type f -name '*.log' | wc -l | tr -d ' ')"
  if [ "${started_nodes}" -ge "${NNODES}" ]; then
    echo "[ok] all ${started_nodes}/${NNODES} node logs appeared within startup window"
    break
  fi
  if [ "${SECONDS}" -ge "${startup_deadline}" ]; then
    echo "[error] only ${started_nodes}/${NNODES} node logs appeared after ${NODE_START_TIMEOUT_SEC}s" >&2
    echo "[error] aborting pbsdsh launch before burning the full walltime" >&2
    kill "${pbsdsh_pid}" 2>/dev/null || true
    startup_timed_out=1
    break
  fi
  sleep 5
done
wait "${pbsdsh_pid}"
status=$?
if [ "${startup_timed_out}" -eq 1 ]; then
  status=98
fi
set -e
if [ "${status}" -ne 0 ]; then
  echo "[error] pbsdsh returned ${status}" >&2
fi
if [ ! -f "${DONE_MARKER}" ]; then
  echo "[error] missing completion marker: ${DONE_MARKER}" >&2
  echo "Per-node logs:" >&2
  ls -lh "${RUN_DIR}/logs" >&2 || true
  exit 97
fi
if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

echo "[done] multi-node train completed: ${EXP_NAME}"
echo "[logs] ${RUN_DIR}/logs"
