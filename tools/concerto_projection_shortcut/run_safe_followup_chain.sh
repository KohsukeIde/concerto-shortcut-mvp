#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

ENV_PYTHON="${PYTHON_BIN}"
LOG_DIR="${REPO_ROOT}/tools/concerto_projection_shortcut/logs"
CHAIN_LOG="${LOG_DIR}/safe_followup_chain.log"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${CHAIN_LOG}"
}

checkpoint_path() {
  local exp_name="$1"
  printf '%s/exp/concerto/%s/model/model_last.pth' "${REPO_ROOT}" "${exp_name}"
}

train_log_path() {
  local exp_name="$1"
  printf '%s/exp/concerto/%s/train.log' "${REPO_ROOT}" "${exp_name}"
}

process_running() {
  local pattern="$1"
  pgrep -af "${pattern}" >/dev/null 2>&1
}

gpu_free() {
  local gpu_id="$1"
  python3 - "$gpu_id" <<'PY'
import csv
import subprocess
import sys

gpu_id = sys.argv[1]
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
if not out.strip():
    sys.exit(0)
gpu_map = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
    text=True,
)
uuid_for_index = {}
for line in gpu_map.strip().splitlines():
    idx, uuid = [x.strip() for x in line.split(",", 1)]
    uuid_for_index[idx] = uuid
target_uuid = uuid_for_index[gpu_id]
busy = False
for line in out.strip().splitlines():
    gpu_uuid, pid, _pname, _used = [x.strip() for x in line.split(",", 3)]
    if gpu_uuid == target_uuid:
        busy = True
        break
sys.exit(1 if busy else 0)
PY
}

promote_coord_mlp_checkpoint() {
  local src_exp="arkit-full-continue-coord-mlp-debug2"
  local dst_exp="arkit-full-continue-coord-mlp"
  local src_ckpt
  local dst_ckpt
  src_ckpt="$(checkpoint_path "${src_exp}")"
  dst_ckpt="$(checkpoint_path "${dst_exp}")"
  if [ ! -f "${src_ckpt}" ] || [ -f "${dst_ckpt}" ]; then
    return 0
  fi
  mkdir -p "$(dirname "${dst_ckpt}")"
  cp -f "${src_ckpt}" "${dst_ckpt}"
  if [ -f "$(train_log_path "${src_exp}")" ]; then
    cp -f "$(train_log_path "${src_exp}")" "$(train_log_path "${dst_exp}")"
  fi
  log "promoted coord-mlp checkpoint ${src_exp} -> ${dst_exp}"
}

launch_concerto_continue_if_possible() {
  launch_continuation_if_possible \
    "arkit-full-continue-concerto" \
    "pretrain-concerto-v1m1-0-arkit-full-continue-safe"
}

launch_continuation_if_possible() {
  local exp_name="$1"
  local config_name="$2"
  local extra_running_pattern="${3:-$1}"
  local ckpt
  ckpt="$(checkpoint_path "${exp_name}")"
  if [ -f "${ckpt}" ] || process_running "${extra_running_pattern}"; then
    return 0
  fi

  local gpu_id=""
  if gpu_free 0; then
    gpu_id="0"
  elif gpu_free 1; then
    gpu_id="1"
  else
    return 0
  fi

  log "launch ${exp_name} on gpu=${gpu_id}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "${REPO_ROOT}/scripts/train.sh" \
      -p "${ENV_PYTHON}" \
      -d concerto \
      -g 1 \
      -c "${config_name}" \
      -n "${exp_name}" \
      -w "${REPO_ROOT}/weights/concerto/concerto_base_origin.pth" \
      >> "${LOG_DIR}/${exp_name}.launch.log" 2>&1 &
}

all_continuations_ready() {
  [ -f "$(checkpoint_path "arkit-full-continue-concerto")" ] && \
  [ -f "$(checkpoint_path "arkit-full-continue-no-enc2d")" ] && \
  [ -f "$(checkpoint_path "arkit-full-continue-coord-mlp")" ]
}

linears_done() {
  [ -f "$(checkpoint_path "scannet-proxy-concerto-continue-lin")" ] && \
  [ -f "$(checkpoint_path "scannet-proxy-no-enc2d-continue-lin")" ] && \
  [ -f "$(checkpoint_path "scannet-proxy-coord-mlp-continue-lin")" ]
}

launch_linear_trio_if_ready() {
  if ! all_continuations_ready || linears_done || process_running 'scannet-proxy-.*-continue-lin'; then
    return 0
  fi
  log "launch ScanNet linear trio"
  nohup env PYTHON_BIN="${ENV_PYTHON}" LIN_NUM_GPU=1 PARALLEL_SINGLE_GPU=1 GPU_IDS_CSV=0,1 \
    bash "${REPO_ROOT}/tools/concerto_projection_shortcut/run_scannet_proxy.sh" lin \
    >> "${LOG_DIR}/scannet_lin_followup.launch.log" 2>&1 &
}

log "safe follow-up chain start"
while true; do
  promote_coord_mlp_checkpoint
  launch_continuation_if_possible \
    "arkit-full-continue-no-enc2d" \
    "pretrain-concerto-v1m1-0-arkit-full-no-enc2d-continue-safe"
  launch_continuation_if_possible \
    "arkit-full-continue-coord-mlp" \
    "pretrain-concerto-v1m1-0-arkit-full-coord-mlp-continue-safe" \
    "arkit-full-continue-coord-mlp|arkit-full-continue-coord-mlp-debug2"
  launch_concerto_continue_if_possible
  launch_linear_trio_if_ready

  if linears_done; then
    log "all follow-up stages complete"
    exit 0
  fi
  sleep 60
done
