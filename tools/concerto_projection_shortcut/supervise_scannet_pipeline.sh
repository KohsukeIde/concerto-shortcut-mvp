#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

NUM_GPU="${NUM_GPU:-2}"
GATE_NUM_GPU="${GATE_NUM_GPU:-1}"
PRETRAIN_NUM_GPU="${PRETRAIN_NUM_GPU:-1}"
LIN_NUM_GPU="${LIN_NUM_GPU:-1}"
FT_NUM_GPU="${FT_NUM_GPU:-1}"
PARALLEL_SINGLE_GPU="${PARALLEL_SINGLE_GPU:-1}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1}"
DATASET_NAME="${DATASET_NAME:-concerto}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="${REPO_ROOT}/tools/concerto_projection_shortcut/logs"
STATUS_PATH="${REPO_ROOT}/tools/concerto_projection_shortcut/scannet_pipeline_status.md"
DONE_STAMP="${REPO_ROOT}/tools/concerto_projection_shortcut/scannet_pipeline.done"
LOCK_PATH="${REPO_ROOT}/tools/concerto_projection_shortcut/scannet_pipeline.lock"
SUP_LOG="${LOG_DIR}/supervise_scannet_pipeline.log"

OFFICIAL_GATE_EXP="scannet-proxy-official-origin-lin"
FOLLOWUP_CONFIG="pretrain-concerto-v1m1-0-probe-enc2d-full-prepool-global-feature-index-permutation"
FOLLOWUP_NAME="arkit-full-causal-prepool-global-feature-index-permutation"
GATE_CONFIG="${GATE_CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-safe}"
LIN_CONFIG="${LIN_CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-safe}"
FT_CONFIG="${FT_CONFIG:-semseg-ptv3-base-v1m1-0c-scannet-ft-proxy-safe}"

CONTINUE_EXPS=(
  "arkit-full-continue-concerto"
  "arkit-full-continue-no-enc2d"
  "arkit-full-continue-coord-mlp"
)
LIN_EXPS=(
  "scannet-proxy-concerto-continue-lin"
  "scannet-proxy-no-enc2d-continue-lin"
  "scannet-proxy-coord-mlp-continue-lin"
)

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "[skip] supervisor already running: ${LOCK_PATH}"
  exit 0
fi

ensure_conda_active

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${SUP_LOG}"
}

checkpoint_path() {
  local exp_name="$1"
  printf '%s/exp/%s/%s/model/model_last.pth' "${REPO_ROOT}" "${DATASET_NAME}" "${exp_name}"
}

scannet_root() {
  if [ -d "${SCANNET_EXTRACT_DIR}/splits" ]; then
    printf '%s' "${SCANNET_EXTRACT_DIR}"
  elif [ -d "${SCANNET_EXTRACT_DIR}/scannet/splits" ]; then
    printf '%s' "${SCANNET_EXTRACT_DIR}/scannet"
  else
    printf ''
  fi
}

scannet_ready() {
  [ -n "$(scannet_root)" ]
}

process_running() {
  local pattern="$1"
  pgrep -af "${pattern}" >/dev/null 2>&1
}

start_bg() {
  local tag="$1"
  shift
  local launch_log="${LOG_DIR}/${tag}.launch.log"
  log "[start] ${tag}"
  nohup bash -lc "cd '${REPO_ROOT}' && source '${CONDA_ROOT}/etc/profile.d/conda.sh' && set +u && conda activate '${CONDA_ENV_NAME}' && set -u && $*" >> "${launch_log}" 2>&1 &
}

stash_incomplete_exp() {
  local exp_name="$1"
  local exp_dir="${REPO_ROOT}/exp/${DATASET_NAME}/${exp_name}"
  local checkpoint="${exp_dir}/model/model_last.pth"
  if [ -d "${exp_dir}" ] && [ ! -f "${checkpoint}" ]; then
    local stamp
    stamp="$(date +%Y%m%d-%H%M%S)"
    local stale_dir="${exp_dir}-stale-${stamp}"
    mv "${exp_dir}" "${stale_dir}"
    log "[move] ${exp_dir} -> ${stale_dir}"
  fi
}

all_exist() {
  local exp_name
  for exp_name in "$@"; do
    if [ ! -f "$(checkpoint_path "${exp_name}")" ]; then
      return 1
    fi
  done
  return 0
}

ensure_scannet_symlink() {
  local root
  root="$(scannet_root)"
  if [ -n "${root}" ]; then
    mkdir -p "${REPO_ROOT}/data"
    ln -sfn "${root}" "${REPO_ROOT}/data/scannet"
  fi
}

write_status() {
  local stage="$1"
  local root
  root="$(scannet_root)"
  {
    echo "# ScanNet Pipeline Status"
    echo
    echo "- Time: $(date '+%F %T %Z')"
    echo "- Repo Root: ${REPO_ROOT}"
    echo "- Stage: ${stage}"
    echo "- ScanNet Ready: $( [ -n "${root}" ] && echo yes || echo no )"
    echo "- ScanNet Root: ${root:-pending}"
    echo "- Data Symlink: $(readlink -f "${REPO_ROOT}/data/scannet" 2>/dev/null || echo pending)"
    echo "- Official Gate Checkpoint: $( [ -f "$(checkpoint_path "${OFFICIAL_GATE_EXP}")" ] && echo ready || echo pending )"
    echo "- Continuation Trio: $( all_exist "${CONTINUE_EXPS[@]}" && echo ready || echo pending )"
    echo "- Linear Trio: $( all_exist "${LIN_EXPS[@]}" && echo ready || echo pending )"
    echo "- Follow-up: $( [ -f "$(checkpoint_path "${FOLLOWUP_NAME}")" ] && echo ready || echo pending )"
    echo
    echo "## Running Processes"
    pgrep -af 'setup_downstream_assets.sh|run_scannet_proxy.sh|scripts/train.sh|tar --skip-old-files|supervise_scannet_pipeline.sh' || true
  } > "${STATUS_PATH}"
}

log "supervisor start repo=${REPO_ROOT}"

while true; do
  stage="unknown"

  if ! scannet_ready; then
    stage="extract_scannet"
    if ! process_running 'setup_downstream_assets.sh' && ! process_running "tar --skip-old-files -xzf - -C ${SCANNET_EXTRACT_DIR}"; then
      start_bg \
        "setup_downstream_assets" \
        "DOWNLOAD_WEIGHTS=0 DOWNLOAD_SCANNET=0 EXTRACT_SCANNET=1 PYTHON_BIN='${PYTHON_BIN}' ionice -c3 nice -n 19 bash tools/concerto_projection_shortcut/setup_downstream_assets.sh"
    fi
    write_status "${stage}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  ensure_scannet_symlink

  if [ ! -f "$(checkpoint_path "${OFFICIAL_GATE_EXP}")" ]; then
    stage="official_gate"
    if ! process_running 'scannet-proxy-official-origin-lin'; then
      stash_incomplete_exp "${OFFICIAL_GATE_EXP}"
      start_bg \
        "scannet_gate" \
        "PYTHON_BIN='${PYTHON_BIN}' GATE_NUM_GPU='${GATE_NUM_GPU}' GATE_CONFIG='${GATE_CONFIG}' PARALLEL_SINGLE_GPU='${PARALLEL_SINGLE_GPU}' GPU_IDS_CSV='${GPU_IDS_CSV}' bash tools/concerto_projection_shortcut/run_scannet_proxy.sh gate"
    fi
    write_status "${stage}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  if ! all_exist "${CONTINUE_EXPS[@]}"; then
    stage="continuations"
    if ! process_running 'arkit-full-continue-concerto|arkit-full-continue-no-enc2d|arkit-full-continue-coord-mlp'; then
      local_exp=""
      for local_exp in "${CONTINUE_EXPS[@]}"; do
        if [ ! -f "$(checkpoint_path "${local_exp}")" ]; then
          stash_incomplete_exp "${local_exp}"
        fi
      done
      start_bg \
        "scannet_pretrain" \
        "PYTHON_BIN='${PYTHON_BIN}' PRETRAIN_NUM_GPU='${PRETRAIN_NUM_GPU}' PARALLEL_SINGLE_GPU='${PARALLEL_SINGLE_GPU}' GPU_IDS_CSV='${GPU_IDS_CSV}' bash tools/concerto_projection_shortcut/run_scannet_proxy.sh pretrain"
    fi
    write_status "${stage}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  if ! all_exist "${LIN_EXPS[@]}"; then
    stage="linear_probe"
    if ! process_running 'scannet-proxy-concerto-continue-lin|scannet-proxy-no-enc2d-continue-lin|scannet-proxy-coord-mlp-continue-lin'; then
      local_exp=""
      for local_exp in "${LIN_EXPS[@]}"; do
        if [ ! -f "$(checkpoint_path "${local_exp}")" ]; then
          stash_incomplete_exp "${local_exp}"
        fi
      done
      start_bg \
        "scannet_lin" \
        "PYTHON_BIN='${PYTHON_BIN}' LIN_NUM_GPU='${LIN_NUM_GPU}' LIN_CONFIG='${LIN_CONFIG}' PARALLEL_SINGLE_GPU='${PARALLEL_SINGLE_GPU}' GPU_IDS_CSV='${GPU_IDS_CSV}' bash tools/concerto_projection_shortcut/run_scannet_proxy.sh lin"
    fi
    write_status "${stage}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  if [ ! -f "$(checkpoint_path "${FOLLOWUP_NAME}")" ]; then
    stage="post_scannet_followup"
    if ! process_running "${FOLLOWUP_NAME}"; then
      stash_incomplete_exp "${FOLLOWUP_NAME}"
      start_bg \
        "post_scannet_followup" \
        "CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh -p '${PYTHON_BIN}' -d '${DATASET_NAME}' -g 1 -c '${FOLLOWUP_CONFIG}' -n '${FOLLOWUP_NAME}'"
    fi
    write_status "${stage}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  stage="done"
  if [ ! -f "${DONE_STAMP}" ]; then
    date '+%F %T %Z' > "${DONE_STAMP}"
    log "[done] wrote ${DONE_STAMP}"
  fi
  write_status "${stage}"
  break
done
