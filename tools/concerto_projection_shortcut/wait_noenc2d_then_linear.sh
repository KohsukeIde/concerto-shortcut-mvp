#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"

WATCH_PID="${1:-}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="${LOG_DIR:-tools/concerto_projection_shortcut/logs}"
CONTINUE_LOG="${CONTINUE_LOG:-${LOG_DIR}/arkit-full-continue-no-enc2d.launch.log}"
NOENC_CKPT="${NOENC_CKPT:-exp/concerto/arkit-full-continue-no-enc2d/model/model_last.pth}"
LINEAR_CKPT="${LINEAR_CKPT:-exp/concerto/scannet-proxy-no-enc2d-continue-lin/model/model_last.pth}"
CHAIN_LOG="${CHAIN_LOG:-${LOG_DIR}/noenc2d_linear_chain.log}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

is_noenc_process_running() {
  if [ -n "${WATCH_PID}" ]; then
    ps -p "${WATCH_PID}" -o args= 2>/dev/null | grep -q "arkit-full-continue-no-enc2d" && return 0
  fi
  ps -eo args= \
    | grep -E "[a]rkit-full-continue-no-enc2d" \
    | grep -E "scripts/train.sh|tools/train.py" >/dev/null 2>&1
}

is_noenc_complete() {
  [ -f "${NOENC_CKPT}" ] \
    && [ -f "${CONTINUE_LOG}" ] \
    && rg -q "Train: \\[5/5\\]\\[11238/11238\\]" "${CONTINUE_LOG}"
}

mkdir -p "${LOG_DIR}"
{
  echo "[$(timestamp)] watch start pid=${WATCH_PID:-auto} continue_log=${CONTINUE_LOG}"

  if [ -f "${LINEAR_CKPT}" ]; then
    echo "[$(timestamp)] skip: linear checkpoint already exists: ${LINEAR_CKPT}"
    exit 0
  fi

  while ! is_noenc_complete; do
    if ! is_noenc_process_running; then
      echo "[$(timestamp)] error: no-enc2d process ended before completion marker" >&2
      echo "[$(timestamp)] missing marker: Train: [5/5][11238/11238]" >&2
      exit 1
    fi
    echo "[$(timestamp)] waiting for no-enc2d continuation; next check in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done

  echo "[$(timestamp)] no-enc2d continuation complete; launching ScanNet linear"
  LINEAR_GPU_NOENC2D="${LINEAR_GPU_NOENC2D:-0}" \
    bash tools/concerto_projection_shortcut/run_scannet_gonogo_4gpu.sh lin
  echo "[$(timestamp)] ScanNet linear chain complete"
} >> "${CHAIN_LOG}" 2>&1
