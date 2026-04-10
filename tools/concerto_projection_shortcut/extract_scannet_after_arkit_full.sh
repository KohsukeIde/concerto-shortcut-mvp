#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

POLL_SECONDS="${POLL_SECONDS:-300}"
DONE_STAMP="${DONE_STAMP:-${REPO_ROOT}/tools/concerto_projection_shortcut/arkit_full_causal.done}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/tools/concerto_projection_shortcut/logs/scannet_extract.log}"

ensure_conda_active

mkdir -p "$(dirname "${LOG_PATH}")"

echo "[wait] ARKit full completion stamp: ${DONE_STAMP}"
while [ ! -f "${DONE_STAMP}" ]; do
  date '+[wait] %F %T waiting for ARKit full causal completion'
  sleep "${POLL_SECONDS}"
done

echo "[extract] starting ScanNet extraction after ARKit full completion"
ionice -c3 nice -n 19 bash -lc \
  "cd '${REPO_ROOT}' && DOWNLOAD_WEIGHTS=0 DOWNLOAD_SCANNET=0 EXTRACT_SCANNET=1 PYTHON_BIN='${PYTHON_BIN}' bash tools/concerto_projection_shortcut/setup_downstream_assets.sh" \
  | tee -a "${LOG_PATH}"
