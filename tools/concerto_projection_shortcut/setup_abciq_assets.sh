#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

SCANNET_SOURCE="${SCANNET_SOURCE:-/groups/qgah50055/ide/3d-sans-3dscans/scannet}"
LINK_SCANNET="${LINK_SCANNET:-1}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DOWNLOAD_ARKIT="${DOWNLOAD_ARKIT:-1}"
EXTRACT_ARKIT="${EXTRACT_ARKIT:-1}"
CACHE_DINO="${CACHE_DINO:-1}"
DINO_REPO_ID="${DINO_REPO_ID:-facebook/dinov2-with-registers-giant}"

mkdir -p "${POINTCEPT_DATA_ROOT}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TORCH_HOME}"

if [ -f "${VENV_ACTIVATE}" ]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
  PYTHON_BIN="$(command -v python)"
fi

if [ "${LINK_SCANNET}" = "1" ]; then
  if [ ! -d "${SCANNET_SOURCE}" ]; then
    echo "[fail] ScanNet source is missing: ${SCANNET_SOURCE}" >&2
    exit 1
  fi
  link_repo_data "${SCANNET_SOURCE}" "scannet"
fi

DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS}" \
DOWNLOAD_SCANNET=0 \
EXTRACT_SCANNET=0 \
bash tools/concerto_projection_shortcut/setup_downstream_assets.sh

DOWNLOAD_ARKIT="${DOWNLOAD_ARKIT}" \
EXTRACT_ARKIT="${EXTRACT_ARKIT}" \
bash tools/concerto_projection_shortcut/setup_arkit_full_assets.sh

if [ "${CACHE_DINO}" = "1" ]; then
  "${PYTHON_BIN}" - "${DINO_REPO_ID}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], repo_type="model")
PY
fi

bash tools/concerto_projection_shortcut/check_setup_status.sh
