#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

NUM_GPU="${NUM_GPU:-2}"
DATASET_NAME="${DATASET_NAME:-concerto}"
OFFICIAL_GATE_EXP="${OFFICIAL_GATE_EXP:-scannet-proxy-official-origin-lin}"
OFFICIAL_GATE_DIR="${REPO_ROOT}/exp/${DATASET_NAME}/${OFFICIAL_GATE_EXP}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/tools/concerto_projection_shortcut/logs/scannet_remaining_chain.log}"

ensure_conda_active

mkdir -p "$(dirname "${LOG_PATH}")"

{
  stamp="$(date +%Y%m%d-%H%M%S)"
  if [ -d "${OFFICIAL_GATE_DIR}" ] && [ ! -f "${OFFICIAL_GATE_DIR}/model/model_last.pth" ]; then
    mv "${OFFICIAL_GATE_DIR}" "${OFFICIAL_GATE_DIR}-stale-${stamp}"
    echo "[move] ${OFFICIAL_GATE_DIR} -> ${OFFICIAL_GATE_DIR}-stale-${stamp}"
  fi

  DOWNLOAD_WEIGHTS=0 DOWNLOAD_SCANNET=0 EXTRACT_SCANNET=1 PYTHON_BIN="${PYTHON_BIN}" \
    bash tools/concerto_projection_shortcut/setup_downstream_assets.sh

  PYTHON_BIN="${PYTHON_BIN}" NUM_GPU="${NUM_GPU}" \
    bash tools/concerto_projection_shortcut/run_scannet_proxy.sh all
} | tee -a "${LOG_PATH}"
