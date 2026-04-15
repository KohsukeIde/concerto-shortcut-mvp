#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

print_path_status() {
  local label="$1"
  local path="$2"
  if [ -L "${path}" ]; then
    echo "[link] ${label}: ${path} -> $(readlink -f "${path}")"
  elif [ -e "${path}" ]; then
    echo "[ok] ${label}: ${path}"
  else
    echo "[missing] ${label}: ${path}"
  fi
}

count_incomplete() {
  local root="$1"
  find "${root}" -type f -name '*.incomplete' 2>/dev/null | wc -l
}

echo "[env] venv: ${VENV_DIR}"
echo "[env] python: ${PYTHON_BIN}"
echo "[env] python module: ${PYTHON_MODULE}"
echo "[env] cuda module: ${CUDA_MODULE}"
echo "[env] HF_HOME: ${HF_HOME}"
echo "[env] HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "[env] HF_XET_CACHE: ${HF_XET_CACHE}"
echo "[env] TORCH_HOME: ${TORCH_HOME}"

print_path_status "weights dir" "${WEIGHT_DIR}"
print_path_status "arkit compressed" "${ARKIT_COMPRESSED_DIR}"
print_path_status "arkit source root" "${ARKIT_FULL_SOURCE_ROOT}"
print_path_status "arkit meta root" "${ARKIT_FULL_META_ROOT}"
print_path_status "scannet compressed" "${SCANNET_COMPRESSED_DIR}"
print_path_status "scannet extract root" "${SCANNET_EXTRACT_DIR}"
print_path_status "repo data/arkitscenes" "${REPO_ROOT}/data/arkitscenes"
print_path_status "repo data/arkitscenes_absmeta" "${REPO_ROOT}/data/arkitscenes_absmeta"
print_path_status "repo data/scannet" "${REPO_ROOT}/data/scannet"

echo "[status] arkit incomplete files: $(count_incomplete "${ARKIT_COMPRESSED_DIR}")"
echo "[status] scannet incomplete files: $(count_incomplete "${SCANNET_COMPRESSED_DIR}")"

if [ -d "${ARKIT_FULL_SOURCE_ROOT}/images" ] && [ -d "${ARKIT_FULL_META_ROOT}/splits" ]; then
  echo "[ready] ARKitScenes appears prepared"
else
  echo "[info] ARKitScenes is not fully prepared yet"
fi

if [ -d "${SCANNET_EXTRACT_DIR}/train" ] && [ -d "${SCANNET_EXTRACT_DIR}/val" ]; then
  echo "[ready] ScanNet appears prepared"
elif [ -d "${SCANNET_EXTRACT_DIR}/splits" ] || [ -d "${SCANNET_EXTRACT_DIR}/scannet/splits" ]; then
  echo "[ready] ScanNet appears prepared"
else
  echo "[info] ScanNet is not fully prepared yet"
fi

echo "[proc] active setup processes:"
pgrep -af 'setup_arkit_full_assets.sh|setup_downstream_assets.sh|create_venv_abciq.sh|tar --skip-old-files -xzf' || true
