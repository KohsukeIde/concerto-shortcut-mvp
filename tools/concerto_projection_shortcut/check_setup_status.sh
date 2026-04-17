#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi
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
print_path_status "scannet image-point root" "${SCANNET_IMAGEPOINT_ROOT}"
print_path_status "scannet image-point meta root" "${SCANNET_IMAGEPOINT_META_ROOT}"
print_path_status "scannet++ image-point meta root" "${SCANNETPP_IMAGEPOINT_META_ROOT}"
print_path_status "s3dis image-point meta root" "${S3DIS_IMAGEPOINT_META_ROOT}"
print_path_status "hm3d image-point meta root" "${HM3D_IMAGEPOINT_META_ROOT}"
print_path_status "structured3d image-point meta root" "${STRUCTURED3D_IMAGEPOINT_META_ROOT}"
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
pgrep -af 'setup_arkit_full_assets.sh|setup_downstream_assets.sh|setup_concerto_six_imagepoint.sh|create_venv_abciq.sh|tar --skip-old-files -xzf' || true
