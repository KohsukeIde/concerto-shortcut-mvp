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

DOWNLOAD_SCANNET_IMAGEPOINT="${DOWNLOAD_SCANNET_IMAGEPOINT:-1}"
EXTRACT_SCANNET_IMAGEPOINT="${EXTRACT_SCANNET_IMAGEPOINT:-1}"

mkdir -p "${SCANNET_COMPRESSED_DIR}" "${SCANNET_IMAGEPOINT_ROOT}" "${HF_HOME}" "${TORCH_HOME}"

if [ "${DOWNLOAD_SCANNET_IMAGEPOINT}" = "1" ]; then
  "${PYTHON_BIN}" - "${SCANNET_COMPRESSED_DIR}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(sys.argv[1])
snapshot_download(
    repo_id="Pointcept/concerto_scannet_compressed",
    repo_type="dataset",
    local_dir=str(target),
    local_dir_use_symlinks=False,
)
PY
fi

resolve_root() {
  local root="$1"
  if [ -f "${root}/splits/val.json" ] || [ -f "${root}/splits/train.json" ]; then
    printf '%s\n' "${root}"
    return 0
  fi
  if [ -f "${root}/scannet/splits/val.json" ] || [ -f "${root}/scannet/splits/train.json" ]; then
    printf '%s\n' "${root}/scannet"
    return 0
  fi
  find "${root}" -maxdepth 3 -type f \( -path '*/splits/val.json' -o -path '*/splits/train.json' \) \
    | sed 's#/splits/[^/]*$##' \
    | head -n 1
}

if [ "${EXTRACT_SCANNET_IMAGEPOINT}" = "1" ] && [ -z "$(resolve_root "${SCANNET_IMAGEPOINT_ROOT}")" ]; then
  shopt -s nullglob
  parts=(
    "${SCANNET_COMPRESSED_DIR}"/scannet.tar.gz.part*
    "${SCANNET_COMPRESSED_DIR}"/scannet.tar.gz.*
  )
  shopt -u nullglob
  if [ "${#parts[@]}" -eq 0 ]; then
    echo "[error] no ScanNet archive parts found in ${SCANNET_COMPRESSED_DIR}" >&2
    exit 2
  fi
  printf '%s\n' "${parts[@]}" | sort -u > "${SCANNET_IMAGEPOINT_ROOT}/.archive_parts"
  echo "[extract] concatenating $(wc -l < "${SCANNET_IMAGEPOINT_ROOT}/.archive_parts") parts into ${SCANNET_IMAGEPOINT_ROOT}"
  # shellcheck disable=SC2046
  cat $(cat "${SCANNET_IMAGEPOINT_ROOT}/.archive_parts") | tar --skip-old-files -xzf - -C "${SCANNET_IMAGEPOINT_ROOT}"
fi

ROOT="$(resolve_root "${SCANNET_IMAGEPOINT_ROOT}")"
if [ -z "${ROOT}" ]; then
  echo "[fail] Concerto ScanNet image-point root is not ready under ${SCANNET_IMAGEPOINT_ROOT}" >&2
  exit 1
fi

echo "[ok] Concerto ScanNet image-point root: ${ROOT}"
echo "[ok] split files:"
find "${ROOT}/splits" -maxdepth 1 -type f -name '*.json' -print | sort

"${PYTHON_BIN}" tools/concerto_projection_shortcut/prepare_scannet_imagepoint_splits.py \
  --source-root "${ROOT}" \
  --output-root "${SCANNET_IMAGEPOINT_META_ROOT}"

echo "[ok] Concerto ScanNet absmeta root: ${SCANNET_IMAGEPOINT_META_ROOT}"
