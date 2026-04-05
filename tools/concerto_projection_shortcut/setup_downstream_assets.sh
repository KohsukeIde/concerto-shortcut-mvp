#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/cvrt/datasets}"
SCANNET_COMPRESSED_DIR="${SCANNET_COMPRESSED_DIR:-${DATA_ROOT}/concerto_scannet_compressed}"
SCANNET_EXTRACT_DIR="${SCANNET_EXTRACT_DIR:-${DATA_ROOT}/scannet}"
WEIGHT_DIR="${WEIGHT_DIR:-${REPO_ROOT}/weights/concerto}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DOWNLOAD_SCANNET="${DOWNLOAD_SCANNET:-1}"
EXTRACT_SCANNET="${EXTRACT_SCANNET:-1}"

mkdir -p "${SCANNET_COMPRESSED_DIR}" "${SCANNET_EXTRACT_DIR}" "${WEIGHT_DIR}" "${REPO_ROOT}/data"

"${PYTHON_BIN}" - "${SCANNET_COMPRESSED_DIR}" "${WEIGHT_DIR}" "${DOWNLOAD_WEIGHTS}" "${DOWNLOAD_SCANNET}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

compressed_dir = Path(sys.argv[1])
weight_dir = Path(sys.argv[2])
download_weights = sys.argv[3] == "1"
download_scannet = sys.argv[4] == "1"

if download_weights:
    for filename in ("concerto_base.pth", "concerto_base_origin.pth"):
        hf_hub_download(
            repo_id="Pointcept/Concerto",
            filename=filename,
            repo_type="model",
            local_dir=str(weight_dir),
            local_dir_use_symlinks=False,
        )

if download_scannet:
    snapshot_download(
        repo_id="Pointcept/concerto_scannet_compressed",
        repo_type="dataset",
        local_dir=str(compressed_dir),
        local_dir_use_symlinks=False,
    )
PY

if [ "${EXTRACT_SCANNET}" = "1" ] && [ ! -d "${SCANNET_EXTRACT_DIR}/splits" ] && [ ! -d "${SCANNET_EXTRACT_DIR}/scannet/splits" ]; then
  cat "${SCANNET_COMPRESSED_DIR}"/scannet.tar.gz.part_* | tar -xzf - -C "${SCANNET_EXTRACT_DIR}"
fi

if [ -d "${SCANNET_EXTRACT_DIR}/splits" ]; then
  SCANNET_ROOT="${SCANNET_EXTRACT_DIR}"
elif [ -d "${SCANNET_EXTRACT_DIR}/scannet/splits" ]; then
  SCANNET_ROOT="${SCANNET_EXTRACT_DIR}/scannet"
else
  SCANNET_ROOT=""
fi

if [ -n "${SCANNET_ROOT}" ]; then
  ln -sfn "${SCANNET_ROOT}" "${REPO_ROOT}/data/scannet"
fi

echo "[ok] weights:"
ls -1 "${WEIGHT_DIR}"
if [ -n "${SCANNET_ROOT}" ]; then
  echo "[ok] scannet root: ${SCANNET_ROOT}"
  echo "[ok] symlink: ${REPO_ROOT}/data/scannet -> $(readlink -f "${REPO_ROOT}/data/scannet")"
elif [ "${DOWNLOAD_SCANNET}" = "1" ] && [ "${EXTRACT_SCANNET}" = "0" ]; then
  echo "[ok] ScanNet compressed snapshot downloaded to: ${SCANNET_COMPRESSED_DIR}"
  echo "[info] Extraction skipped because EXTRACT_SCANNET=0"
elif [ "${DOWNLOAD_SCANNET}" = "0" ] && [ "${EXTRACT_SCANNET}" = "1" ]; then
  echo "[info] Extraction requested, but ScanNet root is still not prepared"
else
  echo "[info] ScanNet dataset not prepared yet"
fi
