#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ARKIT_COMPRESSED_DIR="${ARKIT_COMPRESSED_DIR:-/home/cvrt/datasets/concerto_arkitscenes_compressed}"
ARKIT_PARENT_DIR="${ARKIT_PARENT_DIR:-/home/cvrt/datasets/arkitscenes}"
ARKIT_FULL_SOURCE_ROOT="${ARKIT_FULL_SOURCE_ROOT:-${ARKIT_PARENT_DIR}/arkitscenes}"
ARKIT_FULL_META_ROOT="${ARKIT_FULL_META_ROOT:-${ARKIT_PARENT_DIR}/arkitscenes_absmeta}"

mkdir -p "${ARKIT_PARENT_DIR}"

if [ ! -d "${ARKIT_FULL_SOURCE_ROOT}/images" ]; then
  cat "${ARKIT_COMPRESSED_DIR}"/arkitscenes.tar.gz.part* | tar -xzf - -C "${ARKIT_PARENT_DIR}"
fi

"${PYTHON_BIN}" tools/concerto_projection_shortcut/prepare_arkit_full_splits.py \
  --source-root "${ARKIT_FULL_SOURCE_ROOT}" \
  --output-root "${ARKIT_FULL_META_ROOT}"

echo "[ok] arkit source root: ${ARKIT_FULL_SOURCE_ROOT}"
echo "[ok] arkit meta root: ${ARKIT_FULL_META_ROOT}"
