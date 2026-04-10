#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

DOWNLOAD_ARKIT="${DOWNLOAD_ARKIT:-1}"
EXTRACT_ARKIT="${EXTRACT_ARKIT:-1}"

mkdir -p "${ARKIT_COMPRESSED_DIR}" "${ARKIT_PARENT_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${REPO_ROOT}/data"

if [ "${DOWNLOAD_ARKIT}" = "1" ] && ! compgen -G "${ARKIT_COMPRESSED_DIR}/arkitscenes.tar.gz.part*" >/dev/null; then
  "${PYTHON_BIN}" - "${ARKIT_COMPRESSED_DIR}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Pointcept/concerto_arkitscenes_compressed",
    repo_type="dataset",
    local_dir=str(Path(sys.argv[1])),
    local_dir_use_symlinks=False,
)
PY
fi

if [ "${EXTRACT_ARKIT}" = "1" ] && [ ! -d "${ARKIT_FULL_SOURCE_ROOT}/images" ]; then
  cat "${ARKIT_COMPRESSED_DIR}"/arkitscenes.tar.gz.part* | tar --skip-old-files -xzf - -C "${ARKIT_PARENT_DIR}"
fi

if [ ! -d "${ARKIT_FULL_SOURCE_ROOT}/images" ]; then
  echo "[fail] ARKitScenes source root is missing: ${ARKIT_FULL_SOURCE_ROOT}" >&2
  exit 1
fi

"${PYTHON_BIN}" tools/concerto_projection_shortcut/prepare_arkit_full_splits.py \
  --source-root "${ARKIT_FULL_SOURCE_ROOT}" \
  --output-root "${ARKIT_FULL_META_ROOT}"

link_repo_data "${ARKIT_FULL_SOURCE_ROOT}" "arkitscenes"
link_repo_data "${ARKIT_FULL_META_ROOT}" "arkitscenes_absmeta"

echo "[ok] arkit source root: ${ARKIT_FULL_SOURCE_ROOT}"
echo "[ok] arkit meta root: ${ARKIT_FULL_META_ROOT}"
