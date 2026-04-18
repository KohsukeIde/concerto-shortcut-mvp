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

DOWNLOAD_MISSING="${DOWNLOAD_MISSING:-1}"
EXTRACT_MISSING="${EXTRACT_MISSING:-1}"
VERIFY_REWRITTEN="${VERIFY_REWRITTEN:-1}"
DATASET_FILTER="${DATASET_FILTER:-arkit,scannet,scannetpp,s3dis,hm3d,structured3d}"

contains_dataset() {
  local needle="$1"
  case ",${DATASET_FILTER}," in
    *",${needle},"*) return 0 ;;
    *) return 1 ;;
  esac
}

download_dataset() {
  local repo_id="$1"
  local target="$2"
  mkdir -p "${target}"
  if [ "${DOWNLOAD_MISSING}" != "1" ]; then
    echo "[skip] download disabled: ${repo_id}"
    return 0
  fi
  echo "[download] ${repo_id} -> ${target}"
  "${PYTHON_BIN}" - "${repo_id}" "${target}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, target = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=target,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
}

resolve_root() {
  local root="$1"
  if [ -d "${root}/splits" ]; then
    printf '%s\n' "${root}"
    return 0
  fi
  find "${root}" -maxdepth 3 -type f -path '*/splits/*.json' \
    | sed 's#/splits/[^/]*$##' \
    | head -n 1
}

extract_dataset() {
  local dataset="$1"
  local compressed="$2"
  local target="$3"
  local extract_threads
  mkdir -p "${target}"
  if [ "${EXTRACT_MISSING}" != "1" ]; then
    echo "[skip] extract disabled: ${dataset}"
    return 0
  fi
  if [ -n "$(resolve_root "${target}")" ]; then
    echo "[ok] ${dataset} already extracted: $(resolve_root "${target}")"
    return 0
  fi
  shopt -s nullglob
  local parts=(
    "${compressed}/${dataset}.tar.gz.part"*
    "${compressed}/${dataset}.tar.gz."*
    "${compressed}"/*.tar.gz.part*
    "${compressed}"/*.tar.gz.*
  )
  shopt -u nullglob
  if [ "${#parts[@]}" -eq 0 ]; then
    echo "[error] no archive parts for ${dataset} under ${compressed}" >&2
    return 2
  fi
  printf '%s\n' "${parts[@]}" | sort -u > "${target}/.archive_parts"
  echo "[extract] ${dataset}: $(wc -l < "${target}/.archive_parts") parts -> ${target}"
  extract_threads="${EXTRACT_THREADS:-$(nproc)}"
  if command -v pigz >/dev/null 2>&1; then
    echo "[extract] ${dataset}: using pigz with ${extract_threads} threads"
    # shellcheck disable=SC2046
    cat $(cat "${target}/.archive_parts") | pigz -dc -p "${extract_threads}" | tar --skip-old-files -xf - -C "${target}"
  else
    echo "[extract] ${dataset}: pigz not found, falling back to single-thread gzip"
    # shellcheck disable=SC2046
    cat $(cat "${target}/.archive_parts") | tar --skip-old-files -xzf - -C "${target}"
  fi
}

rewrite_dataset() {
  local dataset="$1"
  local source="$2"
  local meta="$3"
  local verify_flag=()
  if [ "${VERIFY_REWRITTEN}" = "1" ]; then
    verify_flag=(--verify)
  fi
  if [ -z "$(resolve_root "${source}")" ]; then
    echo "[missing] ${dataset}: source root not ready under ${source}" >&2
    return 3
  fi
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/prepare_concerto_imagepoint_splits.py \
    --dataset "${dataset}" \
    --source-root "$(resolve_root "${source}")" \
    --output-root "${meta}" \
    "${verify_flag[@]}"
}

if contains_dataset arkit; then
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/prepare_arkit_full_splits.py \
    --source-root "${ARKIT_FULL_SOURCE_ROOT}" \
    --output-root "${ARKIT_FULL_META_ROOT}"
fi

if contains_dataset scannet; then
  if [ ! -d "${SCANNET_IMAGEPOINT_META_ROOT}/splits" ]; then
    download_dataset "Pointcept/concerto_scannet_compressed" "${SCANNET_COMPRESSED_DIR}"
    extract_dataset "scannet" "${SCANNET_COMPRESSED_DIR}" "${SCANNET_IMAGEPOINT_ROOT}"
    rewrite_dataset "scannet" "${SCANNET_IMAGEPOINT_ROOT}" "${SCANNET_IMAGEPOINT_META_ROOT}"
  else
    echo "[ok] scannet absmeta exists: ${SCANNET_IMAGEPOINT_META_ROOT}"
  fi
fi

if contains_dataset scannetpp; then
  download_dataset "Pointcept/concerto_scannetpp_compressed" "${SCANNETPP_COMPRESSED_DIR}"
  extract_dataset "scannetpp" "${SCANNETPP_COMPRESSED_DIR}" "${SCANNETPP_IMAGEPOINT_ROOT}"
  rewrite_dataset "scannetpp" "${SCANNETPP_IMAGEPOINT_ROOT}" "${SCANNETPP_IMAGEPOINT_META_ROOT}"
fi

if contains_dataset s3dis; then
  download_dataset "Pointcept/concerto_s3dis_compressed" "${S3DIS_COMPRESSED_DIR}"
  extract_dataset "s3dis" "${S3DIS_COMPRESSED_DIR}" "${S3DIS_IMAGEPOINT_ROOT}"
  rewrite_dataset "s3dis" "${S3DIS_IMAGEPOINT_ROOT}" "${S3DIS_IMAGEPOINT_META_ROOT}"
fi

if contains_dataset hm3d; then
  download_dataset "Pointcept/concerto_hm3d_compressed" "${HM3D_COMPRESSED_DIR}"
  extract_dataset "hm3d" "${HM3D_COMPRESSED_DIR}" "${HM3D_IMAGEPOINT_ROOT}"
  rewrite_dataset "hm3d" "${HM3D_IMAGEPOINT_ROOT}" "${HM3D_IMAGEPOINT_META_ROOT}"
fi

if contains_dataset structured3d; then
  download_dataset "Pointcept/concerto_structured3d_compressed" "${STRUCTURED3D_COMPRESSED_DIR}"
  extract_dataset "structured3d" "${STRUCTURED3D_COMPRESSED_DIR}" "${STRUCTURED3D_IMAGEPOINT_ROOT}"
  rewrite_dataset "structured3d" "${STRUCTURED3D_IMAGEPOINT_ROOT}" "${STRUCTURED3D_IMAGEPOINT_META_ROOT}"
fi

VERIFY_ARGS=()
if [ -n "${ALLOW_MISSING_VERIFY:-}" ]; then
  VERIFY_ARGS+=(--allow-missing)
elif [ "${DATASET_FILTER}" != "arkit,scannet,scannetpp,s3dis,hm3d,structured3d" ]; then
  VERIFY_ARGS+=(--allow-missing)
fi

"${PYTHON_BIN}" tools/concerto_projection_shortcut/verify_concerto_six_datasets.py \
  "${VERIFY_ARGS[@]}"
