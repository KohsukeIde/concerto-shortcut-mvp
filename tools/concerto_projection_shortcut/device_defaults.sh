#!/usr/bin/env bash
# shellcheck shell=bash

if [ -z "${REPO_ROOT:-}" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi

default_python_bin() {
  local candidate=""
  for candidate in \
    "${VENV_DIR}/bin/python" \
    "${VENV_DIR}/bin/python3" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"; do
    if [ -n "${candidate}" ] && [ -x "${candidate}" ] && "${candidate}" -c 'import sys' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' python3
}

ensure_venv_active() {
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "${VENV_DIR}" ] && [ "$(cd "${VIRTUAL_ENV}" && pwd -P)" = "$(cd "${VENV_DIR}" && pwd -P)" ]; then
    return 0
  fi
  if [ ! -f "${VENV_ACTIVATE}" ]; then
    echo "[error] venv activate script not found: ${VENV_ACTIVATE}" >&2
    echo "        create it with tools/concerto_projection_shortcut/create_venv_abciq.sh" >&2
    return 2
  fi
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
}

# Backward-compatible name for older local helpers. This project now uses venv,
# not conda, on ABCI-Q.
ensure_conda_active() {
  ensure_venv_active
}

link_repo_data() {
  local target="$1"
  local name="$2"
  local link_path="${REPO_ROOT}/data/${name}"
  local target_abs
  local link_abs
  mkdir -p "${REPO_ROOT}/data"
  target_abs="$(cd "$(dirname "${target}")" && pwd -P)/$(basename "${target}")"
  link_abs="$(cd "$(dirname "${link_path}")" && pwd -P)/$(basename "${link_path}")"
  if [ "${target_abs}" = "${link_abs}" ]; then
    echo "[ok] ${link_path} already is the requested target"
    return 0
  fi
  if [ -e "${link_path}" ] && [ ! -L "${link_path}" ]; then
    echo "[warn] keep existing non-symlink ${link_path}" >&2
    return 0
  fi
  if [ -L "${link_path}" ]; then
    rm -f "${link_path}"
  fi
  ln -s "${target}" "${link_path}"
  echo "[ok] symlink: ${link_path} -> $(readlink -f "${link_path}")"
}

POINTCEPT_DATA_ROOT="${POINTCEPT_DATA_ROOT:-${REPO_ROOT}/data}"
VENV_DIR="${VENV_DIR:-${POINTCEPT_DATA_ROOT}/venv/pointcept-concerto-py311-cu124}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${VENV_DIR}/bin/activate}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
ARKIT_COMPRESSED_DIR="${ARKIT_COMPRESSED_DIR:-${POINTCEPT_DATA_ROOT}/concerto_arkitscenes_compressed}"
ARKIT_PARENT_DIR="${ARKIT_PARENT_DIR:-${POINTCEPT_DATA_ROOT}}"
ARKIT_FULL_SOURCE_ROOT="${ARKIT_FULL_SOURCE_ROOT:-${ARKIT_PARENT_DIR}/arkitscenes}"
ARKIT_FULL_META_ROOT="${ARKIT_FULL_META_ROOT:-${ARKIT_PARENT_DIR}/arkitscenes_absmeta}"
DATA_ROOT="${DATA_ROOT:-${POINTCEPT_DATA_ROOT}}"
SCANNET_COMPRESSED_DIR="${SCANNET_COMPRESSED_DIR:-${DATA_ROOT}/concerto_scannet_compressed}"
SCANNET_EXTRACT_DIR="${SCANNET_EXTRACT_DIR:-${DATA_ROOT}/scannet}"
WEIGHT_DIR="${WEIGHT_DIR:-${POINTCEPT_DATA_ROOT}/weights/concerto}"
POINTCEPT_HF_HOME="${POINTCEPT_HF_HOME:-${POINTCEPT_DATA_ROOT}/hf-home}"
POINTCEPT_HF_HUB_CACHE="${POINTCEPT_HF_HUB_CACHE:-${POINTCEPT_HF_HOME}/hub}"
POINTCEPT_HF_XET_CACHE="${POINTCEPT_HF_XET_CACHE:-${POINTCEPT_HF_HOME}/xet}"
POINTCEPT_TORCH_HOME="${POINTCEPT_TORCH_HOME:-${POINTCEPT_DATA_ROOT}/torch-home}"
HF_HOME="${POINTCEPT_HF_HOME}"
HF_HUB_CACHE="${POINTCEPT_HF_HUB_CACHE}"
HUGGINGFACE_HUB_CACHE="${POINTCEPT_HF_HUB_CACHE}"
HF_XET_CACHE="${POINTCEPT_HF_XET_CACHE}"
TORCH_HOME="${POINTCEPT_TORCH_HOME}"
PYTHON_BIN="${PYTHON_BIN:-$(default_python_bin)}"

export POINTCEPT_DATA_ROOT
export VENV_DIR
export VENV_ACTIVATE
export PYTHON_MODULE
export CUDA_MODULE
export HF_HOME
export HF_HUB_CACHE
export HUGGINGFACE_HUB_CACHE
export HF_XET_CACHE
export TORCH_HOME
