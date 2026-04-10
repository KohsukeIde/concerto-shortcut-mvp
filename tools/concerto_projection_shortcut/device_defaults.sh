#!/usr/bin/env bash
# shellcheck shell=bash

if [ -z "${REPO_ROOT:-}" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi

default_conda_root() {
  local base=""
  if [ -n "${CONDA_EXE:-}" ]; then
    base="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd -P)"
  elif command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null || true)"
  fi
  if [ -n "${base}" ]; then
    printf '%s\n' "${base}"
    return 0
  fi
  for candidate in "${HOME}/anaconda3" "${HOME}/miniconda3"; do
    if [ -d "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${HOME}/anaconda3"
}

default_python_bin() {
  local candidate=""
  for candidate in \
    "${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin/python3.10" \
    "${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin/python" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"; do
    if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' python3
}

ensure_conda_active() {
  if ! command -v conda >/dev/null 2>&1 || [ "$(type -t conda)" != "function" ]; then
    set +u
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    set -u
  fi
  if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]; then
    set +u
    conda activate "${CONDA_ENV_NAME}"
    set -u
  fi
}

link_repo_data() {
  local target="$1"
  local name="$2"
  local link_path="${REPO_ROOT}/data/${name}"
  mkdir -p "${REPO_ROOT}/data"
  if [ -e "${link_path}" ] && [ ! -L "${link_path}" ]; then
    echo "[warn] keep existing non-symlink ${link_path}" >&2
    return 0
  fi
  ln -sfn "${target}" "${link_path}"
  echo "[ok] symlink: ${link_path} -> $(readlink -f "${link_path}")"
}

CONDA_ROOT="${CONDA_ROOT:-$(default_conda_root)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-${POINTCEPT_CONDA_ENV:-pointcept-concerto-cu121}}"
POINTCEPT_DATA_ROOT="${POINTCEPT_DATA_ROOT:-/mnt/urashima/users/minesawa/pointcept_data}"
ARKIT_COMPRESSED_DIR="${ARKIT_COMPRESSED_DIR:-${POINTCEPT_DATA_ROOT}/concerto_arkitscenes_compressed}"
ARKIT_PARENT_DIR="${ARKIT_PARENT_DIR:-${POINTCEPT_DATA_ROOT}/arkitscenes}"
ARKIT_FULL_SOURCE_ROOT="${ARKIT_FULL_SOURCE_ROOT:-${ARKIT_PARENT_DIR}/arkitscenes}"
ARKIT_FULL_META_ROOT="${ARKIT_FULL_META_ROOT:-${ARKIT_PARENT_DIR}/arkitscenes_absmeta}"
DATA_ROOT="${DATA_ROOT:-${POINTCEPT_DATA_ROOT}}"
SCANNET_COMPRESSED_DIR="${SCANNET_COMPRESSED_DIR:-${DATA_ROOT}/concerto_scannet_compressed}"
SCANNET_EXTRACT_DIR="${SCANNET_EXTRACT_DIR:-${DATA_ROOT}/scannet}"
WEIGHT_DIR="${WEIGHT_DIR:-${REPO_ROOT}/weights/concerto}"
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
export HF_HOME
export HF_HUB_CACHE
export HUGGINGFACE_HUB_CACHE
export HF_XET_CACHE
export TORCH_HOME
