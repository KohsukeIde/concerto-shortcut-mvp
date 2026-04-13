#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

DATASET_NAME="${DATASET_NAME:-concerto}"
OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${REPO_ROOT}/weights/concerto/concerto_base_origin.pth}"
CONTINUE_CONFIG="${CONTINUE_CONFIG:-pretrain-concerto-v1m1-0-arkit-full-no-enc2d-renorm-continue-a1004}"
LINEAR_CONFIG="${LINEAR_CONFIG:-semseg-ptv3-base-v1m1-0a-scannet-lin-proxy}"
CONTINUE_EXP="${CONTINUE_EXP:-arkit-full-continue-no-enc2d-renorm}"
LINEAR_EXP="${LINEAR_EXP:-scannet-proxy-no-enc2d-renorm-continue-lin}"
CONTINUE_GPUS="${CONTINUE_GPUS:-1,2}"
LINEAR_GPU="${LINEAR_GPU:-3}"
LOG_DIR="${LOG_DIR:-tools/concerto_projection_shortcut/logs}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

device_count() {
  awk -F',' '{print NF}' <<< "$1"
}

checkpoint_path() {
  local exp_name="$1"
  printf '%s/exp/%s/%s/model/model_last.pth' "${REPO_ROOT}" "${DATASET_NAME}" "${exp_name}"
}

run_train() {
  local config_name="$1"
  local exp_name="$2"
  local weight_path="$3"
  local devices="$4"
  local gpu_count
  gpu_count="$(device_count "${devices}")"

  if [ -f "$(checkpoint_path "${exp_name}")" ]; then
    echo "[$(timestamp)] skip: ${exp_name} already has model_last.pth"
    return 0
  fi

  echo "[$(timestamp)] run: gpus=${devices} config=${config_name} exp=${exp_name}"
  CUDA_VISIBLE_DEVICES="${devices}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    bash "${REPO_ROOT}/scripts/train.sh" \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g "${gpu_count}" \
      -c "${config_name}" \
      -n "${exp_name}" \
      -w "${weight_path}"
}

ensure_conda_active
mkdir -p "${LOG_DIR}"

if [ ! -f "${OFFICIAL_WEIGHT}" ]; then
  echo "[$(timestamp)] error: missing official weight: ${OFFICIAL_WEIGHT}" >&2
  exit 1
fi

echo "[$(timestamp)] start no-enc2d-renorm chain"
echo "official_weight=${OFFICIAL_WEIGHT}"
echo "continue_config=${CONTINUE_CONFIG}"
echo "linear_config=${LINEAR_CONFIG}"
echo "continue_exp=${CONTINUE_EXP}"
echo "linear_exp=${LINEAR_EXP}"
echo "continue_gpus=${CONTINUE_GPUS}"
echo "linear_gpu=${LINEAR_GPU}"

run_train "${CONTINUE_CONFIG}" "${CONTINUE_EXP}" "${OFFICIAL_WEIGHT}" "${CONTINUE_GPUS}"
run_train "${LINEAR_CONFIG}" "${LINEAR_EXP}" "$(checkpoint_path "${CONTINUE_EXP}")" "${LINEAR_GPU}"

echo "[$(timestamp)] done no-enc2d-renorm chain"
