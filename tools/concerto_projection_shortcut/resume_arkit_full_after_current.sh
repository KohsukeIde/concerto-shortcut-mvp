#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

POLL_SECONDS="${POLL_SECONDS:-300}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_NAME="${DATASET_NAME:-concerto}"
EXP_PREFIX="${EXP_PREFIX:-arkit-full-causal}"
CONDA_ROOT="${CONDA_ROOT:-/home/cvrt/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointcept-cu128}"

if ! command -v conda >/dev/null 2>&1; then
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

BASELINE_CKPT="exp/${DATASET_NAME}/${EXP_PREFIX}-baseline/model/model_last.pth"
COORD_MLP_CKPT="exp/${DATASET_NAME}/${EXP_PREFIX}-coord-mlp/model/model_last.pth"

echo "[wait] baseline checkpoint: ${BASELINE_CKPT}"
echo "[wait] coord-mlp checkpoint: ${COORD_MLP_CKPT}"
while [ ! -f "${BASELINE_CKPT}" ] || [ ! -f "${COORD_MLP_CKPT}" ]; do
  date '+[wait] %F %T still waiting for initial pair'
  sleep "${POLL_SECONDS}"
done

echo "[resume] initial pair finished, launching the remaining ARKit full causal runs"
START_AT_INDEX=2 \
INCLUDE_FIX=1 \
RUN_STRESS=1 \
PYTHON_BIN="${PYTHON_BIN}" \
bash tools/concerto_projection_shortcut/run_arkit_full_causal.sh
