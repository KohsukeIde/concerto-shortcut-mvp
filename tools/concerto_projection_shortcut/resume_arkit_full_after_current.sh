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
BASELINE_PATTERN="pretrain-concerto-v1m1-0-probe-enc2d-full-baseline -n ${EXP_PREFIX}-baseline"
COORD_MLP_PATTERN="pretrain-concerto-v1m1-0-probe-enc2d-full-coord-mlp -n ${EXP_PREFIX}-coord-mlp"

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

archive_stale_run() {
  local exp_name="$1"
  local exp_dir="exp/${DATASET_NAME}/${exp_name}"
  local checkpoint="${exp_dir}/model/model_last.pth"
  if [ ! -d "${exp_dir}" ] || [ -f "${checkpoint}" ]; then
    return 0
  fi
  if pgrep -af "${exp_name}" >/dev/null 2>&1; then
    return 0
  fi
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local archived="${exp_dir}-stale-${stamp}"
  echo "[archive] ${exp_dir} -> ${archived}"
  mv "${exp_dir}" "${archived}"
}

echo "[wait] baseline process pattern: ${BASELINE_PATTERN}"
echo "[wait] coord-mlp process pattern: ${COORD_MLP_PATTERN}"
while pgrep -af "${BASELINE_PATTERN}" >/dev/null 2>&1 || pgrep -af "${COORD_MLP_PATTERN}" >/dev/null 2>&1; do
  date '+[wait] %F %T still waiting for initial pair processes to exit'
  sleep "${POLL_SECONDS}"
done

archive_stale_run "${EXP_PREFIX}-global-target-permutation"
archive_stale_run "${EXP_PREFIX}-cross-image-target-swap"
archive_stale_run "${EXP_PREFIX}-cross-scene-target-swap"
archive_stale_run "${EXP_PREFIX}-coord-residual-target"

echo "[resume] initial pair exited, launching the remaining ARKit full causal runs"
START_AT_INDEX=2 \
INCLUDE_FIX=1 \
RUN_STRESS=1 \
PYTHON_BIN="${PYTHON_BIN}" \
bash tools/concerto_projection_shortcut/run_arkit_full_causal.sh
