#!/usr/bin/env bash
set -euo pipefail

# Run from the Pointcept repo root.
# Example:
#   bash tools/concerto_projection_shortcut/run_mvp.sh

PYTHON_BIN=${PYTHON_BIN:-python3}
DATASET_NAME=${DATASET_NAME:-concerto}
GPUS=${GPUS:-1}
MACHINES=${MACHINES:-1}
EXP_PREFIX=${EXP_PREFIX:-arkit-shortcut-}
TIMEOUT_DURATION=${TIMEOUT_DURATION:-}

run() {
  local config="$1"
  local name="$2"
  echo "[run] $name"
  if [ -n "${TIMEOUT_DURATION}" ]; then
    set +e
    timeout --signal=INT --kill-after=30s "${TIMEOUT_DURATION}" \
      bash scripts/train.sh \
        -p "${PYTHON_BIN}" \
        -m "${MACHINES}" \
        -g "${GPUS}" \
        -d "${DATASET_NAME}" \
        -c "${config}" \
        -n "${name}"
    status=$?
    set -e
    if [ "${status}" -ne 0 ] && [ "${status}" -ne 124 ] && [ "${status}" -ne 130 ]; then
      return "${status}"
    fi
    echo "[run] ${name} finished with status ${status}"
    return 0
  fi

  bash scripts/train.sh \
    -p "${PYTHON_BIN}" \
    -m "${MACHINES}" \
    -g "${GPUS}" \
    -d "${DATASET_NAME}" \
    -c "${config}" \
    -n "${name}"
}

run pretrain-concerto-v1m1-0-probe-enc2d-baseline "${EXP_PREFIX}baseline-enc2d"
run pretrain-concerto-v1m1-0-probe-enc2d-zero-appearance "${EXP_PREFIX}zero-appearance-enc2d"
run pretrain-concerto-v1m1-0-probe-enc2d-coord-mlp "${EXP_PREFIX}coord-mlp-enc2d"
run pretrain-concerto-v1m1-0-probe-enc2d-jitter "${EXP_PREFIX}jitter-enc2d"
run pretrain-concerto-v1m1-0-probe-enc2d-cross-scene-target-swap "${EXP_PREFIX}cross-scene-target-swap-enc2d"

"${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_logs.py \
  "exp/${DATASET_NAME}/${EXP_PREFIX}*/train.log" || true
