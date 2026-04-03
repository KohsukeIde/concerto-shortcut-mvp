#!/usr/bin/env bash
set -euo pipefail

# Run from the Pointcept repo root.
# This is a bounded smoke run for the baseline probe. On most systems it will
# cover roughly the first 100-200 training iterations before timing out.

PYTHON_BIN=${PYTHON_BIN:-python3}
DATASET_NAME=${DATASET_NAME:-concerto}
GPUS=${GPUS:-1}
MACHINES=${MACHINES:-1}
CONFIG_NAME=${CONFIG_NAME:-pretrain-concerto-v1m1-0-probe-enc2d-baseline}
EXP_NAME=${EXP_NAME:-arkit-shortcut-smoke-baseline}
TIMEOUT_DURATION=${TIMEOUT_DURATION:-45m}

if ! command -v timeout >/dev/null 2>&1; then
  echo "timeout command not found; install coreutils or run the baseline manually." >&2
  exit 2
fi

set +e
timeout --signal=INT --kill-after=30s "${TIMEOUT_DURATION}" \
  bash scripts/train.sh \
    -p "${PYTHON_BIN}" \
    -m "${MACHINES}" \
    -g "${GPUS}" \
    -d "${DATASET_NAME}" \
    -c "${CONFIG_NAME}" \
    -n "${EXP_NAME}"
status=$?
set -e

if [ "${status}" -ne 0 ] && [ "${status}" -ne 124 ] && [ "${status}" -ne 130 ]; then
  exit "${status}"
fi

echo "smoke run finished with status ${status}"
"${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_logs.py \
  "exp/${DATASET_NAME}/${EXP_NAME}/train.log" || true
