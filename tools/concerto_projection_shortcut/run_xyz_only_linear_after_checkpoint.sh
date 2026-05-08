#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

CKPT="${CKPT:-exp/concerto/arkit-full-continue-xyz-only-input-seed0-gpu123/model/model_last.pth}"
EXP_PREFIX="${EXP_PREFIX:-scannet-proxy-xyz-only-input-continue-lin-seed}"
LIN_CONFIG="${LIN_CONFIG:-configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin-proxy.py}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
GPU_IDS_CSV="${GPU_IDS_CSV:-1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/envs/pointcept-concerto-cu121/bin/python}"
LOG_DIR="${LOG_DIR:-logs/xyz_only_input_continue}"
LOG="${LOG_DIR}/linear_driver.log"

mkdir -p "${LOG_DIR}"
echo "[watcher] linear wait start $(date -Is)" >> "${LOG}"
echo "[watcher] ckpt=${CKPT}" >> "${LOG}"

while [ ! -f "${CKPT}" ]; do
  sleep 300
done
sleep 120
echo "[watcher] checkpoint found $(date -Is)" >> "${LOG}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

run_one() {
  local seed="$1"
  local gpu="$2"
  local exp_name="${EXP_PREFIX}${seed}"
  local save_path="exp/concerto/${exp_name}"
  local log_path="${LOG_DIR}/${exp_name}.log"
  if [ -f "${save_path}/model/model_last.pth" ] && grep -q "Val result: mIoU/mAcc/allAcc" "${save_path}/train.log" 2>/dev/null; then
    echo "[skip] ${exp_name} already finished" >> "${LOG}"
    return 0
  fi
  echo "[run] gpu=${gpu} seed=${seed} exp=${exp_name}" >> "${LOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONPATH="$(pwd -P):${PYTHONPATH:-}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "${PYTHON_BIN}" tools/train.py \
    --config-file "${LIN_CONFIG}" \
    --num-gpus 1 \
    --num-machines 1 \
    --machine-rank 0 \
    --dist-url "tcp://127.0.0.1:$((17500 + seed * 101 + gpu))" \
    --options \
      save_path="${save_path}" \
      resume=False \
      weight="${CKPT}" \
      seed="${seed}" \
    > "${log_path}" 2>&1
}

status=0
declare -a SLOT_PID=()
declare -a SLOT_LABEL=()
next_idx=0
active=0

while [ "${next_idx}" -lt "${#SEEDS[@]}" ] || [ "${active}" -gt 0 ]; do
  for slot in "${!GPU_IDS[@]}"; do
    pid="${SLOT_PID[$slot]:-}"
    if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
      if ! wait "${pid}"; then
        status=1
        echo "[warn] ${SLOT_LABEL[$slot]} exited non-zero" >> "${LOG}"
      fi
      SLOT_PID[$slot]=""
      SLOT_LABEL[$slot]=""
    fi

    if [ -z "${SLOT_PID[$slot]:-}" ] && [ "${next_idx}" -lt "${#SEEDS[@]}" ]; then
      seed="${SEEDS[$next_idx]}"
      next_idx=$((next_idx + 1))
      gpu="${GPU_IDS[$slot]}"
      label="${EXP_PREFIX}${seed}"
      run_one "${seed}" "${gpu}" &
      SLOT_PID[$slot]="$!"
      SLOT_LABEL[$slot]="${label}"
      sleep 5
    fi
  done

  active=0
  for slot in "${!GPU_IDS[@]}"; do
    if [ -n "${SLOT_PID[$slot]:-}" ]; then
      active=$((active + 1))
    fi
  done
  if [ "${active}" -gt 0 ]; then
    sleep 30
  fi
done

echo "[done] linear jobs status=${status} $(date -Is)" >> "${LOG}"
exit "${status}"
