#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

WAIT_PID="${WAIT_PID:-}"
GPU_IDS_CSV="${GPU_IDS_CSV:-1,2,3}"
EXP_NAME="${EXP_NAME:-arkit-full-continue-xyz-only-input-seed0-gpu123}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/concerto/pretrain-concerto-v1m1-0-arkit-full-xyz-only-continue-env.py}"
WEIGHT="${WEIGHT:-weights/concerto/concerto_base_origin.pth}"
PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/envs/pointcept-concerto-cu121/bin/python}"
LOG_DIR="${LOG_DIR:-logs/xyz_only_input_continue}"
LOG="${LOG_DIR}/driver.log"

mkdir -p "${LOG_DIR}"

echo "[watcher] start $(date -Is)" >> "${LOG}"
echo "[watcher] wait_pid=${WAIT_PID:-none}" >> "${LOG}"
if [ -n "${WAIT_PID}" ]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "${LOG}" 2>/dev/null || true
    sleep 300
  done
fi
echo "[watcher] dependency finished $(date -Is)" >> "${LOG}"

if [ -f "exp/concerto/${EXP_NAME}/model/model_last.pth" ]; then
  echo "[watcher] checkpoint already exists; skip exp/concerto/${EXP_NAME}/model/model_last.pth" >> "${LOG}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS_CSV}"
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-3}"
export CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-3}"
export CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-6}"
export CONCERTO_EPOCH="${CONCERTO_EPOCH:-5}"
export CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-0}"

NPROC_PER_NODE="$(awk -F',' '{print NF}' <<< "${GPU_IDS_CSV}")"
DIST_PORT="${DIST_PORT:-17431}"

echo "[run] xyz-only continuation $(date -Is)" >> "${LOG}"
echo "[run] gpus=${GPU_IDS_CSV} nproc=${NPROC_PER_NODE} exp=${EXP_NAME} seed=${SEED}" >> "${LOG}"
echo "[run] batch=${CONCERTO_GLOBAL_BATCH_SIZE} grad_accum=${CONCERTO_GRAD_ACCUM} workers=${CONCERTO_NUM_WORKER} epoch=${CONCERTO_EPOCH}" >> "${LOG}"

"${PYTHON_BIN}" tools/train.py \
  --config-file "${CONFIG}" \
  --num-gpus "${NPROC_PER_NODE}" \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url "tcp://127.0.0.1:${DIST_PORT}" \
  --options \
    save_path="exp/concerto/${EXP_NAME}" \
    resume=False \
    weight="${WEIGHT}" \
    seed="${SEED}" \
  >> "${LOG}" 2>&1

echo "[done] xyz-only continuation $(date -Is)" >> "${LOG}"
