#!/usr/bin/env bash
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/data/logs/abciq/

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${WORKDIR}"

module load python/3.11/3.11.14
module load cuda/12.6/12.6.2

VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/data/venv/pointcept-concerto-py311-cu124/bin/activate}"
source "${VENV_ACTIVATE}"

export HF_HOME="${HF_HOME:-${WORKDIR}/data/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${WORKDIR}/data/hf-home/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-${WORKDIR}/data/hf-home/xet}"
export TORCH_HOME="${TORCH_HOME:-${WORKDIR}/data/torch-home}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS_CSV:-0}"

CONFIG="${CONFIG:-configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py}"
WEIGHT="${WEIGHT:-data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth}"
BASE_WEIGHT="${BASE_WEIGHT:-${WEIGHT}}"
DATA_ROOT="${DATA_ROOT:-data/scannet}"
TAG="${TAG:-cida-inloop-base}"
OUTPUT_DIR="${OUTPUT_DIR:-data/runs/cida_inloop_decoder_adaptation/${TAG}}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
CLASS_PAIRS="${CLASS_PAIRS:-picture:wall,desk:table,sink:cabinet,counter:cabinet}"
WEAK_CLASSES="${WEAK_CLASSES:-picture,counter,desk,sink,cabinet}"
EPOCHS="${EPOCHS:-50}"
MAX_TRAIN_ITERS="${MAX_TRAIN_ITERS:--1}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKER="${NUM_WORKER:-16}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
WEAK_CLASS_WEIGHT="${WEAK_CLASS_WEIGHT:-2.0}"
LAMBDA_PAIR="${LAMBDA_PAIR:-0.2}"
LAMBDA_KL="${LAMBDA_KL:-0.05}"
LAMBDA_DIST="${LAMBDA_DIST:-0.05}"
TEMPERATURE="${TEMPERATURE:-2.0}"
CLIP_GRAD="${CLIP_GRAD:-3.0}"
EVAL_EVERY_EPOCH="${EVAL_EVERY_EPOCH:-0}"
SEED="${SEED:-20260419}"
AMP_FLAG="${AMP_FLAG:---amp}"
SAVE_FLAG="${SAVE_FLAG:---save-checkpoint}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

echo "=== CIDA in-loop decoder adaptation ==="
echo "date=$(date --iso-8601=seconds)"
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "workdir=${WORKDIR}"
echo "config=${CONFIG}"
echo "weight=${WEIGHT}"
echo "base_weight=${BASE_WEIGHT}"
echo "data_root=${DATA_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "tag=${TAG}"
echo "epochs=${EPOCHS}"
echo "max_train_iters=${MAX_TRAIN_ITERS}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "batch_size=${BATCH_SIZE}"
echo "lambda_pair=${LAMBDA_PAIR}"
echo "lambda_kl=${LAMBDA_KL}"
echo "lambda_dist=${LAMBDA_DIST}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

CMD=(python tools/concerto_projection_shortcut/train_cida_inloop_decoder_adaptation.py
  --config "${CONFIG}"
  --weight "${WEIGHT}"
  --base-weight "${BASE_WEIGHT}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --tag "${TAG}"
  --train-split "${TRAIN_SPLIT}"
  --val-split "${VAL_SPLIT}"
  --class-pairs "${CLASS_PAIRS}"
  --weak-classes "${WEAK_CLASSES}"
  --epochs "${EPOCHS}"
  --max-train-iters "${MAX_TRAIN_ITERS}"
  --max-val-batches "${MAX_VAL_BATCHES}"
  --batch-size "${BATCH_SIZE}"
  --num-worker "${NUM_WORKER}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --weak-class-weight "${WEAK_CLASS_WEIGHT}"
  --lambda-pair "${LAMBDA_PAIR}"
  --lambda-kl "${LAMBDA_KL}"
  --lambda-dist "${LAMBDA_DIST}"
  --temperature "${TEMPERATURE}"
  --clip-grad "${CLIP_GRAD}"
  --eval-every-epoch "${EVAL_EVERY_EPOCH}"
  --seed "${SEED}")

if [[ -n "${AMP_FLAG}" ]]; then
  CMD+=("${AMP_FLAG}")
fi
if [[ -n "${SAVE_FLAG}" ]]; then
  CMD+=("${SAVE_FLAG}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi
if [[ "${EVAL_ONLY}" == "1" ]]; then
  CMD+=(--eval-only)
fi

"${CMD[@]}"

echo "[done] CIDA in-loop decoder adaptation"
