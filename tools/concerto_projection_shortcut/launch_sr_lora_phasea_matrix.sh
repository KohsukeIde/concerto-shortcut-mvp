#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

COORD_RIVAL_PATH="${COORD_RIVAL_PATH:?COORD_RIVAL_PATH is required}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-tools/concerto_projection_shortcut/submit_sr_lora_phasea_abciq_qf.sh}"
RANKS=( ${RANKS:-4 8} )
DISTILLS=( ${DISTILLS:-0.3 1.0} )
MARGIN_ALPHA="${SR_MARGIN_ALPHA:-1.0}"
MARGIN_VALUE="${SR_MARGIN_VALUE:-0.1}"
QSUB_RESOURCE="${QSUB_RESOURCE:-rt_QF=4}"
WALLTIME="${WALLTIME:-00:40:00}"
EXP_PREFIX="${EXP_PREFIX:-sr-lora-v5}"
EXP_SUFFIX="${EXP_SUFFIX:-}"

VARS_COMMON=(
  "WORKDIR=${REPO_ROOT}"
  "COORD_RIVAL_PATH=${COORD_RIVAL_PATH}"
  "SR_MARGIN_ALPHA=${MARGIN_ALPHA}"
  "SR_MARGIN_VALUE=${MARGIN_VALUE}"
)

for name in \
  OFFICIAL_WEIGHT WEIGHT_PATH CONFIG_NAME DATASET_NAME \
  CONCERTO_MAX_TRAIN_ITER CONCERTO_EPOCH CONCERTO_EVAL_EPOCH CONCERTO_STOP_EPOCH \
  CONCERTO_GLOBAL_BATCH_SIZE CONCERTO_NUM_WORKER NPROC_PER_NODE TRAIN_GPU_IDS_CSV \
  PYTHON_MODULE CUDA_MODULE; do
  if [ -n "${!name:-}" ]; then
    VARS_COMMON+=( "${name}=${!name}" )
  fi
done

for rank in "${RANKS[@]}"; do
  for distill in "${DISTILLS[@]}"; do
    distill_tag="${distill//./p}"
    exp="${EXP_PREFIX}-r${rank}-d${distill_tag}${EXP_SUFFIX}"
    VARS=(
      "${VARS_COMMON[@]}"
      "SR_LORA_RANK=${rank}"
      "SR_LORA_ALPHA=$((rank * 2))"
      "SR_DISTILL_WEIGHT=${distill}"
      "EXP_NAME=${exp}"
    )
    echo "[submit] ${exp} resource=${QSUB_RESOURCE} walltime=${WALLTIME}"
    qsub \
      -N "sr_lora_r${rank}_d${distill_tag}" \
      -l "${QSUB_RESOURCE}" \
      -l "walltime=${WALLTIME}" \
      -v "$(IFS=,; echo "${VARS[*]}")" \
      "${SUBMIT_SCRIPT}"
  done
done
