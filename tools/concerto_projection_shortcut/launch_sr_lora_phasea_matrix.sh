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
QSUB_RESOURCE="${QSUB_RESOURCE:-rt_QF=1}"
WALLTIME="${WALLTIME:-01:00:00}"

for rank in "${RANKS[@]}"; do
  for distill in "${DISTILLS[@]}"; do
    distill_tag="${distill//./p}"
    exp="sr-lora-v5-r${rank}-d${distill_tag}"
    echo "[submit] ${exp} resource=${QSUB_RESOURCE} walltime=${WALLTIME}"
    qsub \
      -l "${QSUB_RESOURCE}" \
      -l "walltime=${WALLTIME}" \
      -v "WORKDIR=${REPO_ROOT},COORD_RIVAL_PATH=${COORD_RIVAL_PATH},SR_LORA_RANK=${rank},SR_LORA_ALPHA=$((rank * 2)),SR_DISTILL_WEIGHT=${distill},SR_MARGIN_ALPHA=${MARGIN_ALPHA},SR_MARGIN_VALUE=${MARGIN_VALUE},EXP_NAME=${exp}" \
      "${SUBMIT_SCRIPT}"
  done
done
