#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

EXP_TAG="${EXP_TAG:--h10016-qf1-v1b-smoke}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1b}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
CONCERTO_MAX_TRAIN_ITER="${CONCERTO_MAX_TRAIN_ITER:-0}"
CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-8}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-12}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-1}"
CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"
WALLTIME="${WALLTIME:-01:15:00}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
MIN_RESIDUAL_NORM="${MIN_RESIDUAL_NORM:-0.80}"
QSUB_EXTRA="${QSUB_EXTRA:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SUMMARY_ROOT}"
MANIFEST="${SUMMARY_ROOT}/job_manifest.tsv"
: > "${MANIFEST}"
printf 'arm\talpha\tbeta\tjob_id\n' >> "${MANIFEST}"

default_arm_specs() {
  cat <<'EOF'
resonly-b025-a000 0.00 0.25
resonly-b050-a000 0.00 0.50
resonly-b075-a000 0.00 0.75
penalty-b000-a001 0.01 0.00
penalty-b000-a002 0.02 0.00
combo-b025-a001 0.01 0.25
combo-b050-a001 0.01 0.50
combo-b075-a001 0.01 0.75
combo-b025-a002 0.02 0.25
combo-b050-a002 0.02 0.50
combo-b075-a002 0.02 0.75
EOF
}

pbs_v_escape() {
  printf '%s' "$1" | sed 's/,/\\,/g'
}

if [ -n "${ARM_SPECS_FILE:-}" ]; then
  arm_specs_cmd=(cat "${ARM_SPECS_FILE}")
else
  arm_specs_cmd=(default_arm_specs)
fi

echo "=== Launch ProjRes v1b smoke matrix ==="
echo "date=$(date -Is)"
echo "exp_tag=${EXP_TAG}"
echo "summary_root=${SUMMARY_ROOT}"
echo "concerto_max_train_iter=${CONCERTO_MAX_TRAIN_ITER}"
echo "walltime=${WALLTIME}"
echo "dry_run=${DRY_RUN}"

"${arm_specs_cmd[@]}" | while read -r arm alpha beta; do
  [ -n "${arm}" ] || continue
  case "${arm}" in
    \#*) continue ;;
  esac
  vars="ARM_NAME=${arm},COORD_PROJECTION_ALPHA=${alpha},COORD_PROJECTION_BETA=${beta},EXP_TAG=${EXP_TAG},EXP_MIRROR_ROOT=${EXP_MIRROR_ROOT},SUMMARY_ROOT=${SUMMARY_ROOT},GPU_IDS_CSV=$(pbs_v_escape "${GPU_IDS_CSV}"),CONCERTO_MAX_TRAIN_ITER=${CONCERTO_MAX_TRAIN_ITER},CONCERTO_GLOBAL_BATCH_SIZE=${CONCERTO_GLOBAL_BATCH_SIZE},CONCERTO_GRAD_ACCUM=${CONCERTO_GRAD_ACCUM},CONCERTO_NUM_WORKER=${CONCERTO_NUM_WORKER},CONCERTO_ENABLE_FLASH=${CONCERTO_ENABLE_FLASH},RUN_PREFLIGHT=${RUN_PREFLIGHT},MIN_RESIDUAL_NORM=${MIN_RESIDUAL_NORM}"
  cmd=(qsub -l "walltime=${WALLTIME}" -v "${vars}")
  if [ -n "${QSUB_EXTRA}" ]; then
    # shellcheck disable=SC2206
    extra_parts=(${QSUB_EXTRA})
    cmd+=("${extra_parts[@]}")
  fi
  cmd+=(tools/concerto_projection_shortcut/submit_projres_v1b_smoke_arm_abciq_qf.sh)
  echo "+ ${cmd[*]}"
  if [ "${DRY_RUN}" = "1" ]; then
    job_id="DRY_RUN"
  else
    job_id="$("${cmd[@]}")"
  fi
  printf '%s\t%s\t%s\t%s\n' "${arm}" "${alpha}" "${beta}" "${job_id}" | tee -a "${MANIFEST}"
done

echo "[manifest] ${MANIFEST}"
