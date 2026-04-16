#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

EXP_TAG="${EXP_TAG:--h10016-qf1-v1c-prior256}"
EXP_PREFIX="${EXP_PREFIX:-arkit-full-projres-v1c}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1c}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
CONCERTO_MAX_TRAIN_ITER="${CONCERTO_MAX_TRAIN_ITER:-256}"
CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-8}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-12}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-1}"
CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"
WALLTIME="${WALLTIME:-00:35:00}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
MIN_RESIDUAL_NORM="${MIN_RESIDUAL_NORM:-0.80}"
QSUB_EXTRA="${QSUB_EXTRA:-}"
DRY_RUN="${DRY_RUN:-0}"

LINEAR_XYZ_PRIOR="${LINEAR_XYZ_PRIOR:-${POINTCEPT_DATA_ROOT}/runs/projres_v1/priors/linear/model_last.pth}"
LINEAR_Z_PRIOR="${LINEAR_Z_PRIOR:-${EXP_MIRROR_ROOT}/priors/linear_z/model_last.pth}"
MLP_Z_PRIOR="${MLP_Z_PRIOR:-${EXP_MIRROR_ROOT}/priors/mlp_z/model_last.pth}"

mkdir -p "${SUMMARY_ROOT}"
MANIFEST="${SUMMARY_ROOT}/job_manifest.tsv"
: > "${MANIFEST}"
printf 'arm\tprior\talpha\tbeta\tprior_path\tjob_id\n' >> "${MANIFEST}"

default_arm_specs() {
  cat <<EOF
linxyz-b075-a000 linear_xyz 0.00 0.75 ${LINEAR_XYZ_PRIOR}
linxyz-b075-a001 linear_xyz 0.01 0.75 ${LINEAR_XYZ_PRIOR}
linz-b075-a000 linear_z 0.00 0.75 ${LINEAR_Z_PRIOR}
linz-b075-a001 linear_z 0.01 0.75 ${LINEAR_Z_PRIOR}
mlpz-b075-a000 mlp_z 0.00 0.75 ${MLP_Z_PRIOR}
mlpz-b075-a001 mlp_z 0.01 0.75 ${MLP_Z_PRIOR}
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

echo "=== Launch ProjRes v1c prior-family smoke matrix ==="
echo "date=$(date -Is)"
echo "exp_tag=${EXP_TAG}"
echo "exp_prefix=${EXP_PREFIX}"
echo "summary_root=${SUMMARY_ROOT}"
echo "concerto_max_train_iter=${CONCERTO_MAX_TRAIN_ITER}"
echo "walltime=${WALLTIME}"
echo "dry_run=${DRY_RUN}"

"${arm_specs_cmd[@]}" | while read -r arm prior alpha beta prior_path; do
  [ -n "${arm}" ] || continue
  case "${arm}" in
    \#*) continue ;;
  esac
  if [ ! -f "${prior_path}" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "[error] missing prior for ${arm}: ${prior_path}" >&2
    exit 2
  fi
  vars="ARM_NAME=${arm},PRIOR_NAME=${prior},COORD_PRIOR_PATH=${prior_path},COORD_PROJECTION_ALPHA=${alpha},COORD_PROJECTION_BETA=${beta},EXP_PREFIX=${EXP_PREFIX},EXP_TAG=${EXP_TAG},EXP_MIRROR_ROOT=${EXP_MIRROR_ROOT},SUMMARY_ROOT=${SUMMARY_ROOT},GPU_IDS_CSV=$(pbs_v_escape "${GPU_IDS_CSV}"),CONCERTO_MAX_TRAIN_ITER=${CONCERTO_MAX_TRAIN_ITER},CONCERTO_GLOBAL_BATCH_SIZE=${CONCERTO_GLOBAL_BATCH_SIZE},CONCERTO_GRAD_ACCUM=${CONCERTO_GRAD_ACCUM},CONCERTO_NUM_WORKER=${CONCERTO_NUM_WORKER},CONCERTO_ENABLE_FLASH=${CONCERTO_ENABLE_FLASH},RUN_PREFLIGHT=${RUN_PREFLIGHT},MIN_RESIDUAL_NORM=${MIN_RESIDUAL_NORM}"
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${arm}" "${prior}" "${alpha}" "${beta}" "${prior_path}" "${job_id}" | tee -a "${MANIFEST}"
done

echo "[manifest] ${MANIFEST}"
