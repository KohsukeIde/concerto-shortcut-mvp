#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1c}"
EXP_TAG="${EXP_TAG:--h10016x4-qf16-v1c}"
EXP_PREFIX="${EXP_PREFIX:-arkit-full-projres-v1c}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}}"
MANIFEST="${MANIFEST:-${SUMMARY_ROOT}/continue_job_manifest.tsv}"
WALLTIME="${WALLTIME:-01:05:00}"
RT_QF="${RT_QF:-1}"
MAX_FOLLOWUP_ARMS="${MAX_FOLLOWUP_ARMS:-2}"
FOLLOWUP_PARALLEL="${FOLLOWUP_PARALLEL:-1}"
RUN_STRESS="${RUN_STRESS:-1}"
RUN_LINEAR="${RUN_LINEAR:-1}"
MAX_STRESS_BATCHES="${MAX_STRESS_BATCHES:-20}"
STRESS_GPU="${STRESS_GPU:-0}"
LINEAR_GPU="${LINEAR_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "${MANIFEST}" ]; then
  echo "[error] missing continuation manifest: ${MANIFEST}" >&2
  exit 2
fi

FOLLOWUP_MANIFEST="${SUMMARY_ROOT}/followup_job_manifest.tsv"
: > "${FOLLOWUP_MANIFEST}"
printf 'arm\tprior\talpha\tbeta\tprior_path\tjob_id\n' >> "${FOLLOWUP_MANIFEST}"

echo "=== Launch ProjRes v1c follow-up jobs ==="
echo "date=$(date -Is)"
echo "manifest=${MANIFEST}"
echo "summary_root=${SUMMARY_ROOT}"
echo "walltime=${WALLTIME}"
echo "dry_run=${DRY_RUN}"

awk -F'\t' 'NR > 1 {print}' "${MANIFEST}" | head -n "${MAX_FOLLOWUP_ARMS}" | while IFS=$'\t' read -r arm prior alpha beta prior_path continue_job; do
  continue_exp="${EXP_PREFIX}-${arm}${EXP_TAG}-continue"
  linear_exp="scannet-proxy-projres-v1c-${arm}${EXP_TAG}-lin"
  vars="EXP_MIRROR_ROOT=${EXP_MIRROR_ROOT},EXP_TAG=${EXP_TAG},SUMMARY_ROOT=${SUMMARY_ROOT},CONTINUE_CONFIG=pretrain-concerto-v1m1-0-arkit-full-projres-v1b-continue-h10016,CONTINUE_EXP=${continue_exp},LINEAR_EXP=${linear_exp},SELECTED_ALPHA=${alpha},SELECTED_BETA=${beta},COORD_PRIOR_PATH=${prior_path},FOLLOWUP_PARALLEL=${FOLLOWUP_PARALLEL},RUN_STRESS=${RUN_STRESS},RUN_LINEAR=${RUN_LINEAR},MAX_STRESS_BATCHES=${MAX_STRESS_BATCHES},STRESS_GPU=${STRESS_GPU},LINEAR_GPU=${LINEAR_GPU}"
  cmd=(qsub -l "rt_QF=${RT_QF}" -l "walltime=${WALLTIME}" -v "${vars}")
  if [[ "${continue_job}" != "DRY_RUN" && "${continue_job}" != "" ]]; then
    cmd+=(-W "depend=afterok:${continue_job}")
  fi
  cmd+=(tools/concerto_projection_shortcut/submit_projres_v1_followup_abciq_qf.sh)
  echo "+ ${cmd[*]}"
  if [ "${DRY_RUN}" = "1" ]; then
    job_id="DRY_RUN"
  else
    job_id="$("${cmd[@]}")"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${arm}" "${prior}" "${alpha}" "${beta}" "${prior_path}" "${job_id}" | tee -a "${FOLLOWUP_MANIFEST}"
done

echo "[manifest] ${FOLLOWUP_MANIFEST}"
