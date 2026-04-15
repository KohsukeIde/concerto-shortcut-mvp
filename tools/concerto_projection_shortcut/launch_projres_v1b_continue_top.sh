#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

SMOKE_SUMMARY_ROOT="${SMOKE_SUMMARY_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1b/summaries/h10016-qf1-v1b-smoke}"
SELECTED_JSON="${SELECTED_JSON:-${SMOKE_SUMMARY_ROOT}/selected_smoke.json}"
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1b}"
EXP_TAG="${EXP_TAG:--h10032-qf32}"
MAX_CONTINUE_ARMS="${MAX_CONTINUE_ARMS:-2}"
WALLTIME="${WALLTIME:-01:00:00}"
RT_QF="${RT_QF:-8}"
CONCERTO_EPOCH="${CONCERTO_EPOCH:-5}"
CONCERTO_MAX_TRAIN_ITER="${CONCERTO_MAX_TRAIN_ITER:-0}"
CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-32}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-3}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-64}"
CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "${SELECTED_JSON}" ]; then
  "${PYTHON_BIN:-python}" tools/concerto_projection_shortcut/select_projres_v1b_smoke.py \
    "${SMOKE_SUMMARY_ROOT}" --top-k "${MAX_CONTINUE_ARMS}" --out "${SELECTED_JSON}"
fi

mkdir -p "${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}"
MANIFEST="${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}/continue_job_manifest.tsv"
: > "${MANIFEST}"
printf 'arm\talpha\tbeta\tjob_id\n' >> "${MANIFEST}"

"${PYTHON_BIN:-python}" - "${SELECTED_JSON}" "${MAX_CONTINUE_ARMS}" <<'PY' | while IFS=$'\t' read -r arm alpha beta summary_json; do
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
limit = int(sys.argv[2])
for row in payload.get("top", [])[:limit]:
    print(
        row["arm"],
        row["alpha"],
        row["beta"],
        row["summary_json"],
        sep="\t",
    )
PY
  continue_exp="arkit-full-projres-v1b-${arm}${EXP_TAG}-continue"
  vars="EXP_MIRROR_ROOT=${EXP_MIRROR_ROOT},CONTINUE_CONFIG=pretrain-concerto-v1m1-0-arkit-full-projres-v1b-continue-h10016,CONTINUE_EXP=${continue_exp},SELECTED_SMOKE_JSON=${summary_json},SELECTED_ALPHA=${alpha},SELECTED_BETA=${beta},SELECTED_PRIOR_JSON=${POINTCEPT_DATA_ROOT}/runs/projres_v1/priors/selected_prior.json,CONCERTO_EPOCH=${CONCERTO_EPOCH},CONCERTO_MAX_TRAIN_ITER=${CONCERTO_MAX_TRAIN_ITER},EXP_TAG=${EXP_TAG},CONCERTO_GLOBAL_BATCH_SIZE=${CONCERTO_GLOBAL_BATCH_SIZE},CONCERTO_GRAD_ACCUM=${CONCERTO_GRAD_ACCUM},CONCERTO_NUM_WORKER=${CONCERTO_NUM_WORKER},CONCERTO_ENABLE_FLASH=${CONCERTO_ENABLE_FLASH},PREFLIGHT_CONCERTO_GLOBAL_BATCH_SIZE=1,PREFLIGHT_CONCERTO_NUM_WORKER=0,NPROC_PER_NODE=${NPROC_PER_NODE}"
  cmd=(qsub -l "rt_QF=${RT_QF}" -l "walltime=${WALLTIME}" -v "${vars}" tools/concerto_projection_shortcut/submit_projres_v1_continue_abciq_qf16.sh)
  echo "+ ${cmd[*]}"
  if [ "${DRY_RUN}" = "1" ]; then
    job_id="DRY_RUN"
  else
    job_id="$("${cmd[@]}")"
  fi
  printf '%s\t%s\t%s\t%s\n' "${arm}" "${alpha}" "${beta}" "${job_id}" | tee -a "${MANIFEST}"
done

echo "[manifest] ${MANIFEST}"
