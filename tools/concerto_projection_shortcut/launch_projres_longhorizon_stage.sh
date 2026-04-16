#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

STAGE_EPOCH="${STAGE_EPOCH:-25}"
TARGET_EPOCH="${TARGET_EPOCH:-${STAGE_EPOCH}}"
STOP_EPOCH="${STOP_EPOCH:-${STAGE_EPOCH}}"
RT_QF="${RT_QF:-8}"
CONTINUE_WALLTIME="${CONTINUE_WALLTIME:-03:50:00}"
FOLLOWUP_WALLTIME="${FOLLOWUP_WALLTIME:-00:45:00}"
if [ -z "${EXP_TAG+x}" ]; then
  if [ "${TARGET_EPOCH}" = "${STOP_EPOCH}" ]; then
    EXP_TAG="-long-e$(printf '%03d' "${STOP_EPOCH}")-qf32"
  else
    EXP_TAG="-long-t$(printf '%03d' "${TARGET_EPOCH}")-e$(printf '%03d' "${STOP_EPOCH}")-qf32"
  fi
fi
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_long}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/${EXP_TAG#-}}"
OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${WEIGHT_DIR}/concerto_base_origin.pth}"
COORD_PRIOR_PATH_DEFAULT="${COORD_PRIOR_PATH_DEFAULT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1/priors/mlp/model_last.pth}"
CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-32}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-3}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-64}"
CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"
LINEAR_GPU_IDS_CSV="${LINEAR_GPU_IDS_CSV:-0,1,2,3}"
LINEAR_NUM_GPU="${LINEAR_NUM_GPU:-$(awk -F',' '{print NF}' <<< "${LINEAR_GPU_IDS_CSV}")}"
LINEAR_TRAIN_LAUNCHER="${LINEAR_TRAIN_LAUNCHER:-torchrun}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SUMMARY_ROOT}"
MANIFEST="${SUMMARY_ROOT}/longhorizon_job_manifest.tsv"
: > "${MANIFEST}"
printf 'arm\tconfig\texp\tweight\talpha\tbeta\tprior\ttarget_epoch\tstop_epoch\tcontinue_job\tfollowup_job\n' >> "${MANIFEST}"

pbs_v_escape() {
  printf '%s' "$1" | sed 's/,/\\,/g'
}

submit_continue() {
  local arm="$1"
  local config="$2"
  local exp="$3"
  local weight="$4"
  local alpha="$5"
  local beta="$6"
  local prior="$7"
  local vars
  vars="EXP_MIRROR_ROOT=${EXP_MIRROR_ROOT},CONFIG_NAME=${config},EXP_NAME=${exp},WEIGHT_PATH=${weight},CONCERTO_EPOCH=${TARGET_EPOCH},CONCERTO_STOP_EPOCH=${STOP_EPOCH},CONCERTO_GLOBAL_BATCH_SIZE=${CONCERTO_GLOBAL_BATCH_SIZE},CONCERTO_GRAD_ACCUM=${CONCERTO_GRAD_ACCUM},CONCERTO_NUM_WORKER=${CONCERTO_NUM_WORKER},CONCERTO_ENABLE_FLASH=${CONCERTO_ENABLE_FLASH},COORD_PROJECTION_ALPHA=${alpha},COORD_PROJECTION_BETA=${beta},COORD_PRIOR_PATH=${prior},GPU_IDS_CSV=$(pbs_v_escape "0,1,2,3"),NPROC_PER_NODE=4"
  local cmd=(qsub -l "rt_QF=${RT_QF}" -l "walltime=${CONTINUE_WALLTIME}" -v "${vars}" tools/concerto_projection_shortcut/submit_concerto_continue_abciq_qf.sh)
  echo "+ ${cmd[*]}" >&2
  if [ "${DRY_RUN}" = "1" ]; then
    printf 'DRY_RUN'
  else
    "${cmd[@]}"
  fi
}

submit_followup() {
  local continue_job="$1"
  local arm="$2"
  local continue_exp="$3"
  local alpha="$4"
  local beta="$5"
  local prior="$6"
  local linear_exp="scannet-proxy-${continue_exp}-lin"
  local continue_ckpt="${REPO_ROOT}/exp/concerto/${continue_exp}/model/model_last.pth"
  local gate_json="${SUMMARY_ROOT}/${linear_exp}_gate.json"
  local vars
  vars="EXP_MIRROR_ROOT=${EXP_MIRROR_ROOT},SUMMARY_ROOT=${SUMMARY_ROOT},EXP_TAG=${EXP_TAG},CONTINUE_EXP=${continue_exp},CONTINUE_CKPT=${continue_ckpt},LINEAR_EXP=${linear_exp},LINEAR_GATE_JSON=${gate_json},RUN_STRESS=0,RUN_LINEAR=1,FOLLOWUP_PARALLEL=0,LINEAR_GPU=0,LINEAR_GPU_IDS_CSV=$(pbs_v_escape "${LINEAR_GPU_IDS_CSV}"),LINEAR_NUM_GPU=${LINEAR_NUM_GPU},LINEAR_TRAIN_LAUNCHER=${LINEAR_TRAIN_LAUNCHER},SELECTED_ALPHA=${alpha},SELECTED_BETA=${beta},COORD_PRIOR_PATH=${prior},LINEAR_CONFIG=semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly"
  local cmd=(qsub -W "depend=afterok:${continue_job}" -l rt_QF=1 -l "walltime=${FOLLOWUP_WALLTIME}" -v "${vars}" tools/concerto_projection_shortcut/submit_projres_v1_followup_abciq_qf.sh)
  echo "+ ${cmd[*]}" >&2
  if [ "${DRY_RUN}" = "1" ]; then
    printf 'DRY_RUN'
  else
    "${cmd[@]}"
  fi
}

echo "=== Launch ProjRes long-horizon staged continuation ==="
echo "date=$(date -Is)"
echo "stage_epoch=${STAGE_EPOCH}"
echo "target_epoch=${TARGET_EPOCH}"
echo "stop_epoch=${STOP_EPOCH}"
echo "rt_qf=${RT_QF}"
echo "continue_walltime=${CONTINUE_WALLTIME}"
echo "followup_walltime=${FOLLOWUP_WALLTIME}"
echo "exp_tag=${EXP_TAG}"
echo "summary_root=${SUMMARY_ROOT}"
echo "linear_gpu_ids_csv=${LINEAR_GPU_IDS_CSV}"
echo "linear_num_gpu=${LINEAR_NUM_GPU}"
echo "linear_train_launcher=${LINEAR_TRAIN_LAUNCHER}"
echo "dry_run=${DRY_RUN}"

arms=(
  "original|pretrain-concerto-v1m1-0-arkit-full-continue-h10016|arkit-full-original${EXP_TAG}-continue|${OFFICIAL_WEIGHT}|0.0|1.0|${COORD_PRIOR_PATH_DEFAULT}"
  "v1b-combo-b075-a001|pretrain-concerto-v1m1-0-arkit-full-projres-v1b-continue-h10016|arkit-full-projres-v1b-combo-b075-a001${EXP_TAG}-continue|${OFFICIAL_WEIGHT}|0.01|0.75|${COORD_PRIOR_PATH_DEFAULT}"
)

for spec in "${arms[@]}"; do
  IFS='|' read -r arm config exp weight alpha beta prior <<< "${spec}"
  cont_job="$(submit_continue "${arm}" "${config}" "${exp}" "${weight}" "${alpha}" "${beta}" "${prior}")"
  follow_job="$(submit_followup "${cont_job}" "${arm}" "${exp}" "${alpha}" "${beta}" "${prior}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${arm}" "${config}" "${exp}" "${weight}" "${alpha}" "${beta}" "${prior}" \
    "${TARGET_EPOCH}" "${STOP_EPOCH}" \
    "${cont_job}" "${follow_job}" | tee -a "${MANIFEST}"
done

echo "[manifest] ${MANIFEST}"
