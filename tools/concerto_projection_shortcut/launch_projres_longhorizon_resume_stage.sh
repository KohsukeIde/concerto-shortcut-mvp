#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

FROM_STAGE_EPOCH="${FROM_STAGE_EPOCH:-25}"
TO_STAGE_EPOCH="${TO_STAGE_EPOCH:-50}"
TARGET_EPOCH="${TARGET_EPOCH:-${TO_STAGE_EPOCH}}"
RT_QF="${RT_QF:-8}"
CONTINUE_WALLTIME="${CONTINUE_WALLTIME:-03:40:00}"
FOLLOWUP_WALLTIME="${FOLLOWUP_WALLTIME:-00:45:00}"
if [ -z "${FROM_EXP_TAG+x}" ]; then
  if [ "${TARGET_EPOCH}" = "${FROM_STAGE_EPOCH}" ]; then
    FROM_EXP_TAG="-long-e$(printf '%03d' "${FROM_STAGE_EPOCH}")-qf32"
  else
    FROM_EXP_TAG="-long-t$(printf '%03d' "${TARGET_EPOCH}")-e$(printf '%03d' "${FROM_STAGE_EPOCH}")-qf32"
  fi
fi
if [ -z "${TO_EXP_TAG+x}" ]; then
  if [ "${TARGET_EPOCH}" = "${TO_STAGE_EPOCH}" ]; then
    TO_EXP_TAG="-long-e$(printf '%03d' "${TO_STAGE_EPOCH}")-qf32"
  else
    TO_EXP_TAG="-long-t$(printf '%03d' "${TARGET_EPOCH}")-e$(printf '%03d' "${TO_STAGE_EPOCH}")-qf32"
  fi
fi
EXP_MIRROR_ROOT="${EXP_MIRROR_ROOT:-${POINTCEPT_DATA_ROOT}/runs/projres_long}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXP_MIRROR_ROOT}/summaries/${TO_EXP_TAG#-}}"
COORD_PRIOR_PATH_DEFAULT="${COORD_PRIOR_PATH_DEFAULT:-${POINTCEPT_DATA_ROOT}/runs/projres_v1/priors/mlp/model_last.pth}"
CONCERTO_GLOBAL_BATCH_SIZE="${CONCERTO_GLOBAL_BATCH_SIZE:-32}"
CONCERTO_GRAD_ACCUM="${CONCERTO_GRAD_ACCUM:-3}"
CONCERTO_NUM_WORKER="${CONCERTO_NUM_WORKER:-64}"
CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"
LINEAR_GPU_IDS_CSV="${LINEAR_GPU_IDS_CSV:-0,1,2,3}"
LINEAR_NUM_GPU="${LINEAR_NUM_GPU:-$(awk -F',' '{print NF}' <<< "${LINEAR_GPU_IDS_CSV}")}"
LINEAR_TRAIN_LAUNCHER="${LINEAR_TRAIN_LAUNCHER:-torchrun}"
ALLOW_SCHEDULE_MISMATCH="${ALLOW_SCHEDULE_MISMATCH:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SUMMARY_ROOT}"
MANIFEST="${SUMMARY_ROOT}/longhorizon_resume_job_manifest.tsv"
: > "${MANIFEST}"
printf 'arm\tconfig\tfrom_exp\tto_exp\talpha\tbeta\tprior\ttarget_epoch\tfrom_stage_epoch\tto_stage_epoch\tcontinue_job\tfollowup_job\n' >> "${MANIFEST}"

pbs_v_escape() {
  printf '%s' "$1" | sed 's/,/\\,/g'
}

patch_py_assignment() {
  local config_py="$1"
  local key="$2"
  local value="$3"
  if grep -q "^${key} = " "${config_py}"; then
    sed -E -i "s/^${key} = .*/${key} = ${value}/" "${config_py}"
  else
    printf '\n%s = %s\n' "${key}" "${value}" >> "${config_py}"
  fi
}

patch_config_schedule() {
  local config_py="$1"
  local target_epoch="$2"
  local stop_epoch="$3"
  patch_py_assignment "${config_py}" epoch "${target_epoch}"
  patch_py_assignment "${config_py}" eval_epoch "${target_epoch}"
  patch_py_assignment "${config_py}" stop_epoch "${stop_epoch}"
}

validate_source_schedule() {
  local config_py="$1"
  local target_epoch="$2"
  local source_eval_epoch
  source_eval_epoch="$(awk '$1 == "eval_epoch" && $2 == "=" {print $3}' "${config_py}" | tail -n 1)"
  if [ -n "${source_eval_epoch}" ] && [ "${source_eval_epoch}" != "${target_epoch}" ]; then
    echo "[error] source checkpoint was built with eval_epoch=${source_eval_epoch}, not target_epoch=${target_epoch}" >&2
    echo "[error] exact staged resume requires the source stage to use the same target scheduler." >&2
    echo "[hint] relaunch the first stage with TARGET_EPOCH=${target_epoch} STOP_EPOCH=${FROM_STAGE_EPOCH}, or set ALLOW_SCHEDULE_MISMATCH=1 for a non-exact resume." >&2
    if [ "${ALLOW_SCHEDULE_MISMATCH}" != "1" ]; then
      exit 3
    fi
    echo "[warn] ALLOW_SCHEDULE_MISMATCH=1; continuing with non-exact resume." >&2
  fi
}

prepare_resume_exp() {
  local from_exp="$1"
  local to_exp="$2"
  local target_epoch="$3"
  local from_link="exp/concerto/${from_exp}"
  local to_link="exp/concerto/${to_exp}"
  local from_target
  local to_target="${EXP_MIRROR_ROOT}/exp/${to_exp}"

  if [ ! -e "${from_link}/model/model_last.pth" ]; then
    echo "[error] missing source checkpoint: ${from_link}/model/model_last.pth" >&2
    exit 2
  fi
  from_target="$(readlink -f "${from_link}")"
  validate_source_schedule "${from_target}/config.py" "${target_epoch}"

  echo "[prepare] ${from_exp} -> ${to_exp}"
  echo "[prepare] source=${from_target}"
  echo "[prepare] target=${to_target}"

  if [ "${DRY_RUN}" = "1" ]; then
    return
  fi

  mkdir -p "exp/concerto" "${to_target}/model"
  if [ ! -e "${to_link}" ]; then
    ln -s "${to_target}" "${to_link}"
  fi
  if [ ! -e "${to_target}/code" ]; then
    ln -s "${from_target}/code" "${to_target}/code"
  fi
  cp "${from_target}/config.py" "${to_target}/config.py"
  patch_config_schedule "${to_target}/config.py" "${target_epoch}" "${TO_STAGE_EPOCH}"
  if [ ! -f "${to_target}/model/model_last.pth" ]; then
    cp --reflink=auto "${from_target}/model/model_last.pth" "${to_target}/model/model_last.pth"
  fi
}

submit_continue() {
  local arm="$1"
  local config="$2"
  local to_exp="$3"
  local alpha="$4"
  local beta="$5"
  local prior="$6"
  local vars
  vars="EXP_MIRROR_ROOT=${EXP_MIRROR_ROOT},CONFIG_NAME=${config},EXP_NAME=${to_exp},TRAIN_RESUME=true,RUN_PREFLIGHT=0,WEIGHT_PATH=None,CONCERTO_EPOCH=${TARGET_EPOCH},CONCERTO_STOP_EPOCH=${TO_STAGE_EPOCH},CONCERTO_GLOBAL_BATCH_SIZE=${CONCERTO_GLOBAL_BATCH_SIZE},CONCERTO_GRAD_ACCUM=${CONCERTO_GRAD_ACCUM},CONCERTO_NUM_WORKER=${CONCERTO_NUM_WORKER},CONCERTO_ENABLE_FLASH=${CONCERTO_ENABLE_FLASH},COORD_PROJECTION_ALPHA=${alpha},COORD_PROJECTION_BETA=${beta},COORD_PRIOR_PATH=${prior},GPU_IDS_CSV=$(pbs_v_escape "0,1,2,3"),NPROC_PER_NODE=4"
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
  local to_exp="$2"
  local alpha="$3"
  local beta="$4"
  local prior="$5"
  local linear_exp="scannet-proxy-${to_exp}-lin"
  local continue_ckpt="${REPO_ROOT}/exp/concerto/${to_exp}/model/model_last.pth"
  local gate_json="${SUMMARY_ROOT}/${linear_exp}_gate.json"
  local vars
  vars="EXP_MIRROR_ROOT=${EXP_MIRROR_ROOT},SUMMARY_ROOT=${SUMMARY_ROOT},EXP_TAG=${TO_EXP_TAG},CONTINUE_EXP=${to_exp},CONTINUE_CKPT=${continue_ckpt},LINEAR_EXP=${linear_exp},LINEAR_GATE_JSON=${gate_json},RUN_STRESS=0,RUN_LINEAR=1,FOLLOWUP_PARALLEL=0,LINEAR_GPU=0,LINEAR_GPU_IDS_CSV=$(pbs_v_escape "${LINEAR_GPU_IDS_CSV}"),LINEAR_NUM_GPU=${LINEAR_NUM_GPU},LINEAR_TRAIN_LAUNCHER=${LINEAR_TRAIN_LAUNCHER},SELECTED_ALPHA=${alpha},SELECTED_BETA=${beta},COORD_PRIOR_PATH=${prior},LINEAR_CONFIG=semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly"
  local cmd=(qsub -W "depend=afterok:${continue_job}" -l rt_QF=1 -l "walltime=${FOLLOWUP_WALLTIME}" -v "${vars}" tools/concerto_projection_shortcut/submit_projres_v1_followup_abciq_qf.sh)
  echo "+ ${cmd[*]}" >&2
  if [ "${DRY_RUN}" = "1" ]; then
    printf 'DRY_RUN'
  else
    "${cmd[@]}"
  fi
}

echo "=== Launch ProjRes long-horizon resume stage ==="
echo "date=$(date -Is)"
echo "from_stage_epoch=${FROM_STAGE_EPOCH}"
echo "to_stage_epoch=${TO_STAGE_EPOCH}"
echo "target_epoch=${TARGET_EPOCH}"
echo "rt_qf=${RT_QF}"
echo "continue_walltime=${CONTINUE_WALLTIME}"
echo "followup_walltime=${FOLLOWUP_WALLTIME}"
echo "from_exp_tag=${FROM_EXP_TAG}"
echo "to_exp_tag=${TO_EXP_TAG}"
echo "summary_root=${SUMMARY_ROOT}"
echo "linear_gpu_ids_csv=${LINEAR_GPU_IDS_CSV}"
echo "linear_num_gpu=${LINEAR_NUM_GPU}"
echo "linear_train_launcher=${LINEAR_TRAIN_LAUNCHER}"
echo "allow_schedule_mismatch=${ALLOW_SCHEDULE_MISMATCH}"
echo "dry_run=${DRY_RUN}"

arms=(
  "original|pretrain-concerto-v1m1-0-arkit-full-continue-h10016|arkit-full-original${FROM_EXP_TAG}-continue|arkit-full-original${TO_EXP_TAG}-continue|0.0|1.0|${COORD_PRIOR_PATH_DEFAULT}"
  "v1b-combo-b075-a001|pretrain-concerto-v1m1-0-arkit-full-projres-v1b-continue-h10016|arkit-full-projres-v1b-combo-b075-a001${FROM_EXP_TAG}-continue|arkit-full-projres-v1b-combo-b075-a001${TO_EXP_TAG}-continue|0.01|0.75|${COORD_PRIOR_PATH_DEFAULT}"
)

for spec in "${arms[@]}"; do
  IFS='|' read -r arm config from_exp to_exp alpha beta prior <<< "${spec}"
  prepare_resume_exp "${from_exp}" "${to_exp}" "${TARGET_EPOCH}"
  cont_job="$(submit_continue "${arm}" "${config}" "${to_exp}" "${alpha}" "${beta}" "${prior}")"
  follow_job="$(submit_followup "${cont_job}" "${to_exp}" "${alpha}" "${beta}" "${prior}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${arm}" "${config}" "${from_exp}" "${to_exp}" "${alpha}" "${beta}" "${prior}" \
    "${TARGET_EPOCH}" "${FROM_STAGE_EPOCH}" "${TO_STAGE_EPOCH}" \
    "${cont_job}" "${follow_job}" | tee -a "${MANIFEST}"
done

echo "[manifest] ${MANIFEST}"
