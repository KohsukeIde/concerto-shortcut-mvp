#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=06:00:00
#PBS -N scene_severity
#PBS -j oe

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}"
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}"

source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/external/Utonia:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/scene_support_severity_${CASE:-unset}_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CASE="${CASE:?set CASE to concerto_decoder, concerto_linear, sonata_linear, utonia, ptv3_scannet20, ptv3_scannet200, or ptv3_s3dis}"

RANDOM_KEEP_RATIOS="${RANDOM_KEEP_RATIOS:-0.8,0.5,0.2,0.1}"
STRUCTURED_KEEP_RATIOS="${STRUCTURED_KEEP_RATIOS:-0.8,0.5,0.2,0.1}"
MASKED_MODEL_KEEP_RATIOS="${MASKED_MODEL_KEEP_RATIOS:-0.5,0.2,0.1}"
FIXED_POINT_COUNTS="${FIXED_POINT_COUNTS:-16000,8000,4000}"
FEATURE_ZERO_RATIOS="${FEATURE_ZERO_RATIOS:-1.0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
MAX_VAL_SCENES="${MAX_VAL_SCENES:--1}"
NUM_WORKER="${NUM_WORKER:-8}"
FULL_SCENE_CHUNK_SIZE="${FULL_SCENE_CHUNK_SIZE:-2048}"

echo "=== Scene support-stress severity ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "case=${CASE}"
echo "repo_root=${REPO_ROOT}"
echo "random_keep_ratios=${RANDOM_KEEP_RATIOS}"
echo "structured_keep_ratios=${STRUCTURED_KEEP_RATIOS}"
echo "masked_model_keep_ratios=${MASKED_MODEL_KEEP_RATIOS}"
echo "fixed_point_counts=${FIXED_POINT_COUNTS}"
nvidia-smi -L || true

run_pointcept_masking() {
  local config="$1"
  local weight="$2"
  local method="$3"
  local data_root="$4"
  local out_dir="$5"
  local summary="$6"
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_masking_battery.py \
    --config "${config}" \
    --weight "${weight}" \
    --method-name "${method}" \
    --data-root "${data_root}" \
    --output-dir "${out_dir}" \
    --max-val-batches "${MAX_VAL_BATCHES}" \
    --random-keep-ratios "${RANDOM_KEEP_RATIOS}" \
    --structured-keep-ratios "${STRUCTURED_KEEP_RATIOS}" \
    --masked-model-keep-ratios "${MASKED_MODEL_KEEP_RATIOS}" \
    --fixed-point-counts "${FIXED_POINT_COUNTS}" \
    --feature-zero-ratios "${FEATURE_ZERO_RATIOS}" \
    --full-scene-scoring \
    --full-scene-chunk-size "${FULL_SCENE_CHUNK_SIZE}" \
    --summary-prefix "${summary}"
}

run_ptv3_masking() {
  local config="$1"
  local weight="$2"
  local method="$3"
  local data_root="$4"
  local segment_key="$5"
  local num_classes="$6"
  local focus="$7"
  local confusion="$8"
  local out_dir="$9"
  local summary="${10}"
  local split="${11:-val}"
  "${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_ptv3_v151_masking_compat.py \
    --official-root data/tmp/Pointcept-v1.5.1 \
    --config "${config}" \
    --weight "${weight}" \
    --method-name "${method}" \
    --data-root "${data_root}" \
    --split "${split}" \
    --segment-key "${segment_key}" \
    --num-classes "${num_classes}" \
    --focus-class "${focus}" \
    --confusion-class "${confusion}" \
    --output-dir "${out_dir}" \
    --max-val-batches "${MAX_VAL_BATCHES}" \
    --random-keep-ratios "${RANDOM_KEEP_RATIOS}" \
    --structured-keep-ratios "${STRUCTURED_KEEP_RATIOS}" \
    --masked-model-keep-ratios "${MASKED_MODEL_KEEP_RATIOS}" \
    --fixed-point-counts "${FIXED_POINT_COUNTS}" \
    --feature-zero-ratios "${FEATURE_ZERO_RATIOS}" \
    --full-scene-scoring \
    --full-scene-chunk-size "${FULL_SCENE_CHUNK_SIZE}" \
    --summary-prefix "${summary}"
}

case "${CASE}" in
  concerto_decoder)
    run_pointcept_masking \
      configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py \
      data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth \
      concerto_decoder_origin_severity \
      data/scannet \
      data/runs/support_severity/concerto_decoder \
      tools/concerto_projection_shortcut/results_support_severity_concerto_decoder
    ;;
  concerto_linear)
    run_pointcept_masking \
      configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-lin-origin-e100.py \
      data/runs/scannet_lora_origin/exp/scannet-lin-origin-e100/model/model_best.pth \
      concerto_linear_origin_severity \
      data/scannet \
      data/runs/support_severity/concerto_linear \
      tools/concerto_projection_shortcut/results_support_severity_concerto_linear
    ;;
  sonata_linear)
    run_pointcept_masking \
      configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py \
      data/weights/sonata/sonata_scannet_linear_merged.pth \
      sonata_linear_scannet_severity \
      data/scannet \
      data/runs/support_severity/sonata_linear \
      tools/concerto_projection_shortcut/results_support_severity_sonata_linear
    ;;
  utonia)
    "${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_utonia_scannet_support_stress.py \
      --repo-root "${REPO_ROOT}" \
      --data-root data/scannet \
      --utonia-weight "${REPO_ROOT}/data/weights/utonia/utonia.pth" \
      --seg-head-weight "${REPO_ROOT}/data/weights/utonia/utonia_linear_prob_head_sc.pth" \
      --output-dir "${REPO_ROOT}/tools/concerto_projection_shortcut/results_support_severity_utonia" \
      --random-keep-ratios "${RANDOM_KEEP_RATIOS}" \
      --structured-keep-ratios "${STRUCTURED_KEEP_RATIOS}" \
      --masked-model-keep-ratios "${MASKED_MODEL_KEEP_RATIOS}" \
      --fixed-point-counts "${FIXED_POINT_COUNTS}" \
      --max-val-scenes "${MAX_VAL_SCENES}" \
      --batch-size 1 \
      --num-worker 2 \
      --feature-zero \
      --disable-flash
    ;;
  ptv3_scannet20)
    run_ptv3_masking \
      configs/scannet/semseg-pt-v3m1-0-base.py \
      data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth \
      ptv3_scannet20_v151_severity \
      data/scannet \
      segment20 20 picture wall \
      data/runs/support_severity/ptv3_scannet20 \
      tools/concerto_projection_shortcut/results_support_severity_ptv3_scannet20
    ;;
  ptv3_scannet200)
    run_ptv3_masking \
      configs/scannet200/semseg-pt-v3m1-0-base.py \
      data/weights/ptv3/scannet200-semseg-pt-v3m1-0-base/model/model_best.pth \
      ptv3_scannet200_v151_severity \
      /groups/qgah50055/ide/3d-sans-3dscans/scannet \
      segment200 200 picture wall \
      data/runs/support_severity/ptv3_scannet200 \
      tools/concerto_projection_shortcut/results_support_severity_ptv3_scannet200
    ;;
  ptv3_s3dis)
    run_ptv3_masking \
      configs/s3dis/semseg-pt-v3m1-0-rpe.py \
      data/weights/ptv3/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth \
      ptv3_s3dis_v151_severity \
      data/concerto_s3dis_imagepoint/s3dis \
      segment 13 board wall \
      data/runs/support_severity/ptv3_s3dis \
      tools/concerto_projection_shortcut/results_support_severity_ptv3_s3dis \
      Area_5
    ;;
  *)
    echo "unknown CASE=${CASE}" >&2
    exit 2
    ;;
esac

echo "[done] scene support-stress severity ${CASE}"
echo "[log] ${LOG_FILE}"
