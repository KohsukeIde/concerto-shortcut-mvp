#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=06:00:00
#PBS -N xmodel_resid_tr
#PBS -j oe

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}" 2>/dev/null || module load python/3.11
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}" 2>/dev/null || module load cuda/12.6

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export PYTHONPATH="${REPO_ROOT}/external/Utonia:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${MODE:-feature}"
if [ "${MODE}" = "logit" ]; then
  FEATURE_FLAG="--no-feature-pairs"
  MODE_SUFFIX="logit"
else
  FEATURE_FLAG=""
  MODE_SUFFIX="feature"
fi

MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-384}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:--1}"
EPOCHS="${EPOCHS:-12}"
SAMPLE_TRAIN="${SAMPLE_TRAIN:-2048}"
SAMPLE_HELDOUT="${SAMPLE_HELDOUT:-4096}"
KL_WEIGHTS="${KL_WEIGHTS:-0.0,0.03}"
SAFE_CE_WEIGHTS="${SAFE_CE_WEIGHTS:-2.0,4.0}"
WITH_PTV3="${WITH_PTV3:-0}"

OUT_DIR="data/runs/cross_model_residual_fusion_scannet20/train_split_${MODE_SUFFIX}_ssl"
PREFIX="tools/concerto_projection_shortcut/results_cross_model_residual_fusion_scannet20_train_split_${MODE_SUFFIX}_ssl"
PTV3_ARG=()
if [ "${WITH_PTV3}" = "1" ]; then
  OUT_DIR="data/runs/cross_model_residual_fusion_scannet20/train_split_${MODE_SUFFIX}_with_ptv3"
  PREFIX="tools/concerto_projection_shortcut/results_cross_model_residual_fusion_scannet20_train_split_${MODE_SUFFIX}_with_ptv3"
  PTV3_ARG=(--cached-expert "PTv3_supervised::data/runs/ptv3_v151_raw_probs_scannet20/train")
fi

LOG_DIR="${POINTCEPT_DATA_ROOT}/logs/abciq"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_split_residual_fusion_${MODE_SUFFIX}_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Train-split residual fusion ScanNet20 ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "mode=${MODE}"
echo "with_ptv3=${WITH_PTV3}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "epochs=${EPOCHS}"
echo "output_dir=${OUT_DIR}"
nvidia-smi -L || true

"${PYTHON_BIN}" tools/concerto_projection_shortcut/train_cross_model_residual_fusion_scannet20.py \
  --repo-root "${REPO_ROOT}" \
  --data-root "${SCANNET_DATA_ROOT:-${SCANNET_EXTRACT_DIR}}" \
  --output-dir "${OUT_DIR}" \
  --summary-prefix "${PREFIX}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --num-worker "${NUM_WORKER:-4}" \
  --full-scene-chunk-size "${FULL_SCENE_CHUNK_SIZE:-2048}" \
  --sample-points-per-train-scene "${SAMPLE_TRAIN}" \
  --sample-points-per-heldout-scene "${SAMPLE_HELDOUT}" \
  --feature-proj-dim "${FEATURE_PROJ_DIM:-64}" \
  --epochs "${EPOCHS}" \
  --batch-size-train "${TRAIN_BATCH_SIZE:-8192}" \
  --kl-weights "${KL_WEIGHTS}" \
  --safe-ce-weights "${SAFE_CE_WEIGHTS}" \
  --include-utonia \
  ${FEATURE_FLAG} \
  "${PTV3_ARG[@]}" \
  ${EXTRA_ARGS:-}

echo "[done] train-split residual fusion ScanNet20"
echo "[log] ${LOG_FILE}"
