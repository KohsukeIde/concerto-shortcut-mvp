#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

CONFIG_NAME="${CONFIG_NAME:-pretrain-concerto-v1m1-2-large-video-sr-lora-v5-phasea}"
DATASET_NAME="${DATASET_NAME:-concerto}"
EXP_NAME="${EXP_NAME:-sr-lora-v5-r${SR_LORA_RANK:-4}-d${SR_DISTILL_WEIGHT:-0.3}}"
OFFICIAL_WEIGHT="${OFFICIAL_WEIGHT:-${WEIGHT_DIR}/pretrain-concerto-v1m1-2-large-video.pth}"
WEIGHT_PATH="${WEIGHT_PATH:-${OFFICIAL_WEIGHT}}"
COORD_RIVAL_PATH="${COORD_RIVAL_PATH:?COORD_RIVAL_PATH is required}"

export CONFIG_NAME DATASET_NAME EXP_NAME WEIGHT_PATH COORD_RIVAL_PATH
export POINTCEPT_TRAIN_LAUNCHER="${POINTCEPT_TRAIN_LAUNCHER:-torchrun}"
export SR_LORA_RANK="${SR_LORA_RANK:-4}"
export SR_LORA_ALPHA="${SR_LORA_ALPHA:-$((SR_LORA_RANK * 2))}"
export SR_LORA_DROPOUT="${SR_LORA_DROPOUT:-0.05}"
export SR_MARGIN_ALPHA="${SR_MARGIN_ALPHA:-1.0}"
export SR_MARGIN_VALUE="${SR_MARGIN_VALUE:-0.1}"
export SR_DISTILL_WEIGHT="${SR_DISTILL_WEIGHT:-0.3}"
export CONCERTO_EPOCH="${CONCERTO_EPOCH:-5}"
export CONCERTO_EVAL_EPOCH="${CONCERTO_EVAL_EPOCH:-${CONCERTO_EPOCH}}"
export CONCERTO_STOP_EPOCH="${CONCERTO_STOP_EPOCH:-${CONCERTO_EPOCH}}"
export CONCERTO_ENABLE_FLASH="${CONCERTO_ENABLE_FLASH:-1}"

if [ ! -f "${WEIGHT_PATH}" ]; then
  echo "[error] weight not found: ${WEIGHT_PATH}" >&2
  exit 2
fi
if [ ! -f "${COORD_RIVAL_PATH}" ]; then
  echo "[error] coord rival checkpoint not found: ${COORD_RIVAL_PATH}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${WEIGHT_PATH}" "${COORD_RIVAL_PATH}" <<'PY'
import sys
import torch

weight, rival = sys.argv[1:3]
ckpt = torch.load(weight, map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
if not any("enc2d_head" in key for key in state):
    raise SystemExit(f"[error] {weight} does not contain enc2d_head weights")
if not any("patch_proj" in key for key in state):
    raise SystemExit(f"[error] {weight} does not contain patch_proj weights")
rival_ckpt = torch.load(rival, map_location="cpu", weights_only=False)
print(
    "[ok] weights: "
    f"phase_a={weight} keys={len(state)} "
    f"rival={rival} target_dim={rival_ckpt.get('target_dim')}"
)
PY

echo "=== SR-LoRA Phase A ==="
echo "date=$(date -Is)"
echo "repo_root=${REPO_ROOT}"
echo "config=${CONFIG_NAME}"
echo "dataset=${DATASET_NAME}"
echo "exp=${EXP_NAME}"
echo "weight=${WEIGHT_PATH}"
echo "coord_rival=${COORD_RIVAL_PATH}"
echo "sr_rank=${SR_LORA_RANK}"
echo "sr_alpha=${SR_LORA_ALPHA}"
echo "sr_dropout=${SR_LORA_DROPOUT}"
echo "sr_margin_alpha=${SR_MARGIN_ALPHA}"
echo "sr_margin_value=${SR_MARGIN_VALUE}"
echo "sr_distill_weight=${SR_DISTILL_WEIGHT}"
echo "concerto_epoch=${CONCERTO_EPOCH}"
echo "concerto_global_batch_size=${CONCERTO_GLOBAL_BATCH_SIZE:-}"
echo "concerto_num_worker=${CONCERTO_NUM_WORKER:-}"

bash "${REPO_ROOT}/tools/concerto_projection_shortcut/run_pointcept_train_multinode_pbsdsh.sh"
