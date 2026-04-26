#!/usr/bin/env bash
#PBS -l rt_QF=1
#PBS -l walltime=02:00:00
#PBS -N s3dis_coord_hv
#PBS -j oe

set -euo pipefail

cd "${PBS_O_WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
REPO_ROOT="$(pwd -P)"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11/3.11.14}" 2>/dev/null || true
module load "${CUDA_MODULE:-cuda/12.6/12.6.2}" 2>/dev/null || true

# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"
ensure_venv_active

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_ROOT="${REPO_ROOT}/data/runs/main_variant_coord_mlp_rival/s3dis-highval-probe"

python tools/concerto_projection_shortcut/fit_main_variant_coord_mlp_rival.py \
  --repo-root "${REPO_ROOT}" \
  --weight "${WEIGHT_DIR}/concerto_base_origin.pth" \
  --datasets s3dis \
  --output-root "${OUT_ROOT}" \
  --tag s3dis-highval-probe \
  --max-train-batches-per-dataset 256 \
  --max-val-batches-per-dataset 512 \
  --max-rows-per-batch 4096 \
  --prior-epochs 80 \
  --prior-batch-size 8192 \
  --num-worker 2 \
  --causal-csv "${REPO_ROOT}/tools/concerto_projection_shortcut/results_main_variant_causal_battery.csv" \
  --skip-repo-results

python - <<'PY'
import json
from pathlib import Path
root = Path("data/runs/main_variant_coord_mlp_rival/s3dis-highval-probe")
m = json.loads((root / "metrics.json").read_text())
print("[summary] per_dataset", m["per_dataset_scaled_cosine_loss"])
for split in ["train", "val"]:
    import torch
    c = torch.load(root / "cache" / f"s3dis_{split}.pt", map_location="cpu", weights_only=False)
    print("[summary]", split, "rows", c["coord"].shape[0], "target_dim", c["target_dim"])
PY
