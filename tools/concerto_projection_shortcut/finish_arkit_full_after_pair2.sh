#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

POLL_SECONDS="${POLL_SECONDS:-300}"
CONDA_ROOT="${CONDA_ROOT:-/home/cvrt/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointcept-cu128}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_NAME="${DATASET_NAME:-concerto}"
ARKIT_META_ROOT="${ARKIT_META_ROOT:-/home/cvrt/datasets/arkitscenes/arkitscenes_absmeta}"
DONE_STAMP="${DONE_STAMP:-${REPO_ROOT}/tools/concerto_projection_shortcut/arkit_full_causal.done}"
STRESS_CSV="${STRESS_CSV:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_arkit_full_stress.csv}"
STRESS_MD="${STRESS_MD:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_arkit_full_stress.md}"
CAUSAL_CSV="${CAUSAL_CSV:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_arkit_full_causal.csv}"
CAUSAL_MD="${CAUSAL_MD:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_arkit_full_causal.md}"

if ! command -v conda >/dev/null 2>&1; then
  set +u
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  set -u
fi
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]; then
  set +u
  conda activate "${CONDA_ENV_NAME}"
  set -u
fi

wait_for_exit() {
  local pattern="$1"
  while pgrep -af "${pattern}" >/dev/null 2>&1; do
    date "+[wait] %F %T still waiting for ${pattern}"
    sleep "${POLL_SECONDS}"
  done
}

GLOBAL_CKPT="exp/${DATASET_NAME}/arkit-full-causal-global-target-permutation/model/model_last.pth"
CROSS_SCENE_CKPT="exp/${DATASET_NAME}/arkit-full-causal-cross-scene-target-swap/model/model_last.pth"
COORD_RESIDUAL_CKPT="exp/${DATASET_NAME}/arkit-full-causal-coord-residual-target/model/model_last.pth"
COORD_RESIDUAL_PATTERN="pretrain-concerto-v1m1-0-probe-enc2d-full-coord-residual-target -n arkit-full-causal-coord-residual-target"

wait_for_exit "pretrain-concerto-v1m1-0-probe-enc2d-full-global-target-permutation -n arkit-full-causal-global-target-permutation"
wait_for_exit "pretrain-concerto-v1m1-0-probe-enc2d-full-cross-scene-target-swap -n arkit-full-causal-cross-scene-target-swap"

if [ ! -f "${GLOBAL_CKPT}" ]; then
  echo "[warn] missing ${GLOBAL_CKPT}"
fi
if [ ! -f "${CROSS_SCENE_CKPT}" ]; then
  echo "[warn] missing ${CROSS_SCENE_CKPT}"
fi

if pgrep -af "${COORD_RESIDUAL_PATTERN}" >/dev/null 2>&1; then
  wait_for_exit "${COORD_RESIDUAL_PATTERN}"
elif [ ! -f "${COORD_RESIDUAL_CKPT}" ]; then
  echo "[run] coord_residual_target on GPU 0"
  CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh \
    -p "${PYTHON_BIN}" \
    -d "${DATASET_NAME}" \
    -g 1 \
    -c pretrain-concerto-v1m1-0-probe-enc2d-full-coord-residual-target \
    -n arkit-full-causal-coord-residual-target
fi

echo "checkpoint,stress,batches,enc2d_loss_mean" > "${STRESS_CSV}"
for exp_name in \
  arkit-full-causal-baseline \
  arkit-full-causal-coord-mlp \
  arkit-full-causal-coord-residual-target; do
  checkpoint="exp/${DATASET_NAME}/${exp_name}/model/model_last.pth"
  if [ ! -f "${checkpoint}" ]; then
    echo "[warn] skip stress for missing ${checkpoint}"
    continue
  fi
  tmp="$(mktemp)"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" tools/concerto_projection_shortcut/eval_enc2d_stress.py \
    --config pretrain-concerto-v1m1-0-probe-enc2d-full-baseline \
    --weight "${checkpoint}" \
    --data-root "${ARKIT_META_ROOT}" \
    --max-batches 20 > "${tmp}"
  tail -n +2 "${tmp}" | sed "s#^#${exp_name},#" >> "${STRESS_CSV}"
  rm -f "${tmp}"
done

"${PYTHON_BIN}" - "${STRESS_CSV}" "${STRESS_MD}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))
lines = [
    "# ARKit Full Stress",
    "",
    "| checkpoint | stress | batches | enc2d loss mean |",
    "| --- | --- | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| {row['checkpoint']} | {row['stress']} | {row['batches']} | {row['enc2d_loss_mean']} |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

"${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_logs.py \
  "exp/${DATASET_NAME}/arkit-full-causal-*/train.log" > "${CAUSAL_CSV}"

"${PYTHON_BIN}" - "${CAUSAL_CSV}" "${CAUSAL_MD}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))
lines = [
    "# ARKit Full Causal Branch",
    "",
    "| experiment | enc2d first | enc2d last | enc2d min | count |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for row in rows:
    exp_name = Path(row["log"]).parent.name
    lines.append(
        f"| {exp_name} | {row['enc2d_loss_first']} | {row['enc2d_loss_last']} | {row['enc2d_loss_min']} | {row['enc2d_loss_count']} |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

date '+%F %T' > "${DONE_STAMP}"
echo "[done] wrote ${CAUSAL_CSV}, ${CAUSAL_MD}, ${STRESS_CSV}, ${STRESS_MD}, ${DONE_STAMP}"
