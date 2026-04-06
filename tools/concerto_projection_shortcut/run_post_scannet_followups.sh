#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

CONDA_ROOT="${CONDA_ROOT:-/home/cvrt/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointcept-cu128}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_NAME="${DATASET_NAME:-concerto}"
POLL_SECONDS="${POLL_SECONDS:-300}"
WAIT_CHECKPOINT="${WAIT_CHECKPOINT:-${REPO_ROOT}/exp/${DATASET_NAME}/scannet-proxy-coord-mlp-continue-lin/model/model_last.pth}"
FOLLOWUP_CONFIG="${FOLLOWUP_CONFIG:-pretrain-concerto-v1m1-0-probe-enc2d-full-prepool-global-feature-index-permutation}"
FOLLOWUP_NAME="${FOLLOWUP_NAME:-arkit-full-causal-prepool-global-feature-index-permutation}"
FOLLOWUP_CKPT="${REPO_ROOT}/exp/${DATASET_NAME}/${FOLLOWUP_NAME}/model/model_last.pth"
SUMMARY_CSV="${SUMMARY_CSV:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_arkit_followup.csv}"
SUMMARY_MD="${SUMMARY_MD:-${REPO_ROOT}/tools/concerto_projection_shortcut/results_arkit_followup.md}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/tools/concerto_projection_shortcut/logs/post_scannet_followups.log}"

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

mkdir -p "$(dirname "${LOG_PATH}")"

{
  echo "[wait] ${WAIT_CHECKPOINT}"
  while [ ! -f "${WAIT_CHECKPOINT}" ]; do
    date "+[wait] %F %T waiting for ScanNet replacement gate completion"
    sleep "${POLL_SECONDS}"
  done

  if [ ! -f "${FOLLOWUP_CKPT}" ]; then
    CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh \
      -p "${PYTHON_BIN}" \
      -d "${DATASET_NAME}" \
      -g 1 \
      -c "${FOLLOWUP_CONFIG}" \
      -n "${FOLLOWUP_NAME}"
  else
    echo "[skip] ${FOLLOWUP_NAME} already has ${FOLLOWUP_CKPT}"
  fi

  "${PYTHON_BIN}" tools/concerto_projection_shortcut/summarize_logs.py \
    "${REPO_ROOT}/exp/${DATASET_NAME}/${FOLLOWUP_NAME}/train.log" > "${SUMMARY_CSV}"

  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))
lines = [
    "# ARKit Follow-up",
    "",
    "| experiment | enc2d first | enc2d last | enc2d min | count |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for row in rows:
    exp_name = Path(row["log"]).parent.name
    lines.append(
        f"| {exp_name} | {row['enc2d_loss_first']} | {row['enc2d_loss_last']} | "
        f"{row['enc2d_loss_min']} | {row['enc2d_loss_count']} |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
} | tee -a "${LOG_PATH}"
