#!/usr/bin/env bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -N concerto_env_setup
#PBS -j oe

set -euo pipefail

cd "${WORKDIR:-/groups/qgah50055/ide/concerto-shortcut-mvp}" || exit 1

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11/3.11.14}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE}" 2>/dev/null || true
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source tools/concerto_projection_shortcut/device_defaults.sh

mkdir -p "${POINTCEPT_DATA_ROOT}/logs/abciq"
LOG_PATH="${LOG_PATH:-${POINTCEPT_DATA_ROOT}/logs/abciq/env_setup_${PBS_JOBID:-manual}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "=== ABCI-Q Concerto env setup ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=$(pwd -P)"
echo "venv=${VENV_DIR}"
echo "python_module=${PYTHON_MODULE}"
echo "cuda_module=${CUDA_MODULE}"
echo "install_flash_attn=${INSTALL_FLASH_ATTN:-1}"
echo "flash_attn_spec=${FLASH_ATTN_SPEC:-flash-attn==2.7.4.post1}"
echo "flash_attention_force_build=${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"

bash tools/concerto_projection_shortcut/create_venv_abciq.sh

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_XET_CACHE TORCH_HOME

nvidia-smi -L || true

python - <<'PY'
import importlib
import torch

print("python", __import__("sys").version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())

for name in ("torch_scatter", "spconv", "pointops", "transformers", "pointcept", "flash_attn"):
    mod = importlib.import_module(name)
    print(name, "OK", getattr(mod, "__version__", ""))
PY

echo "[done] env setup and import validation completed"
echo "[log] ${LOG_PATH}"
