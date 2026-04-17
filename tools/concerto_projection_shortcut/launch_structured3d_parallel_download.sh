#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"

SHARD_COUNT="${SHARD_COUNT:-8}"
WALLTIME="${WALLTIME:-08:00:00}"
HF_REPO_ID="${HF_REPO_ID:-Pointcept/concerto_structured3d_compressed}"
HF_LOCAL_DIR="${HF_LOCAL_DIR:-${REPO_ROOT}/data/concerto_structured3d_compressed}"
HF_INCLUDE_PATTERN="${HF_INCLUDE_PATTERN:-structured3d.tar.gz.part}"

job_ids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  job_id=$(qsub -V \
    -l walltime="${WALLTIME}" \
    -N "s3d_dl_${shard_index}" \
    -v REPO_ROOT="${REPO_ROOT}",HF_REPO_ID="${HF_REPO_ID}",HF_LOCAL_DIR="${HF_LOCAL_DIR}",HF_INCLUDE_PATTERN="${HF_INCLUDE_PATTERN}",SHARD_COUNT="${SHARD_COUNT}",SHARD_INDEX="${shard_index}" \
    tools/concerto_projection_shortcut/submit_hf_dataset_download_shard_abciq_qc.sh)
  job_ids+=("${job_id}")
  echo "submitted shard ${shard_index}: ${job_id}"
done

printf 'job_ids=%s\n' "${job_ids[*]}"
(
  IFS=:
  printf 'afterok=%s\n' "${job_ids[*]}"
)
