#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one shard of a HF dataset repo.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--include-pattern", default="")
    args = parser.parse_args()

    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")

    api = HfApi()
    files = sorted(api.list_repo_files(args.repo_id, repo_type=args.repo_type))
    if args.include_pattern:
        files = [path for path in files if args.include_pattern in path]
    if not files:
        raise RuntimeError(f"no files matched in {args.repo_id}")

    selected = [
        path
        for idx, path in enumerate(files)
        if idx % args.shard_count == args.shard_index
    ]
    args.local_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[shard] repo={args.repo_id} shard={args.shard_index}/{args.shard_count} "
        f"files={len(selected)} local_dir={args.local_dir}"
    )
    for path in selected:
        print(f"[download] {path}")
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=path,
            local_dir=str(args.local_dir),
        )
    print("[done] shard completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
