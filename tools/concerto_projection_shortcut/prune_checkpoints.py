#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024 / 1024:.1f} GiB"


def should_skip(path: Path, skip_tokens: list[str]) -> bool:
    path_str = str(path)
    return any(token and token in path_str for token in skip_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune saved epoch checkpoints while keeping model_last/model_best."
    )
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path("/home/cvrt/Desktop/Concerto/exp"),
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument(
        "--skip-token",
        action="append",
        default=[],
        help="Substring path filter to skip deleting matching files/directories.",
    )
    parser.add_argument(
        "--delete-stale-dirs",
        action="store_true",
        help="Also remove experiment directories whose names contain 'stale'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without deleting it.",
    )
    args = parser.parse_args()

    exp_root = args.exp_root.resolve()
    if not exp_root.exists():
        raise SystemExit(f"exp root not found: {exp_root}")

    deleted_files = 0
    deleted_bytes = 0
    for path in sorted(exp_root.rglob("epoch_*.pth")):
        if should_skip(path, args.skip_token):
            continue
        size = path.stat().st_size
        deleted_files += 1
        deleted_bytes += size
        print(f"[delete-file] {path} ({format_gib(size)})")
        if not args.dry_run:
            path.unlink()

    deleted_dirs = 0
    if args.delete_stale_dirs:
        for path in sorted(exp_root.rglob("*")):
            if not path.is_dir():
                continue
            if "stale" not in path.name:
                continue
            if should_skip(path, args.skip_token):
                continue
            size = sum(
                child.stat().st_size
                for child in path.rglob("*")
                if child.is_file()
            )
            deleted_dirs += 1
            deleted_bytes += size
            print(f"[delete-dir]  {path} ({format_gib(size)})")
            if not args.dry_run:
                shutil.rmtree(path)

    print(
        f"[summary] files={deleted_files} dirs={deleted_dirs} reclaimed={format_gib(deleted_bytes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
