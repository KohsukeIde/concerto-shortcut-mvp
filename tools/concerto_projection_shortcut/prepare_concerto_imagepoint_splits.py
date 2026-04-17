#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_PREFIX = {
    "arkit": "data/arkitscenes",
    "arkitscenes": "data/arkitscenes",
    "scannet": "data/scannet",
    "scannetpp": "data/scannetpp",
    "s3dis": "data/s3dis",
    "hm3d": "data/hm3d",
    "structured3d": "data/structured3d",
}


def resolve_source_root(root: Path) -> Path:
    root = root.resolve()
    if (root / "splits").is_dir():
        return root
    candidates = sorted(
        {
            path.parent
            for path in root.glob("**/splits/*.json")
            if path.parent.name == "splits"
        }
    )
    if not candidates:
        raise FileNotFoundError(f"missing splits/*.json under {root}")
    return candidates[0].parent


def rewrite_path(value: str, source_root: Path, prefix: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    normalized = value.replace("\\", "/")
    prefix = prefix.rstrip("/") + "/"
    if normalized.startswith(prefix):
        return str(source_root.joinpath(normalized[len(prefix) :]).resolve())
    return str(source_root.joinpath(normalized).resolve())


def rewrite_record(record: dict, source_root: Path, prefix: str) -> dict:
    output = dict(record)
    output["pointclouds"] = rewrite_path(record["pointclouds"], source_root, prefix)
    output["images"] = [
        rewrite_path(item, source_root, prefix) for item in record["images"]
    ]
    output["correspondences"] = [
        rewrite_path(item, source_root, prefix) for item in record["correspondences"]
    ]
    return output


def verify_one_record(record: dict, split_file: Path) -> None:
    missing = []
    pointcloud = Path(record["pointclouds"])
    if not pointcloud.is_dir():
        missing.append(str(pointcloud))
    if not record["images"]:
        missing.append("images:<empty>")
    elif not Path(record["images"][0]).is_file():
        missing.append(str(record["images"][0]))
    if not record["correspondences"]:
        missing.append("correspondences:<empty>")
    elif not Path(record["correspondences"][0]).is_file():
        missing.append(str(record["correspondences"][0]))
    if missing:
        raise FileNotFoundError(
            f"{split_file} first sample does not resolve required assets: {missing}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite Concerto image-point split JSONs to absolute paths."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PREFIX))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    source_root = resolve_source_root(args.source_root)
    output_root = args.output_root.resolve()
    output_splits = output_root / "splits"
    output_splits.mkdir(parents=True, exist_ok=True)
    prefix = DATASET_PREFIX[args.dataset]

    for split_file in sorted((source_root / "splits").glob("*.json")):
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        rewritten = {
            key: rewrite_record(record, source_root, prefix)
            for key, record in payload.items()
        }
        target = output_splits / split_file.name
        target.write_text(json.dumps(rewritten), encoding="utf-8")
        if args.verify and rewritten:
            verify_one_record(next(iter(rewritten.values())), target)
        print(
            f"[ok] {args.dataset}: wrote {target} "
            f"({len(rewritten)} samples; source={source_root})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
