#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def default_specs(repo_root: Path):
    data = Path(os.environ.get("POINTCEPT_DATA_ROOT", repo_root / "data"))
    return [
        ("arkit", data / "arkitscenes_absmeta", ["Training"], ["Validation"]),
        ("scannet", data / "concerto_scannet_imagepoint_absmeta", ["train"], ["val"]),
        ("scannetpp", data / "concerto_scannetpp_imagepoint_absmeta", ["train"], ["val"]),
        (
            "s3dis",
            data / "concerto_s3dis_imagepoint_absmeta",
            ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
            ["Area_5"],
        ),
        ("hm3d", data / "concerto_hm3d_imagepoint_absmeta", ["train"], ["val"]),
        (
            "structured3d",
            data / "concerto_structured3d_imagepoint_absmeta",
            ["train"],
            ["val"],
        ),
    ]


def check_record(root: Path, split: str):
    split_file = root / "splits" / f"{split}.json"
    if not split_file.is_file():
        return False, f"missing split: {split_file}"
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    if not payload:
        return False, f"empty split: {split_file}"
    key, record = next(iter(payload.items()))
    pointcloud = Path(record["pointclouds"])
    image = Path(record["images"][0]) if record.get("images") else None
    corr = Path(record["correspondences"][0]) if record.get("correspondences") else None
    missing = []
    if not pointcloud.is_dir():
        missing.append(str(pointcloud))
    if image is None or not image.is_file():
        missing.append(str(image) if image else "images:<empty>")
    if corr is None or not corr.is_file():
        missing.append(str(corr) if corr else "correspondences:<empty>")
    if missing:
        return False, f"{key} has unresolved assets: {missing}"
    return True, f"{key}: pointclouds/images/correspondences resolved"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Concerto six-dataset absmeta roots.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    failed = False
    for name, root, train_splits, val_splits in default_specs(args.repo_root.resolve()):
        root = root.resolve()
        print(f"[dataset] {name}: {root}")
        for split in [*train_splits, *val_splits]:
            ok, message = check_record(root, split)
            print(f"  [{'ok' if ok else 'missing'}] {split}: {message}")
            failed = failed or not ok
    if failed and not args.allow_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
