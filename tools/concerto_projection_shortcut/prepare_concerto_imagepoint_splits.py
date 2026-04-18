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


def synthesize_structured3d_splits(root: Path) -> Path:
    dataset_root = root.resolve()
    if (dataset_root / "structured3d").is_dir():
        dataset_root = dataset_root / "structured3d"
    images_root = dataset_root / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(f"missing Structured3D images root under {dataset_root}")

    split_root = dataset_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    prefix = Path("data/structured3d")

    for split in ("train", "val", "test"):
        image_split = images_root / split
        point_split = dataset_root / split
        if not image_split.is_dir() or not point_split.is_dir():
            continue

        payload = {}
        for scene_dir in sorted(path for path in image_split.iterdir() if path.is_dir()):
            for room_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
                color_dir = room_dir / "color" / "prsp"
                corr_dir = room_dir / "correspondence" / "prsp_correspondence"
                point_dir = point_split / scene_dir.name / room_dir.name
                if not color_dir.is_dir() or not corr_dir.is_dir() or not point_dir.is_dir():
                    continue

                image_files = sorted(color_dir.glob("*.png"), key=lambda path: int(path.stem))
                if not image_files:
                    continue

                key = f"{scene_dir.name}_{room_dir.name}"
                payload[key] = {
                    "pointclouds": str((prefix / split / scene_dir.name / room_dir.name).as_posix()),
                    "images": [
                        str(
                            (
                                prefix
                                / "images"
                                / split
                                / scene_dir.name
                                / room_dir.name
                                / "color"
                                / "prsp"
                                / image.name
                            ).as_posix()
                        )
                        for image in image_files
                    ],
                    "correspondences": [
                        str(
                            (
                                prefix
                                / "images"
                                / split
                                / scene_dir.name
                                / room_dir.name
                                / "correspondence"
                                / "prsp_correspondence"
                                / f"{image.stem}.npy"
                            ).as_posix()
                        )
                        for image in image_files
                    ],
                }

        if payload:
            (split_root / f"{split}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

    return dataset_root


def resolve_source_root(root: Path, dataset: str) -> Path:
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
    if not candidates and dataset == "structured3d":
        structured_root = synthesize_structured3d_splits(root)
        candidates = [structured_root]
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

    source_root = resolve_source_root(args.source_root, args.dataset)
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
