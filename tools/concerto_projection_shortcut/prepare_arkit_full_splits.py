#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rewrite_path(source_root: Path, raw_path):
    prefix = "data/arkitscenes/"
    if isinstance(raw_path, str):
        if raw_path.startswith(prefix):
            return str(source_root / raw_path[len(prefix) :])
        return raw_path
    if isinstance(raw_path, list):
        return [rewrite_path(source_root, item) for item in raw_path]
    return raw_path


def main() -> int:
    pointcept_data_root = Path(
        os.environ.get(
            "POINTCEPT_DATA_ROOT",
            str(Path(__file__).resolve().parents[2] / "data"),
        )
    )
    parser = argparse.ArgumentParser(
        description="Rewrite ARKit full split JSONs to absolute asset paths."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            os.environ.get(
                "ARKIT_FULL_SOURCE_ROOT",
                str(pointcept_data_root / "arkitscenes"),
            )
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "ARKIT_FULL_META_ROOT",
                str(pointcept_data_root / "arkitscenes_absmeta"),
            )
        ),
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("Training", "Validation"):
        source_file = source_root / "splits" / f"{split_name}.json"
        payload = json.loads(source_file.read_text(encoding="utf-8"))
        rewritten = {}
        for key, record in payload.items():
            rewritten[key] = {
                "pointclouds": rewrite_path(source_root, record["pointclouds"]),
                "images": rewrite_path(source_root, record["images"]),
                "correspondences": rewrite_path(source_root, record["correspondences"]),
            }
        target_file = split_dir / f"{split_name}.json"
        target_file.write_text(json.dumps(rewritten), encoding="utf-8")
        print(f"[ok] wrote {target_file} ({len(rewritten)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
