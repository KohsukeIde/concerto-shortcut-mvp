#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rewrite_path(value: str, source_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    parts = path.parts
    if len(parts) >= 2 and parts[:2] == ("data", "scannet"):
        return str(source_root.joinpath(*parts[2:]).resolve())
    return str((source_root / path).resolve())


def rewrite_record(record: dict, source_root: Path) -> dict:
    output = dict(record)
    output["pointclouds"] = rewrite_path(record["pointclouds"], source_root)
    output["images"] = [rewrite_path(item, source_root) for item in record["images"]]
    output["correspondences"] = [
        rewrite_path(item, source_root) for item in record["correspondences"]
    ]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite Concerto ScanNet split JSON paths away from data/scannet."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_splits = source_root / "splits"
    output_splits = output_root / "splits"
    output_splits.mkdir(parents=True, exist_ok=True)

    if not source_splits.is_dir():
        raise FileNotFoundError(f"missing source splits directory: {source_splits}")

    for split_file in sorted(source_splits.glob("*.json")):
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        rewritten = {
            key: rewrite_record(record, source_root)
            for key, record in payload.items()
        }
        target = output_splits / split_file.name
        target.write_text(json.dumps(rewritten), encoding="utf-8")
        print(f"[ok] wrote {target} ({len(rewritten)} samples)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
