#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_ptv3_v151_masking_compat import (
    build_variants,
    clone_scene,
    input_point_count,
    load_config,
    load_scene,
    masked_model_scene,
    make_variant_batch_with_mask,
    maybe_save_example_scene,
    parse_float_list,
    parse_int_list,
    repo_root_from_here,
    scene_paths,
    seed_everything,
    setup_official_imports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export transformed/voxelized masking examples without model inference. "
            "This is used to inspect what keep20/keep10/fixed4k/masked-model conditions "
            "actually look like on ScanNet/ScanNet200/S3DIS."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--official-root", type=Path, default=Path("data/tmp/Pointcept-v1.5.1"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--segment-key", default="segment20")
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--random-keep-ratios", default="0.2,0.1")
    parser.add_argument("--fixed-point-counts", default="4000")
    parser.add_argument("--masked-model-keep-ratios", default="0.2")
    parser.add_argument("--classwise-keep-ratios", default="")
    parser.add_argument("--structured-keep-ratios", default="")
    parser.add_argument("--feature-zero-ratios", default="")
    parser.add_argument("--structured-block-size", type=int, default=64)
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--scene-ids", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("data/runs/masking_examples"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("tools/concerto_projection_shortcut/results_masking_examples"))
    parser.add_argument("--example-max-export-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260421)
    return parser.parse_args()


def selected_scenes(all_scenes: list[Path], scene_ids: str, num_scenes: int) -> list[Path]:
    if scene_ids.strip():
        wanted = {x.strip() for x in scene_ids.split(",") if x.strip()}
        return [scene for scene in all_scenes if scene.name in wanted]
    return all_scenes[:num_scenes]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.repo_root = args.repo_root.resolve()
    args.official_root = (args.repo_root / args.official_root).resolve() if not args.official_root.is_absolute() else args.official_root
    args.data_root = (args.repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (args.repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (args.repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    args.save_example_scenes = args.num_scenes
    args.example_output_dir = args.output_dir

    Config, Compose, _ = setup_official_imports(args.official_root)
    cfg = load_config(Config, args.official_root / args.config)
    transform = Compose(cfg.data.val.transform)
    scenes = selected_scenes(scene_paths(args.data_root, args.split), args.scene_ids, args.num_scenes)
    variants = build_variants(args)

    rows: list[dict[str, object]] = []
    for scene_idx, scene_path in enumerate(scenes):
        raw_scene = load_scene(scene_path, args.segment_key)
        clean_batch = transform(clone_scene(raw_scene))
        for variant in variants:
            seed = args.seed + scene_idx * 1009 + abs(hash(variant.name)) % 1000
            if variant.kind == "masked_model_drop_raw":
                masked_scene, keep_frac = masked_model_scene(raw_scene, variant.keep_ratio, seed)
                masked_batch = transform(masked_scene)
            else:
                masked_batch, keep_frac, _ = make_variant_batch_with_mask(clean_batch, variant, seed)
            maybe_save_example_scene(args, scene_path.name, variant, clean_batch, masked_batch, keep_frac)
            rows.append(
                {
                    "dataset": args.dataset_tag,
                    "scene": scene_path.name,
                    "variant": variant.name,
                    "kind": variant.kind,
                    "keep_ratio": variant.keep_ratio,
                    "fixed_count": variant.fixed_count,
                    "base_points": int(input_point_count(clean_batch)),
                    "masked_points": int(input_point_count(masked_batch)),
                    "observed_keep_frac": float(keep_frac),
                }
            )

    csv_path = args.summary_prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Masking Example Export",
        "",
        f"- Dataset: `{args.dataset_tag}`",
        f"- Config: `{args.config}`",
        f"- Data root: `{args.data_root}`",
        f"- Split: `{args.split}`",
        f"- Segment key: `{args.segment_key}`",
        f"- Scenes: `{', '.join(scene.name for scene in scenes)}`",
        "",
        "## Variants",
        "",
    ]
    for variant in variants:
        md_lines.append(
            f"- `{variant.name}`: kind=`{variant.kind}`, keep_ratio={variant.keep_ratio}, fixed_count={variant.fixed_count}"
        )
    md_lines += [
        "",
        "## Summary",
        "",
        "| scene | variant | base points | masked points | observed keep |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| `{row['scene']}` | `{row['variant']}` | {row['base_points']} | {row['masked_points']} | {float(row['observed_keep_frac']):.4f} |"
        )
    md_lines += [
        "",
        "## Files",
        "",
        f"- Summary CSV: `{csv_path}`",
        f"- Example root: `{args.output_dir / args.dataset_tag}`",
    ]
    md_path = args.summary_prefix.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n")
    print(json.dumps({"summary_csv": str(csv_path), "example_root": str(args.output_dir / args.dataset_tag)}, indent=2))


if __name__ == "__main__":
    main()
