#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from main_variant_step05_utils import (
    build_loader,
    build_main_variant_model,
    iter_limited,
    move_batch_to_cuda,
    parse_csv,
    repo_root_from_here,
    save_json,
    seed_everything,
    select_dataset_specs,
)


MODES = [
    "global_target_permutation",
    "cross_image_target_swap",
    "cross_scene_target_swap",
]


def corrupt_target(model, target: torch.Tensor, batch_index: torch.Tensor, image_index: torch.Tensor, mode: str):
    if mode == "global_target_permutation":
        return model._apply_global_target_permutation(
            target, mode_name="target_distance_global_target_permutation"
        )
    if mode == "cross_image_target_swap":
        return model._apply_cross_image_target_swap(target, image_index)
    if mode == "cross_scene_target_swap":
        return model._apply_cross_scene_target_swap(target, batch_index)
    raise ValueError(f"unknown mode: {mode}")


def group_index_for_mode(batch_index: torch.Tensor, image_index: torch.Tensor, mode: str):
    if mode == "cross_image_target_swap":
        return image_index
    if mode == "cross_scene_target_swap":
        return batch_index
    return None


def init_stats():
    return {
        "batches": 0,
        "rows": 0,
        "sum_cos": 0.0,
        "sum_cos2": 0.0,
        "fallback_batches": 0,
        "sum_unique_groups": 0.0,
    }


def update_stats(stats: dict, cos: torch.Tensor, unique_groups: int | None) -> None:
    cos_cpu = cos.detach().float().cpu()
    rows = int(cos_cpu.numel())
    stats["batches"] += 1
    stats["rows"] += rows
    stats["sum_cos"] += float(cos_cpu.sum().item())
    stats["sum_cos2"] += float(cos_cpu.square().sum().item())
    if unique_groups is not None:
        stats["sum_unique_groups"] += float(unique_groups)
        if unique_groups <= 1:
            stats["fallback_batches"] += 1


def finalize_stats(dataset: str, mode: str, stats: dict) -> dict:
    rows = int(stats["rows"])
    batches = int(stats["batches"])
    mean = stats["sum_cos"] / rows if rows else float("nan")
    variance = stats["sum_cos2"] / rows - mean * mean if rows else float("nan")
    std = math.sqrt(max(variance, 0.0)) if rows else float("nan")
    mean_unique_groups = (
        stats["sum_unique_groups"] / batches
        if batches and mode != "global_target_permutation"
        else ""
    )
    return {
        "dataset": dataset,
        "mode": mode,
        "batches": batches,
        "rows": rows,
        "mean_cos_original_corrupted": mean,
        "std_cos_original_corrupted": std,
        "mean_one_minus_cos": 1.0 - mean if rows else float("nan"),
        "fallback_batches": int(stats["fallback_batches"]),
        "mean_unique_groups": mean_unique_groups,
    }


def format_float(value) -> str:
    if value == "":
        return ""
    return f"{float(value):.6f}"


def write_results(rows: list[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "mode",
        "batches",
        "rows",
        "mean_cos_original_corrupted",
        "std_cos_original_corrupted",
        "mean_one_minus_cos",
        "fallback_batches",
        "mean_unique_groups",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "mode": row["mode"],
                    "batches": row["batches"],
                    "rows": row["rows"],
                    "mean_cos_original_corrupted": format_float(
                        row["mean_cos_original_corrupted"]
                    ),
                    "std_cos_original_corrupted": format_float(
                        row["std_cos_original_corrupted"]
                    ),
                    "mean_one_minus_cos": format_float(row["mean_one_minus_cos"]),
                    "fallback_batches": row["fallback_batches"],
                    "mean_unique_groups": format_float(row["mean_unique_groups"]),
                }
            )

    lines = [
        "# Target Corruption Distance",
        "",
        "`mean cos` is `cos(t_original, t_corrupted)` after the same corruption helper used by the causal battery. Higher values mean the corrupted target stayed closer to the original target.",
        "",
        "| dataset | mode | batches | rows | mean cos | 1 - mean cos | std cos | fallback batches | mean unique groups |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mode']} | {row['batches']} | {row['rows']} | "
            f"{format_float(row['mean_cos_original_corrupted'])} | "
            f"{format_float(row['mean_one_minus_cos'])} | "
            f"{format_float(row['std_cos_original_corrupted'])} | "
            f"{row['fallback_batches']} | {format_float(row['mean_unique_groups'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure cos(t_original, t_corrupted) for target-swap modes."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="pretrain-concerto-v1m1-0-probe-enc2d-baseline")
    parser.add_argument("--weight", type=Path, default=None)
    parser.add_argument("--datasets", default="arkit,scannet")
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tag", default="main-origin-six-step05")
    parser.add_argument("--max-batches-per-dataset", type=int, default=32)
    parser.add_argument("--max-rows-per-batch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-worker", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    seed_everything(args.seed)
    if args.weight is None:
        args.weight = repo_root / "data" / "weights" / "concerto" / "concerto_base_origin.pth"
    if args.output_root is None:
        args.output_root = (
            repo_root
            / "data"
            / "runs"
            / "main_variant_target_corruption_distance"
            / args.tag
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    cfg, model, load_info = build_main_variant_model(repo_root, args.config, args.weight)
    model.eval()
    specs = select_dataset_specs(
        repo_root,
        parse_csv(args.datasets),
        allow_missing=args.allow_missing_datasets,
    )
    loaders = {
        spec.name: build_loader(
            cfg,
            spec,
            "val",
            batch_size=args.batch_size,
            num_worker=args.num_worker,
            shuffle=False,
        )
        for spec in specs
    }

    aggregate = defaultdict(init_stats)
    with torch.inference_mode():
        for dataset_name, loader in loaders.items():
            for batch_idx, batch in iter_limited(loader, args.max_batches_per_dataset):
                batch = move_batch_to_cuda(batch)
                extracted = model.extract_enc2d_coord_target(
                    batch, max_rows=args.max_rows_per_batch
                )
                target = extracted["target"].detach().float()
                if target.numel() == 0:
                    continue
                batch_index = extracted["batch_index"].detach().long()
                image_index = extracted["image_index"].detach().long()
                for mode in MODES:
                    group_index = group_index_for_mode(batch_index, image_index, mode)
                    unique_groups = (
                        int(torch.unique(group_index).numel())
                        if group_index is not None
                        else None
                    )
                    corrupted = corrupt_target(model, target, batch_index, image_index, mode)
                    cos = F.cosine_similarity(target, corrupted.float(), dim=1, eps=1e-6)
                    update_stats(aggregate[(dataset_name, mode)], cos, unique_groups)
                if (batch_idx + 1) % 8 == 0:
                    print(f"[distance] dataset={dataset_name} batches={batch_idx + 1}", flush=True)

    rows = [
        finalize_stats(dataset, mode, aggregate[(dataset, mode)])
        for dataset in [spec.name for spec in specs]
        for mode in MODES
    ]
    csv_path = args.output_root / "results_target_corruption_distance.csv"
    md_path = args.output_root / "results_target_corruption_distance.md"
    write_results(rows, csv_path, md_path)
    save_json(
        args.output_root / "metadata.json",
        {
            "kind": "target_corruption_distance",
            "config": args.config,
            "source_weight": str(args.weight.resolve()),
            "load_info": load_info,
            "datasets": [spec.name for spec in specs],
            "max_batches_per_dataset": args.max_batches_per_dataset,
            "max_rows_per_batch": args.max_rows_per_batch,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "csv": str(csv_path),
            "md": str(md_path),
        },
    )
    print(f"[distance] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
