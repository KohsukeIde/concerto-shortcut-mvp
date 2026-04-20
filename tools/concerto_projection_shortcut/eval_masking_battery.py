#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    build_model,
    load_config,
    move_to_cuda,
    summarize_confusion,
    update_confusion,
    weak_mean,
)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    keep_ratio: float = 1.0
    feature_zero_ratio: float = 0.0
    block_size: int = 64


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shortcut-sensitive masking battery for ScanNet segmentation checkpoints. "
            "The first use is a cheap Concerto decoder smoke: clean vs random point "
            "drop at high mask ratios, evaluated in the same voxel-level space."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--method-name", default="concerto_decoder_origin")
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--random-keep-ratios", default="0.2")
    parser.add_argument("--structured-keep-ratios", default="")
    parser.add_argument("--feature-zero-ratios", default="")
    parser.add_argument("--structured-block-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_masking_battery"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_names(text: str) -> list[int]:
    ids = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {name}")
        ids.append(NAME_TO_ID[name])
    return ids


def split_dataset_cfg(cfg, split: str, data_root: Path):
    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg.split = split
    ds_cfg.data_root = str(data_root)
    ds_cfg.test_mode = False
    return ds_cfg


def build_loader(cfg, split: str, data_root: Path, batch_size: int, num_worker: int):
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn

    dataset = build_dataset(split_dataset_cfg(cfg, split, data_root))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        collate_fn=lambda batch: point_collate_fn(batch, mix_prob=0),
    )


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants = [Variant("clean_voxel", "clean")]
    for keep in parse_float_list(args.random_keep_ratios):
        variants.append(Variant(f"random_keep{str(keep).replace('.', 'p')}", "random_drop", keep_ratio=keep))
    for keep in parse_float_list(args.structured_keep_ratios):
        variants.append(
            Variant(
                f"structured_b{args.structured_block_size}_keep{str(keep).replace('.', 'p')}",
                "structured_drop",
                keep_ratio=keep,
                block_size=args.structured_block_size,
            )
        )
    for ratio in parse_float_list(args.feature_zero_ratios):
        variants.append(Variant(f"feature_zero{str(ratio).replace('.', 'p')}", "feature_zero", feature_zero_ratio=ratio))
    return variants


def input_point_count(batch: dict) -> int:
    return int(batch["segment"].shape[0])


def clone_batch(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
    return out


def filter_batch(batch: dict, mask: torch.Tensor) -> dict:
    n = input_point_count(batch)
    out = {}
    for key, value in batch.items():
        if key in {"inverse", "origin_segment"}:
            continue
        if isinstance(value, torch.Tensor) and value.shape[:1] == (n,):
            out[key] = value[mask]
        elif key == "offset" and isinstance(value, torch.Tensor):
            out[key] = torch.tensor([int(mask.sum().item())], dtype=value.dtype, device=value.device)
        else:
            out[key] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    return out


def random_drop_mask(n: int, keep_ratio: float, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    keep = torch.rand(n, device=device, generator=generator) < keep_ratio
    if not bool(keep.any()):
        keep[torch.randint(n, (1,), device=device, generator=generator)] = True
    return keep


def structured_drop_mask(grid: torch.Tensor, keep_ratio: float, block_size: int, generator: torch.Generator) -> torch.Tensor:
    keys = torch.div(grid.long(), block_size, rounding_mode="floor")
    _, inv = torch.unique(keys, dim=0, return_inverse=True)
    n_region = int(inv.max().item()) + 1
    region_keep = torch.rand(n_region, device=grid.device, generator=generator) < keep_ratio
    if not bool(region_keep.any()):
        region_keep[torch.randint(n_region, (1,), device=grid.device, generator=generator)] = True
    return region_keep[inv]


def make_variant_batch(batch: dict, variant: Variant, seed: int) -> tuple[dict, float]:
    n = input_point_count(batch)
    device = batch["segment"].device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if variant.kind == "clean":
        return clone_batch(batch), 1.0
    if variant.kind == "random_drop":
        mask = random_drop_mask(n, variant.keep_ratio, device, generator)
        return filter_batch(batch, mask), float(mask.float().mean().item())
    if variant.kind == "structured_drop":
        mask = structured_drop_mask(batch["grid_coord"], variant.keep_ratio, variant.block_size, generator)
        return filter_batch(batch, mask), float(mask.float().mean().item())
    if variant.kind == "feature_zero":
        out = clone_batch(batch)
        zero = random_drop_mask(n, variant.feature_zero_ratio, device, generator)
        if "feat" in out:
            out["feat"][zero] = 0
        return out, 1.0
    raise ValueError(f"unknown variant kind: {variant.kind}")


def inference_batch(batch: dict) -> dict:
    """Drop supervision-only tensors so eval forward does not compute losses."""
    skip = {"segment", "origin_segment", "inverse"}
    out = {}
    for key, value in batch.items():
        if key in skip:
            continue
        out[key] = value
    return out


@torch.no_grad()
def forward_logits_labels(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["segment"].long()
    model_input = inference_batch(batch)
    try:
        out = model(model_input, return_point=True)
    except TypeError as exc:
        # PPT-v1m1 does not accept return_point; masking battery only needs logits.
        if "return_point" not in str(exc):
            raise
        out = model(model_input)
    logits = out["seg_logits"].float()
    if logits.shape[0] != labels.shape[0]:
        raise RuntimeError(f"shape mismatch logits={logits.shape} labels={labels.shape}")
    return logits, labels


def picture_to_wall_from_conf(conf: np.ndarray) -> float:
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    denom = conf[pic].sum()
    return float(conf[pic, wall] / denom) if denom else float("nan")


def summary_row(method: str, variant: Variant, repeat_count: int, mean_keep: float, conf: np.ndarray, base_summary: dict, weak_classes: list[int]) -> dict[str, float | str | int]:
    s = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    floor = NAME_TO_ID["floor"]
    return {
        "method": method,
        "variant": variant.name,
        "kind": variant.kind,
        "keep_ratio": variant.keep_ratio,
        "feature_zero_ratio": variant.feature_zero_ratio,
        "block_size": variant.block_size,
        "repeats": repeat_count,
        "observed_keep_frac": mean_keep,
        "mIoU": s["mIoU"],
        "delta_mIoU": s["mIoU"] - base_summary["mIoU"],
        "mAcc": s["mAcc"],
        "allAcc": s["allAcc"],
        "weak_mean_iou": weak_mean(s, weak_classes),
        "delta_weak_mean_iou": weak_mean(s, weak_classes) - weak_mean(base_summary, weak_classes),
        "picture_iou": float(s["iou"][pic]),
        "delta_picture_iou": float(s["iou"][pic] - base_summary["iou"][pic]),
        "wall_iou": float(s["iou"][wall]),
        "floor_iou": float(s["iou"][floor]),
        "picture_to_wall": picture_to_wall_from_conf(conf),
        "delta_picture_to_wall": picture_to_wall_from_conf(conf) - picture_to_wall_from_conf(base_summary["conf"]),
    }


def eval_masking(args: argparse.Namespace, model, cfg, variants: list[Variant], weak_classes: list[int], num_classes: int):
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    confs = {v.name: torch.zeros((num_classes, num_classes), dtype=torch.long) for v in variants}
    keep_sums = {v.name: 0.0 for v in variants}
    keep_counts = {v.name: 0 for v in variants}

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            for variant in variants:
                repeat_count = args.repeats if variant.kind in {"random_drop", "structured_drop", "feature_zero"} else 1
                for repeat_idx in range(repeat_count):
                    seed = args.seed + batch_idx * 1009 + repeat_idx * 9176 + abs(hash(variant.name)) % 1000
                    masked_batch, keep_frac = make_variant_batch(batch, variant, seed)
                    if input_point_count(masked_batch) <= 0:
                        continue
                    logits, labels = forward_logits_labels(model, masked_batch)
                    pred = logits.argmax(dim=1)
                    update_confusion(confs[variant.name], pred.cpu(), labels.cpu(), num_classes, -1)
                    keep_sums[variant.name] += keep_frac
                    keep_counts[variant.name] += 1
            if (batch_idx + 1) % 25 == 0:
                clean = summarize_confusion(confs["clean_voxel"].numpy(), SCANNET20_CLASS_NAMES)
                print(f"[val] batch={batch_idx+1} clean_voxel_mIoU={clean['mIoU']:.4f}", flush=True)
    return {
        "conf": {k: v.numpy() for k, v in confs.items()},
        "keep_sums": keep_sums,
        "keep_counts": keep_counts,
    }


def write_results(args: argparse.Namespace, results: dict, variants: list[Variant], weak_classes: list[int]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = summarize_confusion(results["conf"]["clean_voxel"], SCANNET20_CLASS_NAMES)
    base["conf"] = results["conf"]["clean_voxel"]
    rows = []
    for v in variants:
        rows.append(
            summary_row(
                args.method_name,
                v,
                results["keep_counts"][v.name],
                results["keep_sums"][v.name] / max(results["keep_counts"][v.name], 1),
                results["conf"][v.name],
                base,
                weak_classes,
            )
        )
    csv_path = args.output_dir / "masking_battery_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    prefix = args.summary_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sorted_rows = sorted(rows, key=lambda r: (r["variant"] != "clean_voxel", -float(r["mIoU"])))
    lines = []
    lines.append("# Masking Battery Pilot\n")
    lines.append("Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.\n")
    lines.append("## Setup\n")
    lines.append(f"- Method: `{args.method_name}`")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Weight: `{args.weight}`")
    lines.append(f"- Random keep ratios: `{args.random_keep_ratios}`")
    lines.append(f"- Structured keep ratios: `{args.structured_keep_ratios}`")
    lines.append(f"- Feature-zero ratios: `{args.feature_zero_ratios}`")
    lines.append(f"- Repeats: `{args.repeats}`")
    lines.append("")
    lines.append("## Results\n")
    lines.append("| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted_rows:
        lines.append(
            f"| `{r['variant']}` | {float(r['observed_keep_frac']):.4f} | "
            f"{float(r['mIoU']):.4f} | {float(r['delta_mIoU']):+.4f} | {float(r['allAcc']):.4f} | "
            f"{float(r['weak_mean_iou']):.4f} | {float(r['picture_iou']):.4f} | {float(r['delta_picture_iou']):+.4f} | "
            f"{float(r['wall_iou']):.4f} | {float(r['floor_iou']):.4f} | {float(r['picture_to_wall']):.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation Gate\n")
    lines.append("- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.")
    lines.append("- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.")
    lines.append("- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.")
    lines.append("")
    lines.append("## Files\n")
    lines.append(f"- Summary CSV: `{csv_path.resolve()}`")
    md = prefix.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    (args.output_dir / "masking_battery.md").write_text("\n".join(lines) + "\n")
    metadata = {
        "method_name": args.method_name,
        "config": str(args.config),
        "weight": str(args.weight),
        "variants": [v.__dict__ for v in variants],
        "outputs": {"summary_csv": str(csv_path), "summary_md": str(md)},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[done] wrote {md}", flush=True)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.repo_root = args.repo_root.resolve()
    args.data_root = (args.repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.weight = (args.repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.output_dir = (args.repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (args.repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    variants = build_variants(args)
    weak_classes = parse_names(args.weak_classes)
    if args.dry_run:
        print(f"[dry-run] data_root={args.data_root}")
        print(f"[dry-run] weight={args.weight}")
        print(f"[dry-run] variants={[v.name for v in variants]}")
        print(f"[dry-run] weak_classes={[SCANNET20_CLASS_NAMES[c] for c in weak_classes]}")
        return
    cfg = load_config(args.repo_root / args.config)
    num_classes = int(cfg.data.num_classes)
    model = build_model(cfg, args.weight)
    results = eval_masking(args, model, cfg, variants, weak_classes, num_classes)
    write_results(args, results, variants, weak_classes)


if __name__ == "__main__":
    main()
