#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import defaultdict
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
    eval_tensors,
    forward_features,
    load_config,
    move_to_cuda,
    summarize_confusion,
    update_confusion,
)


@dataclass
class PairStats:
    target_count: int = 0
    base_correct: int = 0
    base_counter: int = 0
    region_correct: int = 0
    region_counter: int = 0
    point_top2: int = 0
    point_top5: int = 0
    region_top2: int = 0
    region_top5: int = 0
    majority_target: int = 0
    majority_counter: int = 0
    purity_sum: float = 0.0
    hard_region_points: int = 0
    hard_region_count: int = 0
    target_region_count: int = 0


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Region/superpoint-style diagnostic for ScanNet decoder-probe outputs. "
            "This checks whether weak-class failures are coherent at a local region "
            "granularity and whether region-averaged logits change oracle headroom."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--heldout-mod", type=int, default=5)
    parser.add_argument("--heldout-remainder", type=int, default=0)
    parser.add_argument("--region-voxel-sizes", default="4,8,16,32")
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall",
    )
    parser.add_argument("--min-target-points-per-region", type=int, default=64)
    parser.add_argument("--hard-counter-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_region_superpoint_analysis"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_pairs(text: str) -> list[tuple[int, int]]:
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        a, b = raw.split(":", 1)
        a = a.strip()
        b = b.strip()
        if a not in NAME_TO_ID or b not in NAME_TO_ID:
            raise ValueError(f"unknown class pair: {raw}")
        out.append((NAME_TO_ID[a], NAME_TO_ID[b]))
    return out


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


def expanded_grid_coord(batch: dict) -> torch.Tensor | None:
    grid = batch.get("grid_coord")
    if grid is None:
        return None
    inv = batch.get("inverse")
    if inv is not None:
        return grid.long()[inv.long()]
    return grid.long()


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def metric_from_summary(summary: dict, class_id: int) -> float:
    return float(summary["iou"][class_id])


def picture_to_wall_from_conf(conf: np.ndarray) -> float:
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    denom = conf[pic].sum()
    return float(conf[pic, wall] / denom) if denom else float("nan")


def make_oracle_pred(logits: torch.Tensor, target: torch.Tensor, base_pred: torch.Tensor, k: int) -> torch.Tensor:
    topk = logits.topk(k, dim=1).indices
    hit = (topk == target[:, None]).any(dim=1)
    return torch.where(hit, target, base_pred)


def update_pair_stats(
    stats: dict[tuple[str, int, int, int], PairStats],
    domain: str,
    region_size: int,
    target: torch.Tensor,
    base_pred: torch.Tensor,
    region_pred: torch.Tensor,
    point_top2: torch.Tensor,
    point_top5: torch.Tensor,
    region_top2: torch.Tensor,
    region_top5: torch.Tensor,
    region_majority: torch.Tensor,
    region_purity_for_class: dict[int, torch.Tensor],
    region_inv: torch.Tensor,
    pairs: list[tuple[int, int]],
    min_target_points: int,
    hard_counter_frac: float,
) -> None:
    for cls, counter in pairs:
        mask = target == cls
        n = int(mask.sum().item())
        if n == 0:
            continue
        key = (domain, region_size, cls, counter)
        rec = stats[key]
        rec.target_count += n
        rec.base_correct += int((base_pred[mask] == cls).sum().item())
        rec.base_counter += int((base_pred[mask] == counter).sum().item())
        rec.region_correct += int((region_pred[mask] == cls).sum().item())
        rec.region_counter += int((region_pred[mask] == counter).sum().item())
        rec.point_top2 += int(point_top2[mask].sum().item())
        rec.point_top5 += int(point_top5[mask].sum().item())
        rec.region_top2 += int(region_top2[mask].sum().item())
        rec.region_top5 += int(region_top5[mask].sum().item())
        rec.majority_target += int((region_majority[mask] == cls).sum().item())
        rec.majority_counter += int((region_majority[mask] == counter).sum().item())
        rec.purity_sum += float(region_purity_for_class[cls][mask].sum().item())

        inv_cls = region_inv[mask]
        wall_mask = mask & (base_pred == counter)
        target_by_region = torch.bincount(inv_cls, minlength=int(region_inv.max().item()) + 1)
        counter_by_region = torch.bincount(region_inv[wall_mask], minlength=target_by_region.numel())
        has_target = target_by_region >= min_target_points
        if has_target.any():
            frac = counter_by_region.float() / target_by_region.clamp_min(1).float()
            hard = has_target & (frac >= hard_counter_frac)
            rec.target_region_count += int(has_target.sum().item())
            rec.hard_region_count += int(hard.sum().item())
            rec.hard_region_points += int(target_by_region[hard].sum().item())


def eval_split(
    args: argparse.Namespace,
    model,
    cfg,
    split: str,
    domain: str,
    max_batches: int,
    region_sizes: list[int],
    pairs: list[tuple[int, int]],
    num_classes: int,
):
    loader = build_loader(cfg, split, args.data_root, args.batch_size, args.num_worker)
    base_conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    point_oracle_conf = {
        2: torch.zeros((num_classes, num_classes), dtype=torch.long),
        5: torch.zeros((num_classes, num_classes), dtype=torch.long),
    }
    region_conf = {s: torch.zeros((num_classes, num_classes), dtype=torch.long) for s in region_sizes}
    region_majority_conf = {s: torch.zeros((num_classes, num_classes), dtype=torch.long) for s in region_sizes}
    region_oracle_conf = {
        (s, k): torch.zeros((num_classes, num_classes), dtype=torch.long)
        for s in region_sizes
        for k in (2, 5)
    }
    pair_stats: dict[tuple[str, int, int, int], PairStats] = defaultdict(PairStats)
    scene_rows = []
    seen = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches >= 0 and batch_idx >= max_batches:
                break
            batch = move_to_cuda(batch)
            feat, logits_voxel, labels_voxel, batch = forward_features(model, batch)
            _, logits, target = eval_tensors(feat, logits_voxel, labels_voxel, batch)
            grid = expanded_grid_coord(batch)
            if grid is None:
                raise RuntimeError("grid_coord is required for region analysis")
            if grid.device != logits.device:
                grid = grid.to(logits.device)
            target = target.to(logits.device)
            valid = (target >= 0) & (target < num_classes)
            logits = logits[valid]
            target = target[valid]
            grid = grid[valid]
            base_pred = logits.argmax(dim=1)
            update_confusion(base_conf, base_pred.cpu(), target.cpu(), num_classes, -1)

            for k in (2, 5):
                pred = make_oracle_pred(logits, target, base_pred, k)
                update_confusion(point_oracle_conf[k], pred.cpu(), target.cpu(), num_classes, -1)
            point_top2 = (logits.topk(2, dim=1).indices == target[:, None]).any(dim=1)
            point_top5 = (logits.topk(5, dim=1).indices == target[:, None]).any(dim=1)

            for s in region_sizes:
                keys = torch.div(grid, s, rounding_mode="floor")
                _, inv = torch.unique(keys, dim=0, return_inverse=True)
                region_count = torch.bincount(inv).float().clamp_min(1.0)
                r = int(region_count.numel())
                logit_sum = torch.zeros((r, num_classes), device=logits.device, dtype=logits.dtype)
                logit_sum.index_add_(0, inv, logits)
                region_logits = logit_sum / region_count[:, None]
                point_region_logits = region_logits[inv]
                region_pred = point_region_logits.argmax(dim=1)
                update_confusion(region_conf[s], region_pred.cpu(), target.cpu(), num_classes, -1)

                onehot = torch.nn.functional.one_hot(target.long(), num_classes=num_classes).float()
                gt_count = torch.zeros((r, num_classes), device=logits.device, dtype=torch.float32)
                gt_count.index_add_(0, inv, onehot)
                region_majority = gt_count.argmax(dim=1)[inv]
                update_confusion(region_majority_conf[s], region_majority.cpu(), target.cpu(), num_classes, -1)
                region_purity_for_class = {
                    cls: (gt_count[:, cls] / region_count).clamp(0, 1)[inv]
                    for cls, _ in pairs
                }

                region_top2 = (point_region_logits.topk(2, dim=1).indices == target[:, None]).any(dim=1)
                region_top5 = (point_region_logits.topk(5, dim=1).indices == target[:, None]).any(dim=1)
                for k in (2, 5):
                    pred = make_oracle_pred(point_region_logits, target, base_pred, k)
                    update_confusion(region_oracle_conf[(s, k)], pred.cpu(), target.cpu(), num_classes, -1)

                update_pair_stats(
                    pair_stats,
                    domain,
                    s,
                    target,
                    base_pred,
                    region_pred,
                    point_top2,
                    point_top5,
                    region_top2,
                    region_top5,
                    region_majority,
                    region_purity_for_class,
                    inv,
                    pairs,
                    args.min_target_points_per_region,
                    args.hard_counter_frac,
                )

            seen += 1
            if (batch_idx + 1) % 25 == 0:
                base = summarize_confusion(base_conf.numpy(), SCANNET20_CLASS_NAMES)
                print(f"[{domain}] batch={batch_idx+1} base_mIoU={base['mIoU']:.4f}", flush=True)

    return {
        "domain": domain,
        "seen_batches": seen,
        "base_conf": base_conf.numpy(),
        "point_oracle_conf": {k: v.numpy() for k, v in point_oracle_conf.items()},
        "region_conf": {k: v.numpy() for k, v in region_conf.items()},
        "region_majority_conf": {k: v.numpy() for k, v in region_majority_conf.items()},
        "region_oracle_conf": {k: v.numpy() for k, v in region_oracle_conf.items()},
        "pair_stats": pair_stats,
        "scene_rows": scene_rows,
    }


def summary_row(name: str, conf: np.ndarray, base_summary: dict) -> dict[str, float | str]:
    s = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
    pic = NAME_TO_ID["picture"]
    return {
        "variant": name,
        "mIoU": s["mIoU"],
        "delta_mIoU": s["mIoU"] - base_summary["mIoU"],
        "picture_iou": metric_from_summary(s, pic),
        "delta_picture_iou": metric_from_summary(s, pic) - metric_from_summary(base_summary, pic),
        "picture_to_wall": picture_to_wall_from_conf(conf),
        "delta_picture_to_wall": picture_to_wall_from_conf(conf) - picture_to_wall_from_conf(base_summary["conf"]),
    }


def write_results(args: argparse.Namespace, split_results: list[dict], region_sizes: list[int], pairs: list[tuple[int, int]], num_classes: int) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    pair_rows = []
    for res in split_results:
        domain = res["domain"]
        base = summarize_confusion(res["base_conf"], SCANNET20_CLASS_NAMES)
        base["conf"] = res["base_conf"]
        summary_rows.append({"domain": domain, **summary_row("base", res["base_conf"], base)})
        for k, conf in res["point_oracle_conf"].items():
            summary_rows.append({"domain": domain, **summary_row(f"point_oracle_top{k}", conf, base)})
        for s in region_sizes:
            summary_rows.append({"domain": domain, **summary_row(f"region_logits_s{s}", res["region_conf"][s], base)})
            summary_rows.append({"domain": domain, **summary_row(f"region_majority_oracle_s{s}", res["region_majority_conf"][s], base)})
            for k in (2, 5):
                summary_rows.append({"domain": domain, **summary_row(f"region_oracle_s{s}_top{k}", res["region_oracle_conf"][(s, k)], base)})

        for (d, s, cls, counter), st in sorted(res["pair_stats"].items()):
            if d != domain:
                continue
            pair_rows.append(
                {
                    "domain": domain,
                    "region_size": s,
                    "class_id": cls,
                    "class_name": SCANNET20_CLASS_NAMES[cls],
                    "counterpart_id": counter,
                    "counterpart_name": SCANNET20_CLASS_NAMES[counter],
                    "target_count": st.target_count,
                    "base_correct": safe_ratio(st.base_correct, st.target_count),
                    "base_to_counterpart": safe_ratio(st.base_counter, st.target_count),
                    "region_correct": safe_ratio(st.region_correct, st.target_count),
                    "region_to_counterpart": safe_ratio(st.region_counter, st.target_count),
                    "point_top2_hit": safe_ratio(st.point_top2, st.target_count),
                    "point_top5_hit": safe_ratio(st.point_top5, st.target_count),
                    "region_top2_hit": safe_ratio(st.region_top2, st.target_count),
                    "region_top5_hit": safe_ratio(st.region_top5, st.target_count),
                    "region_majority_target": safe_ratio(st.majority_target, st.target_count),
                    "region_majority_counterpart": safe_ratio(st.majority_counter, st.target_count),
                    "mean_target_region_purity": safe_ratio(st.purity_sum, st.target_count),
                    "hard_counter_region_point_frac": safe_ratio(st.hard_region_points, st.target_count),
                    "hard_counter_region_frac": safe_ratio(st.hard_region_count, st.target_region_count),
                    "target_region_count": st.target_region_count,
                }
            )

    summary_csv = args.output_dir / "region_superpoint_summary.csv"
    pair_csv = args.output_dir / "region_superpoint_pair_metrics.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with pair_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    prefix = args.summary_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with (prefix.with_suffix(".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    md = prefix.with_suffix(".md")
    val_rows = [r for r in summary_rows if r["domain"] == "val"]
    val_rows_sorted = sorted(val_rows, key=lambda r: (str(r["variant"]).startswith("base"), -float(r["mIoU"])))
    pair_val = [r for r in pair_rows if r["domain"] == "val" and r["class_name"] == "picture"]
    pair_val = sorted(pair_val, key=lambda r: int(r["region_size"]))
    lines = []
    lines.append("# Region / Superpoint Diagnostic\n")
    lines.append("Quick diagnostic for whether weak-class failures are coherent at a local region granularity.\n")
    lines.append("## Setup\n")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Weight: `{args.weight}`")
    lines.append(f"- Region voxel sizes: `{','.join(str(x) for x in region_sizes)}`")
    lines.append(f"- Class pairs: `{args.class_pairs}`")
    lines.append("")
    lines.append("## Top Val Variants / Oracles\n")
    lines.append("| rank | variant | mIoU | ΔmIoU | picture | Δpicture | picture->wall | Δp->wall |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(val_rows_sorted[:18], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {float(r['mIoU']):.4f} | {float(r['delta_mIoU']):+.4f} | "
            f"{float(r['picture_iou']):.4f} | {float(r['delta_picture_iou']):+.4f} | "
            f"{float(r['picture_to_wall']):.4f} | {float(r['delta_picture_to_wall']):+.4f} |"
        )
    lines.append("")
    lines.append("## Picture Region Metrics On Val\n")
    lines.append("| region size | target pts | base correct | base->wall | region correct | region->wall | point top2 | region top2 | point top5 | region top5 | picture-majority | wall-majority | mean purity | hard-wall point frac |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in pair_val:
        lines.append(
            f"| {int(r['region_size'])} | {int(r['target_count'])} | "
            f"{float(r['base_correct']):.4f} | {float(r['base_to_counterpart']):.4f} | "
            f"{float(r['region_correct']):.4f} | {float(r['region_to_counterpart']):.4f} | "
            f"{float(r['point_top2_hit']):.4f} | {float(r['region_top2_hit']):.4f} | "
            f"{float(r['point_top5_hit']):.4f} | {float(r['region_top5_hit']):.4f} | "
            f"{float(r['region_majority_target']):.4f} | {float(r['region_majority_counterpart']):.4f} | "
            f"{float(r['mean_target_region_purity']):.4f} | {float(r['hard_counter_region_point_frac']):.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation Gate\n")
    lines.append("- If region logits/oracles improve `picture` top-K substantially over point logits, region-level readout has direct headroom.")
    lines.append("- If `picture` points mostly sit in wall-majority / low-purity regions, point-wise correction is not the only bottleneck; region granularity may be too coarse or object masks are needed.")
    lines.append("- If region-level predictions worsen while oracle top-K remains high, region smoothing alone is not the method; region structure is diagnostic only.")
    lines.append("")
    lines.append("## Files\n")
    lines.append(f"- Summary CSV: `{summary_csv.resolve()}`")
    lines.append(f"- Pair metrics CSV: `{pair_csv.resolve()}`")
    (args.output_dir / "region_superpoint_analysis.md").write_text("\n".join(lines) + "\n")
    md.write_text("\n".join(lines) + "\n")
    metadata = {
        "config": str(args.config),
        "weight": str(args.weight),
        "data_root": str(args.data_root),
        "region_sizes": region_sizes,
        "pairs": [(SCANNET20_CLASS_NAMES[a], SCANNET20_CLASS_NAMES[b]) for a, b in pairs],
        "outputs": {
            "summary_csv": str(summary_csv),
            "pair_csv": str(pair_csv),
            "summary_md": str(md),
        },
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
    region_sizes = parse_int_list(args.region_voxel_sizes)
    pairs = parse_pairs(args.class_pairs)
    if args.dry_run:
        print(f"[dry-run] data_root={args.data_root}")
        print(f"[dry-run] region_sizes={region_sizes}")
        print(f"[dry-run] pairs={[(SCANNET20_CLASS_NAMES[a], SCANNET20_CLASS_NAMES[b]) for a,b in pairs]}")
        return
    cfg = load_config(args.repo_root / args.config)
    num_classes = int(cfg.data.num_classes)
    model = build_model(cfg, args.weight)
    split_results = []
    if args.max_train_batches and args.max_train_batches > 0:
        split_results.append(
            eval_split(args, model, cfg, args.train_split, "train", args.max_train_batches, region_sizes, pairs, num_classes)
        )
    split_results.append(
        eval_split(args, model, cfg, args.val_split, "val", args.max_val_batches, region_sizes, pairs, num_classes)
    )
    write_results(args, split_results, region_sizes, pairs, num_classes)


if __name__ == "__main__":
    main()
