#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_masking_battery import (  # noqa: E402
    Variant,
    build_variants,
    make_variant_batch,
    picture_to_wall_from_conf,
)
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    summarize_confusion,
    update_confusion,
    weak_mean,
)


class CoordMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Majority and coordinate-only baselines for the ScanNet masking battery."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py")
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="coord_baselines")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=256)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-train-points", type=int, default=800000)
    parser.add_argument("--max-per-class", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--coord-batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-balanced", action="store_true")
    parser.add_argument("--random-keep-ratios", default="0.5,0.3,0.2,0.1")
    parser.add_argument("--structured-keep-ratios", default="0.5,0.2")
    parser.add_argument("--feature-zero-ratios", default="1.0")
    parser.add_argument("--structured-block-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_masking_coord_baselines"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_config(path: Path):
    from pointcept.utils.config import Config

    return Config.fromfile(str(path))


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


def coord_features(batch: dict) -> torch.Tensor:
    coord = batch["coord"].float()
    grid = batch["grid_coord"].float()
    c_min = coord.min(dim=0).values
    c_max = coord.max(dim=0).values
    c_span = (c_max - c_min).clamp_min(1e-4)
    coord01 = (coord - c_min) / c_span
    coord_centered = coord01 - 0.5
    g_min = grid.min(dim=0).values
    g_max = grid.max(dim=0).values
    g_span = (g_max - g_min).clamp_min(1.0)
    grid01 = (grid - g_min) / g_span
    feats = [
        coord01,
        coord_centered,
        coord_centered * coord_centered,
        grid01,
        grid01 - 0.5,
        torch.sin(2.0 * np.pi * coord01),
        torch.cos(2.0 * np.pi * coord01),
    ]
    return torch.cat(feats, dim=1)


def collect_train(args: argparse.Namespace, cfg, num_classes: int):
    loader = build_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker)
    xs_by_class: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_classes)}
    ys_by_class: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_classes)}
    counts = torch.zeros(num_classes, dtype=torch.long)
    seen = 0
    for batch_idx, batch in enumerate(loader):
        if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
            break
        x = coord_features(batch)
        y = batch["segment"].long()
        valid = (y >= 0) & (y < num_classes)
        x = x[valid]
        y = y[valid]
        counts += torch.bincount(y, minlength=num_classes)
        for cls in range(num_classes):
            current = sum(t.shape[0] for t in xs_by_class[cls])
            remain = args.max_per_class - current
            if remain <= 0:
                continue
            idx = torch.nonzero(y == cls, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            if idx.numel() > remain:
                idx = idx[torch.randperm(idx.numel())[:remain]]
            xs_by_class[cls].append(x[idx])
            ys_by_class[cls].append(y[idx])
        seen += 1
        if (batch_idx + 1) % 50 == 0:
            total = sum(sum(t.shape[0] for t in xs_by_class[c]) for c in range(num_classes))
            print(f"[train-bank] batch={batch_idx+1} stored={total}", flush=True)
        total = sum(sum(t.shape[0] for t in xs_by_class[c]) for c in range(num_classes))
        if total >= args.max_train_points:
            break
    xs = []
    ys = []
    for cls in range(num_classes):
        xs.extend(xs_by_class[cls])
        ys.extend(ys_by_class[cls])
    x_all = torch.cat(xs, dim=0).float()
    y_all = torch.cat(ys, dim=0).long()
    if x_all.shape[0] > args.max_train_points:
        idx = torch.randperm(x_all.shape[0])[: args.max_train_points]
        x_all = x_all[idx]
        y_all = y_all[idx]
    perm = torch.randperm(x_all.shape[0])
    return x_all[perm], y_all[perm], counts, seen


def train_coord_mlp(args: argparse.Namespace, x: torch.Tensor, y: torch.Tensor, num_classes: int) -> CoordMLP:
    device = torch.device("cuda")
    model = CoordMLP(x.shape[1], num_classes).to(device)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.coord_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    weights = None
    if args.class_balanced:
        counts = torch.bincount(y, minlength=num_classes).float()
        weights = (counts.sum() / counts.clamp_min(1.0)).sqrt()
        weights = (weights / weights.mean()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += int(yb.numel())
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            loss_sum += float(loss.item()) * int(yb.numel())
        print(f"[coord-train] epoch={epoch+1}/{args.epochs} loss={loss_sum/max(total,1):.4f} acc={correct/max(total,1):.4f}", flush=True)
    return model.eval()


def eval_baselines(args: argparse.Namespace, cfg, variants: list[Variant], model: CoordMLP, majority_class: int, weak_classes: list[int], num_classes: int):
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    methods = ["majority", "coord_mlp"]
    confs = {(method, v.name): torch.zeros((num_classes, num_classes), dtype=torch.long) for method in methods for v in variants}
    keep_sums = {v.name: 0.0 for v in variants}
    keep_counts = {v.name: 0 for v in variants}
    device = torch.device("cuda")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            for variant in variants:
                repeat_count = args.repeats if variant.kind in {"random_drop", "structured_drop", "feature_zero"} else 1
                for repeat_idx in range(repeat_count):
                    seed = args.seed + batch_idx * 1009 + repeat_idx * 9176 + abs(hash(variant.name)) % 1000
                    masked_batch, keep_frac = make_variant_batch(batch, variant, seed)
                    target = masked_batch["segment"].long()
                    valid = (target >= 0) & (target < num_classes)
                    target = target[valid]
                    majority_pred = torch.full_like(target, majority_class)
                    update_confusion(confs[("majority", variant.name)], majority_pred, target, num_classes, -1)
                    x = coord_features(masked_batch)[valid].to(device)
                    pred = model(x).argmax(dim=1).cpu()
                    update_confusion(confs[("coord_mlp", variant.name)], pred, target, num_classes, -1)
                    keep_sums[variant.name] += keep_frac
                    keep_counts[variant.name] += 1
            if (batch_idx + 1) % 25 == 0:
                clean = summarize_confusion(confs[("coord_mlp", "clean_voxel")].numpy(), SCANNET20_CLASS_NAMES)
                print(f"[val] batch={batch_idx+1} coord_clean_mIoU={clean['mIoU']:.4f}", flush=True)
    return {
        "conf": {f"{m}:{v}": c.numpy() for (m, v), c in confs.items()},
        "keep_sums": keep_sums,
        "keep_counts": keep_counts,
    }


def summary_row(method: str, variant: Variant, repeat_count: int, mean_keep: float, conf: np.ndarray, base_summary: dict, weak_classes: list[int]):
    s = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    floor = NAME_TO_ID["floor"]
    return {
        "method": method,
        "variant": variant.name,
        "kind": variant.kind,
        "observed_keep_frac": mean_keep,
        "repeats": repeat_count,
        "mIoU": s["mIoU"],
        "delta_mIoU": s["mIoU"] - base_summary["mIoU"],
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


def write_results(args: argparse.Namespace, results: dict, variants: list[Variant], weak_classes: list[int], train_meta: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ["majority", "coord_mlp"]:
        base_conf = results["conf"][f"{method}:clean_voxel"]
        base = summarize_confusion(base_conf, SCANNET20_CLASS_NAMES)
        base["conf"] = base_conf
        for v in variants:
            rows.append(
                summary_row(
                    method,
                    v,
                    results["keep_counts"][v.name],
                    results["keep_sums"][v.name] / max(results["keep_counts"][v.name], 1),
                    results["conf"][f"{method}:{v.name}"],
                    base,
                    weak_classes,
                )
            )
    csv_path = args.output_dir / "masking_coord_baselines_summary.csv"
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
    lines = []
    lines.append("# Masking Coord/Majority Baselines\n")
    lines.append("Coordinate-only and train-majority baselines for the masking battery.\n")
    lines.append("## Setup\n")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Train batches: `{train_meta['seen_train_batches']}`")
    lines.append(f"- Train points stored: `{train_meta['train_points']}`")
    lines.append(f"- Majority class: `{SCANNET20_CLASS_NAMES[train_meta['majority_class']]}`")
    lines.append(f"- Class-balanced loss: `{args.class_balanced}`")
    lines.append("")
    lines.append("## Results\n")
    lines.append("| method | variant | keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | wall | floor | p->wall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['method']}` | `{r['variant']}` | {float(r['observed_keep_frac']):.4f} | "
            f"{float(r['mIoU']):.4f} | {float(r['delta_mIoU']):+.4f} | {float(r['allAcc']):.4f} | "
            f"{float(r['weak_mean_iou']):.4f} | {float(r['picture_iou']):.4f} | "
            f"{float(r['wall_iou']):.4f} | {float(r['floor_iou']):.4f} | {float(r['picture_to_wall']):.4f} |"
        )
    lines.append("")
    lines.append("## Files\n")
    lines.append(f"- Summary CSV: `{csv_path.resolve()}`")
    md = prefix.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    (args.output_dir / "masking_coord_baselines.md").write_text("\n".join(lines) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps({"train_meta": train_meta, "summary_md": str(md)}, indent=2) + "\n")
    print(f"[done] wrote {md}", flush=True)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.repo_root = args.repo_root.resolve()
    args.data_root = (args.repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (args.repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (args.repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    cfg = load_config(args.repo_root / args.config)
    num_classes = int(cfg.data.num_classes)
    variants = build_variants(args)
    weak_classes = parse_names(args.weak_classes)
    if args.dry_run:
        print(f"[dry-run] data_root={args.data_root}")
        print(f"[dry-run] variants={[v.name for v in variants]}")
        print(f"[dry-run] weak_classes={[SCANNET20_CLASS_NAMES[c] for c in weak_classes]}")
        return
    x, y, train_counts, seen = collect_train(args, cfg, num_classes)
    majority_class = int(torch.argmax(train_counts).item())
    train_meta = {
        "seen_train_batches": seen,
        "train_points": int(x.shape[0]),
        "majority_class": majority_class,
        "train_class_counts": train_counts.tolist(),
    }
    print(f"[train] meta={train_meta}", flush=True)
    model = train_coord_mlp(args, x, y, num_classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "train_meta": train_meta, "args": vars(args)}, args.output_dir / "coord_mlp_last.pth")
    results = eval_baselines(args, cfg, variants, model, majority_class, weak_classes, num_classes)
    write_results(args, results, variants, weak_classes, train_meta)


if __name__ == "__main__":
    main()
