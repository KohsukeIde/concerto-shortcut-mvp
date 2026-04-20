#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
)


@dataclass
class RecallStats:
    target_points: int = 0
    valid_points: int = 0
    proposals: int = 0
    target_proposals: int = 0
    target_purity_sum: float = 0.0
    counterpart_contam_sum: float = 0.0
    counterpart_majority_points: int = 0
    counterpart_dominates_points: int = 0
    base_pred_target_majority_points: int = 0
    base_pred_counterpart_majority_points: int = 0
    scene_count: int = 0
    scene_best_cover_sum: float = 0.0
    recall_num: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    candidate_points: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    candidate_regions: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    candidate_size_sum: dict[float, int] = field(default_factory=lambda: defaultdict(int))


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Proposal recall gate for proposal-first / mask-lite readout. "
            "Measures whether fine voxel proposals or connected same-prediction "
            "components can cover weak classes as high-purity candidate regions."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--region-voxel-sizes", default="4,8,16")
    parser.add_argument("--purity-thresholds", default="0.5,0.7,0.8,0.9")
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall",
    )
    parser.add_argument(
        "--proposal-sources",
        default="voxel,pred_cc",
        help="Comma-separated proposal sources. Supported: voxel, pred_cc.",
    )
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_proposal_recall_analysis"),
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


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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


def expanded_grid_coord(batch: dict) -> torch.Tensor:
    grid = batch.get("grid_coord")
    if grid is None:
        raise RuntimeError("grid_coord is required for proposal recall analysis")
    inv = batch.get("inverse")
    if inv is not None:
        return grid.long()[inv.long()]
    return grid.long()


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def picture_to_wall_from_conf(conf: np.ndarray) -> float:
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    denom = conf[pic].sum()
    return float(conf[pic, wall] / denom) if denom else float("nan")


def build_pred_connected_components(region_keys: torch.Tensor, majority_pred: torch.Tensor) -> torch.Tensor:
    """Merge adjacent voxel regions only when their base-pred majority class matches."""
    keys = region_keys.detach().cpu().numpy().astype(np.int64)
    labels = majority_pred.detach().cpu().numpy().astype(np.int64)
    key_to_idx = {tuple(k.tolist()): i for i, k in enumerate(keys)}
    visited = np.zeros(len(keys), dtype=bool)
    comp = np.full(len(keys), -1, dtype=np.int64)
    offsets = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    comp_id = 0
    for start in range(len(keys)):
        if visited[start]:
            continue
        visited[start] = True
        comp[start] = comp_id
        q: deque[int] = deque([start])
        label = labels[start]
        while q:
            cur = q.popleft()
            x, y, z = keys[cur]
            for dx, dy, dz in offsets:
                nxt = key_to_idx.get((int(x + dx), int(y + dy), int(z + dz)))
                if nxt is None or visited[nxt] or labels[nxt] != label:
                    continue
                visited[nxt] = True
                comp[nxt] = comp_id
                q.append(nxt)
        comp_id += 1
    return torch.from_numpy(comp)


def aggregate_counts(
    inv: torch.Tensor,
    target: torch.Tensor,
    base_pred: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    region_count = torch.bincount(inv).float().clamp_min(1.0)
    r = int(region_count.numel())
    gt_onehot = F.one_hot(target.long(), num_classes=num_classes).float()
    gt_count = torch.zeros((r, num_classes), device=target.device)
    gt_count.index_add_(0, inv, gt_onehot)
    pred_onehot = F.one_hot(base_pred.long(), num_classes=num_classes).float()
    pred_count = torch.zeros((r, num_classes), device=target.device)
    pred_count.index_add_(0, inv, pred_onehot)
    return region_count, gt_count, pred_count


def update_recall_stats(
    stats: dict[tuple[str, int, int, int], RecallStats],
    source: str,
    region_size: int,
    region_count: torch.Tensor,
    gt_count: torch.Tensor,
    pred_count: torch.Tensor,
    valid_points: int,
    pairs: list[tuple[int, int]],
    thresholds: list[float],
) -> None:
    for cls, counterpart in pairs:
        class_count = gt_count[:, cls]
        target_total = int(class_count.sum().item())
        if target_total == 0:
            continue
        counter_count = gt_count[:, counterpart]
        majority_pred = pred_count.argmax(dim=1)
        class_purity = class_count / region_count
        counter_frac = counter_count / region_count
        target_region = class_count > 0

        rec = stats[(source, region_size, cls, counterpart)]
        rec.target_points += target_total
        rec.valid_points += int(valid_points)
        rec.proposals += int(region_count.numel())
        rec.target_proposals += int(target_region.sum().item())
        rec.target_purity_sum += float((class_count * class_purity).sum().item())
        rec.counterpart_contam_sum += float((class_count * counter_frac).sum().item())
        rec.counterpart_majority_points += int(class_count[counter_frac >= 0.5].sum().item())
        rec.counterpart_dominates_points += int(class_count[counter_count > class_count].sum().item())
        rec.base_pred_target_majority_points += int(class_count[majority_pred == cls].sum().item())
        rec.base_pred_counterpart_majority_points += int(class_count[majority_pred == counterpart].sum().item())
        rec.scene_count += 1
        rec.scene_best_cover_sum += float(class_count.max().item() / target_total)
        for thr in thresholds:
            cand = target_region & (class_purity >= thr)
            rec.recall_num[thr] += int(class_count[cand].sum().item())
            rec.candidate_points[thr] += int(region_count[cand].sum().item())
            rec.candidate_regions[thr] += int(cand.sum().item())
            rec.candidate_size_sum[thr] += int(region_count[cand].sum().item())


def eval_proposals(args: argparse.Namespace, model, cfg, region_sizes: list[int], pairs: list[tuple[int, int]], thresholds: list[float], sources: list[str], num_classes: int):
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    stats: dict[tuple[str, int, int, int], RecallStats] = defaultdict(RecallStats)
    base_conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    seen = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            _, logits_voxel, labels_voxel, batch = forward_features(model, batch)
            _, logits, target = eval_tensors(logits_voxel, logits_voxel, labels_voxel, batch)
            grid = expanded_grid_coord(batch)
            if grid.device != logits.device:
                grid = grid.to(logits.device)
            target = target.to(logits.device)
            valid = (target >= 0) & (target < num_classes)
            logits = logits[valid]
            target = target[valid]
            grid = grid[valid]
            base_pred = logits.argmax(dim=1)

            flat = target.cpu() * num_classes + base_pred.cpu()
            base_conf += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

            for s in region_sizes:
                keys = torch.div(grid, s, rounding_mode="floor")
                unique_keys, inv = torch.unique(keys, dim=0, return_inverse=True)
                region_count, gt_count, pred_count = aggregate_counts(inv, target, base_pred, num_classes)
                if "voxel" in sources:
                    update_recall_stats(
                        stats,
                        "voxel",
                        s,
                        region_count,
                        gt_count,
                        pred_count,
                        int(target.numel()),
                        pairs,
                        thresholds,
                    )

                if "pred_cc" in sources:
                    majority_pred = pred_count.argmax(dim=1)
                    comp_for_region = build_pred_connected_components(unique_keys, majority_pred).to(inv.device)
                    comp_inv = comp_for_region[inv].long()
                    comp_count, comp_gt_count, comp_pred_count = aggregate_counts(comp_inv, target, base_pred, num_classes)
                    update_recall_stats(
                        stats,
                        "pred_cc",
                        s,
                        comp_count,
                        comp_gt_count,
                        comp_pred_count,
                        int(target.numel()),
                        pairs,
                        thresholds,
                    )

            seen += 1
            if (batch_idx + 1) % 25 == 0:
                base = summarize_confusion(base_conf.numpy(), SCANNET20_CLASS_NAMES)
                print(f"[val] batch={batch_idx+1} base_mIoU={base['mIoU']:.4f}", flush=True)

    return {
        "seen_batches": seen,
        "base_conf": base_conf.numpy(),
        "stats": stats,
    }


def build_rows(results: dict, thresholds: list[float]) -> list[dict[str, float | int | str]]:
    rows = []
    for (source, s, cls, counterpart), rec in sorted(results["stats"].items()):
        for thr in thresholds:
            rows.append(
                {
                    "source": source,
                    "region_size": s,
                    "threshold": thr,
                    "class_id": cls,
                    "class_name": SCANNET20_CLASS_NAMES[cls],
                    "counterpart_id": counterpart,
                    "counterpart_name": SCANNET20_CLASS_NAMES[counterpart],
                    "target_points": rec.target_points,
                    "valid_points": rec.valid_points,
                    "proposals": rec.proposals,
                    "target_proposals": rec.target_proposals,
                    "mean_target_purity": safe_ratio(rec.target_purity_sum, rec.target_points),
                    "mean_counterpart_contam": safe_ratio(rec.counterpart_contam_sum, rec.target_points),
                    "counterpart_majority_point_frac": safe_ratio(rec.counterpart_majority_points, rec.target_points),
                    "counterpart_dominates_point_frac": safe_ratio(rec.counterpart_dominates_points, rec.target_points),
                    "base_pred_target_majority_frac": safe_ratio(rec.base_pred_target_majority_points, rec.target_points),
                    "base_pred_counterpart_majority_frac": safe_ratio(rec.base_pred_counterpart_majority_points, rec.target_points),
                    "scene_count": rec.scene_count,
                    "scene_mean_best_proposal_cover": safe_ratio(rec.scene_best_cover_sum, rec.scene_count),
                    "candidate_recall": safe_ratio(rec.recall_num[thr], rec.target_points),
                    "candidate_point_frac": safe_ratio(rec.candidate_points[thr], rec.valid_points),
                    "candidate_region_count": rec.candidate_regions[thr],
                    "candidate_mean_size": safe_ratio(rec.candidate_size_sum[thr], rec.candidate_regions[thr]),
                }
            )
    return rows


def write_results(args: argparse.Namespace, results: dict, region_sizes: list[int], pairs: list[tuple[int, int]], thresholds: list[float], sources: list[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(results, thresholds)
    detail_csv = args.output_dir / "proposal_recall_detail.csv"
    with detail_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    prefix = args.summary_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    base = summarize_confusion(results["base_conf"], SCANNET20_CLASS_NAMES)
    pic = NAME_TO_ID["picture"]
    pic_rows = [r for r in rows if r["class_name"] == "picture"]
    pic_rows = sorted(pic_rows, key=lambda r: (str(r["source"]), int(r["region_size"]), float(r["threshold"])))
    high_thr = max(thresholds)
    high_rows = [r for r in rows if abs(float(r["threshold"]) - high_thr) < 1e-9]
    high_sorted = sorted(high_rows, key=lambda r: (str(r["class_name"]), str(r["source"]), int(r["region_size"])))

    lines = []
    lines.append("# Proposal Recall Analysis\n")
    lines.append("Gate for proposal-first / mask-lite readout: can simple local proposals cover weak-class points as high-purity candidates?\n")
    lines.append("## Setup\n")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Weight: `{args.weight}`")
    lines.append(f"- Proposal sources: `{','.join(sources)}`")
    lines.append(f"- Region voxel sizes: `{','.join(str(x) for x in region_sizes)}`")
    lines.append(f"- Purity thresholds: `{','.join(str(x) for x in thresholds)}`")
    lines.append(f"- Class pairs: `{args.class_pairs}`")
    lines.append(f"- Seen val batches: `{results['seen_batches']}`")
    lines.append("")
    lines.append("## Base Decoder Reference\n")
    lines.append(
        f"- mIoU={base['mIoU']:.4f}, picture={base['iou'][pic]:.4f}, picture->wall={picture_to_wall_from_conf(results['base_conf']):.4f}"
    )
    lines.append("")
    lines.append("## Picture Proposal Recall\n")
    lines.append("| source | s | thr | recall | candidate point frac | candidate regions | mean purity | wall contam | wall-majority frac | pred-picture maj | pred-wall maj | best cover/scene |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in pic_rows:
        lines.append(
            f"| `{r['source']}` | {int(r['region_size'])} | {float(r['threshold']):.1f} | "
            f"{float(r['candidate_recall']):.4f} | {float(r['candidate_point_frac']):.4f} | "
            f"{int(r['candidate_region_count'])} | {float(r['mean_target_purity']):.4f} | "
            f"{float(r['mean_counterpart_contam']):.4f} | {float(r['counterpart_majority_point_frac']):.4f} | "
            f"{float(r['base_pred_target_majority_frac']):.4f} | {float(r['base_pred_counterpart_majority_frac']):.4f} | "
            f"{float(r['scene_mean_best_proposal_cover']):.4f} |"
        )
    lines.append("")
    lines.append(f"## High-Purity Recall At Threshold {high_thr:.1f}\n")
    lines.append("| class | counterpart | source | s | recall | candidate point frac | candidate regions | mean purity | counterpart contam | counterpart-majority frac |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in high_sorted:
        lines.append(
            f"| `{r['class_name']}` | `{r['counterpart_name']}` | `{r['source']}` | {int(r['region_size'])} | "
            f"{float(r['candidate_recall']):.4f} | {float(r['candidate_point_frac']):.4f} | "
            f"{int(r['candidate_region_count'])} | {float(r['mean_target_purity']):.4f} | "
            f"{float(r['mean_counterpart_contam']):.4f} | {float(r['counterpart_majority_point_frac']):.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation Gate\n")
    lines.append("- PVD go: a weak class has high-purity recall at `thr>=0.8` with a reasonably small candidate point fraction.")
    lines.append("- Fine-only go: `voxel s4` has high recall but `pred_cc` / coarse sizes collapse; proposal-first needs learned object-quality masks, not region averaging.")
    lines.append("- No-go: even `voxel s4` has low high-purity recall; proposal-first is unlikely to recover the oracle headroom under this feature family.")
    lines.append("")
    lines.append("## Files\n")
    lines.append(f"- Detail CSV: `{detail_csv.resolve()}`")
    md = prefix.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    (args.output_dir / "proposal_recall_analysis.md").write_text("\n".join(lines) + "\n")
    metadata = {
        "config": str(args.config),
        "weight": str(args.weight),
        "region_sizes": region_sizes,
        "thresholds": thresholds,
        "sources": sources,
        "pairs": [(SCANNET20_CLASS_NAMES[a], SCANNET20_CLASS_NAMES[b]) for a, b in pairs],
        "seen_batches": results["seen_batches"],
        "outputs": {
            "detail_csv": str(detail_csv),
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
    thresholds = parse_float_list(args.purity_thresholds)
    pairs = parse_pairs(args.class_pairs)
    sources = parse_str_list(args.proposal_sources)
    unsupported = set(sources) - {"voxel", "pred_cc"}
    if unsupported:
        raise ValueError(f"unsupported proposal sources: {sorted(unsupported)}")
    if args.dry_run:
        print(f"[dry-run] data_root={args.data_root}")
        print(f"[dry-run] weight={args.weight}")
        print(f"[dry-run] region_sizes={region_sizes}")
        print(f"[dry-run] thresholds={thresholds}")
        print(f"[dry-run] sources={sources}")
        print(f"[dry-run] pairs={[(SCANNET20_CLASS_NAMES[a], SCANNET20_CLASS_NAMES[b]) for a, b in pairs]}")
        return
    cfg = load_config(args.repo_root / args.config)
    num_classes = int(cfg.data.num_classes)
    model = build_model(cfg, args.weight)
    results = eval_proposals(args, model, cfg, region_sizes, pairs, thresholds, sources, num_classes)
    write_results(args, results, region_sizes, pairs, thresholds, sources)


if __name__ == "__main__":
    main()
