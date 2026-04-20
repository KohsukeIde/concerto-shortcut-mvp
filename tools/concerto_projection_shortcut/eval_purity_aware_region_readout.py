#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
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
    update_confusion,
    weak_mean,
)


@dataclass(frozen=True)
class Variant:
    region_size: int
    proxy: str
    threshold: float
    alpha: float
    direction: str

    @property
    def name(self) -> str:
        t = str(self.threshold).replace(".", "p")
        a = str(self.alpha).replace(".", "p")
        return f"phrd_s{self.region_size}_{self.proxy}_{self.direction}_thr{t}_a{a}"


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Purity-aware hybrid region decoder (PHRD) zero-train gate. "
            "Computes label-free region purity proxies and sweeps point/region "
            "logit interpolation without fitting on heldout labels."
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
    parser.add_argument("--region-voxel-sizes", default="4,8")
    parser.add_argument("--thresholds", default="0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--alphas", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--directions", default="high,low")
    parser.add_argument(
        "--proxies",
        default="pred_agreement,mean_conf,region_conf,mean_top_gap,region_entropy_score",
    )
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--hard-class", default="picture")
    parser.add_argument("--hard-counterpart", default="wall")
    parser.add_argument("--min-target-points-per-region", type=int, default=64)
    parser.add_argument("--hard-counter-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_purity_aware_region_readout"),
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


def expanded_grid_coord(batch: dict) -> torch.Tensor:
    grid = batch.get("grid_coord")
    if grid is None:
        raise RuntimeError("grid_coord is required for PHRD")
    inv = batch.get("inverse")
    if inv is not None:
        return grid.long()[inv.long()]
    return grid.long()


def metric_from_summary(summary: dict, class_id: int) -> float:
    return float(summary["iou"][class_id])


def picture_to_wall_from_conf(conf: np.ndarray) -> float:
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    denom = conf[pic].sum()
    return float(conf[pic, wall] / denom) if denom else float("nan")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(x, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = rank_average(a[ok])
    rb = rank_average(b[ok])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def roc_auc(scores: list[float], labels: list[int]) -> float:
    if len(scores) < 2:
        return float("nan")
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    ok = np.isfinite(s)
    s = s[ok]
    y = y[ok]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rank_average(s)
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def summary_row(name: str, conf: np.ndarray, base_summary: dict, weak_classes: list[int]) -> dict[str, float | str]:
    s = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
    pic = NAME_TO_ID["picture"]
    return {
        "variant": name,
        "mIoU": s["mIoU"],
        "delta_mIoU": s["mIoU"] - base_summary["mIoU"],
        "weak_mean_iou": weak_mean(s, weak_classes),
        "delta_weak_mean_iou": weak_mean(s, weak_classes) - weak_mean(base_summary, weak_classes),
        "picture_iou": metric_from_summary(s, pic),
        "delta_picture_iou": metric_from_summary(s, pic) - metric_from_summary(base_summary, pic),
        "picture_to_wall": picture_to_wall_from_conf(conf),
        "delta_picture_to_wall": picture_to_wall_from_conf(conf) - picture_to_wall_from_conf(base_summary["conf"]),
    }


def build_variants(region_sizes: list[int], proxies: list[str], thresholds: list[float], alphas: list[float], directions: list[str]) -> list[Variant]:
    out = []
    for direction in directions:
        if direction not in {"high", "low"}:
            raise ValueError(f"unknown direction: {direction}")
        out.extend(Variant(s, p, t, a, direction) for s in region_sizes for p in proxies for t in thresholds for a in alphas)
    return out


def region_proxy_tensors(
    logits: torch.Tensor,
    feat: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    inv: torch.Tensor,
    region_count: torch.Tensor,
    region_logits: torch.Tensor,
    num_classes: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = logits.device
    r = int(region_count.numel())
    probs = torch.softmax(logits, dim=1)
    top2 = probs.topk(2, dim=1).values
    conf = top2[:, 0]
    top_gap = top2[:, 0] - top2[:, 1]

    conf_sum = torch.zeros(r, device=device)
    gap_sum = torch.zeros(r, device=device)
    conf_sum.index_add_(0, inv, conf)
    gap_sum.index_add_(0, inv, top_gap)
    mean_conf = conf_sum / region_count
    mean_top_gap = gap_sum / region_count

    pred_onehot = F.one_hot(pred.long(), num_classes=num_classes).float()
    pred_count = torch.zeros((r, num_classes), device=device)
    pred_count.index_add_(0, inv, pred_onehot)
    pred_agreement = pred_count.max(dim=1).values / region_count

    region_probs = torch.softmax(region_logits, dim=1)
    region_conf = region_probs.max(dim=1).values
    region_entropy = -(region_probs * region_probs.clamp_min(1e-8).log()).sum(dim=1)
    region_entropy_score = 1.0 - region_entropy / float(np.log(num_classes))

    feat = feat.float()
    feat_norm = F.normalize(feat, dim=1)
    feat_sum = torch.zeros((r, feat_norm.shape[1]), device=device)
    feat_sq_sum = torch.zeros(r, device=device)
    feat_sum.index_add_(0, inv, feat_norm)
    feat_sq_sum.index_add_(0, inv, (feat_norm * feat_norm).sum(dim=1))
    feat_mean = feat_sum / region_count[:, None]
    feat_var = (feat_sq_sum / region_count) - (feat_mean * feat_mean).sum(dim=1)
    feat_tight = 1.0 / (1.0 + feat_var.clamp_min(0.0))

    gt_onehot = F.one_hot(target.long(), num_classes=num_classes).float()
    gt_count = torch.zeros((r, num_classes), device=device)
    gt_count.index_add_(0, inv, gt_onehot)
    true_purity = gt_count.max(dim=1).values / region_count

    proxies = {
        "pred_agreement": pred_agreement.clamp(0, 1),
        "mean_conf": mean_conf.clamp(0, 1),
        "region_conf": region_conf.clamp(0, 1),
        "mean_top_gap": mean_top_gap.clamp(0, 1),
        "region_entropy_score": region_entropy_score.clamp(0, 1),
        "feat_tight": feat_tight.clamp(0, 1),
    }
    point_scores = {name: value[inv] for name, value in proxies.items()}
    diag = {
        "true_purity": true_purity,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "region_count": region_count,
    }
    return proxies, point_scores, diag


def update_proxy_records(
    proxy_records: dict[tuple[int, str], dict[str, list[float] | list[int]]],
    proxies: dict[str, torch.Tensor],
    diag: dict[str, torch.Tensor],
    region_size: int,
    hard_class: int,
    hard_counterpart: int,
    min_target_points: int,
    hard_counter_frac: float,
) -> None:
    true_purity = diag["true_purity"].detach().cpu().numpy()
    gt_count = diag["gt_count"].detach().cpu().numpy()
    pred_count = diag["pred_count"].detach().cpu().numpy()
    region_count = diag["region_count"].detach().cpu().numpy()
    target_count = gt_count[:, hard_class]
    counter_pred = pred_count[:, hard_counterpart]
    hard_region = (target_count >= min_target_points) & ((counter_pred / np.maximum(target_count, 1.0)) >= hard_counter_frac)
    target_region = target_count >= min_target_points
    pure_region = true_purity >= 0.9
    for name, values in proxies.items():
        scores = values.detach().cpu().numpy()
        key = (region_size, name)
        rec = proxy_records[key]
        rec["score"].extend(scores.astype(float).tolist())
        rec["true_purity"].extend(true_purity.astype(float).tolist())
        rec["pure_region"].extend(pure_region.astype(int).tolist())
        rec["picture_score"].extend(scores[target_region].astype(float).tolist())
        rec["hard_picture_region"].extend(hard_region[target_region].astype(int).tolist())
        rec["picture_true_purity"].extend(true_purity[target_region].astype(float).tolist())


def eval_phrd(args: argparse.Namespace, model, cfg, variants: list[Variant], region_sizes: list[int], proxies: list[str], weak_classes: list[int], num_classes: int):
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    base_conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    region_conf = {s: torch.zeros((num_classes, num_classes), dtype=torch.long) for s in region_sizes}
    variant_conf = {v.name: torch.zeros((num_classes, num_classes), dtype=torch.long) for v in variants}
    proxy_records: dict[tuple[int, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seen = 0
    variant_by_region: dict[int, list[Variant]] = defaultdict(list)
    for v in variants:
        variant_by_region[v.region_size].append(v)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            feat_voxel, logits_voxel, labels_voxel, batch = forward_features(model, batch)
            feat, logits, target = eval_tensors(feat_voxel, logits_voxel, labels_voxel, batch)
            grid = expanded_grid_coord(batch)
            if grid.device != logits.device:
                grid = grid.to(logits.device)
            feat = feat.to(logits.device)
            target = target.to(logits.device)
            valid = (target >= 0) & (target < num_classes)
            feat = feat[valid]
            logits = logits[valid]
            target = target[valid]
            grid = grid[valid]
            base_pred = logits.argmax(dim=1)
            update_confusion(base_conf, base_pred.cpu(), target.cpu(), num_classes, -1)

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

                region_proxies, point_scores, diag = region_proxy_tensors(
                    logits, feat, base_pred, target, inv, region_count, region_logits, num_classes
                )
                update_proxy_records(
                    proxy_records,
                    {k: region_proxies[k] for k in proxies if k in region_proxies},
                    diag,
                    s,
                    NAME_TO_ID[args.hard_class],
                    NAME_TO_ID[args.hard_counterpart],
                    args.min_target_points_per_region,
                    args.hard_counter_frac,
                )
                for v in variant_by_region[s]:
                    score = point_scores[v.proxy]
                    mask = score >= v.threshold if v.direction == "high" else score <= v.threshold
                    pred = base_pred.clone()
                    if bool(mask.any()):
                        mixed = (1.0 - v.alpha) * logits[mask] + v.alpha * point_region_logits[mask]
                        pred[mask] = mixed.argmax(dim=1)
                    update_confusion(variant_conf[v.name], pred.cpu(), target.cpu(), num_classes, -1)

            seen += 1
            if (batch_idx + 1) % 25 == 0:
                base = summarize_confusion(base_conf.numpy(), SCANNET20_CLASS_NAMES)
                print(f"[val] batch={batch_idx+1} base_mIoU={base['mIoU']:.4f}", flush=True)

    return {
        "seen_batches": seen,
        "base_conf": base_conf.numpy(),
        "region_conf": {s: c.numpy() for s, c in region_conf.items()},
        "variant_conf": {name: c.numpy() for name, c in variant_conf.items()},
        "proxy_records": proxy_records,
    }


def write_results(args: argparse.Namespace, results: dict, variants: list[Variant], region_sizes: list[int], proxies: list[str], weak_classes: list[int]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_summary = summarize_confusion(results["base_conf"], SCANNET20_CLASS_NAMES)
    base_summary["conf"] = results["base_conf"]
    rows = [summary_row("base", results["base_conf"], base_summary, weak_classes)]
    for s, conf in results["region_conf"].items():
        rows.append(summary_row(f"region_logits_s{s}", conf, base_summary, weak_classes))
    for v in variants:
        rows.append(summary_row(v.name, results["variant_conf"][v.name], base_summary, weak_classes))
    rows_sorted = sorted(rows, key=lambda r: (r["variant"] == "base", -float(r["mIoU"])))

    proxy_rows = []
    for (region_size, proxy), rec in sorted(results["proxy_records"].items()):
        proxy_rows.append(
            {
                "region_size": region_size,
                "proxy": proxy,
                "n_regions": len(rec["score"]),
                "spearman_true_purity": spearman(rec["score"], rec["true_purity"]),
                "auc_purity_ge_0p9": roc_auc(rec["score"], rec["pure_region"]),
                "n_picture_regions": len(rec["picture_score"]),
                "spearman_picture_true_purity": spearman(rec["picture_score"], rec["picture_true_purity"]),
                "auc_hard_picture_wall": roc_auc(rec["picture_score"], rec["hard_picture_region"]),
                "hard_picture_region_rate": safe_div(sum(rec["hard_picture_region"]), len(rec["hard_picture_region"])),
            }
        )

    variant_csv = args.output_dir / "purity_aware_region_variants.csv"
    proxy_csv = args.output_dir / "purity_proxy_diagnostic.csv"
    with variant_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with proxy_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(proxy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(proxy_rows)

    prefix = args.summary_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    pic = NAME_TO_ID["picture"]
    best_miou = max(rows, key=lambda r: float(r["mIoU"]))
    safe = [r for r in rows if float(r["mIoU"]) >= base_summary["mIoU"] - 0.002]
    best_safe_pic = max(safe, key=lambda r: float(r["picture_iou"])) if safe else None
    proxy_sorted = sorted(proxy_rows, key=lambda r: np.nan_to_num(float(r["auc_hard_picture_wall"]), nan=-1.0), reverse=True)
    lines = []
    lines.append("# Purity-Aware Hybrid Region Decoder Gate\n")
    lines.append("Zero-train gate for PHRD: use label-free region purity proxies to decide when to mix region logits with point logits.\n")
    lines.append("## Setup\n")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Weight: `{args.weight}`")
    lines.append(f"- Region sizes: `{args.region_voxel_sizes}`")
    lines.append(f"- Proxies: `{args.proxies}`")
    lines.append(f"- Thresholds: `{args.thresholds}`")
    lines.append(f"- Alphas: `{args.alphas}`")
    lines.append(f"- Directions: `{args.directions}`")
    lines.append("")
    lines.append("## Headline\n")
    lines.append(
        f"- Base: mIoU={base_summary['mIoU']:.4f}, picture={metric_from_summary(base_summary, pic):.4f}, "
        f"picture->wall={picture_to_wall_from_conf(results['base_conf']):.4f}"
    )
    lines.append(
        f"- Best mIoU: `{best_miou['variant']}` mIoU={float(best_miou['mIoU']):.4f} "
        f"(Δ{float(best_miou['delta_mIoU']):+.4f}), picture={float(best_miou['picture_iou']):.4f} "
        f"(Δ{float(best_miou['delta_picture_iou']):+.4f}), p->wall={float(best_miou['picture_to_wall']):.4f} "
        f"(Δ{float(best_miou['delta_picture_to_wall']):+.4f})"
    )
    if best_safe_pic:
        lines.append(
            f"- Best safe picture: `{best_safe_pic['variant']}` mIoU={float(best_safe_pic['mIoU']):.4f} "
            f"(Δ{float(best_safe_pic['delta_mIoU']):+.4f}), picture={float(best_safe_pic['picture_iou']):.4f} "
            f"(Δ{float(best_safe_pic['delta_picture_iou']):+.4f}), p->wall={float(best_safe_pic['picture_to_wall']):.4f} "
            f"(Δ{float(best_safe_pic['delta_picture_to_wall']):+.4f})"
        )
    lines.append("")
    lines.append("## Top Variants\n")
    lines.append("| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | p->wall | Δp->wall |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows_sorted[:20], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {float(r['mIoU']):.4f} | {float(r['delta_mIoU']):+.4f} | "
            f"{float(r['weak_mean_iou']):.4f} | {float(r['delta_weak_mean_iou']):+.4f} | "
            f"{float(r['picture_iou']):.4f} | {float(r['delta_picture_iou']):+.4f} | "
            f"{float(r['picture_to_wall']):.4f} | {float(r['delta_picture_to_wall']):+.4f} |"
        )
    lines.append("")
    lines.append("## Proxy Diagnostic\n")
    lines.append("| rank | region | proxy | rho purity | AUC purity>=0.9 | rho picture purity | AUC hard picture-wall | hard rate |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for i, r in enumerate(proxy_sorted[:16], 1):
        lines.append(
            f"| {i} | {int(r['region_size'])} | `{r['proxy']}` | "
            f"{float(r['spearman_true_purity']):.4f} | {float(r['auc_purity_ge_0p9']):.4f} | "
            f"{float(r['spearman_picture_true_purity']):.4f} | {float(r['auc_hard_picture_wall']):.4f} | "
            f"{float(r['hard_picture_region_rate']):.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation Gate\n")
    lines.append("- Strong PHRD go: mIoU >= base +0.003, or picture >= base +0.02 while mIoU >= base -0.002.")
    lines.append("- Proxy go: label-free proxy should have useful hard-picture-wall AUC and should not merely select wall-dominated coarse regions.")
    lines.append("- If variants are near-tie/no-go, purity-aware coarse region mixing is diagnostic only; next method needs better object proposals or learned masks.")
    lines.append("")
    lines.append("## Files\n")
    lines.append(f"- Variant CSV: `{variant_csv.resolve()}`")
    lines.append(f"- Proxy CSV: `{proxy_csv.resolve()}`")
    md = prefix.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    (args.output_dir / "purity_aware_region_readout.md").write_text("\n".join(lines) + "\n")
    metadata = {
        "config": str(args.config),
        "weight": str(args.weight),
        "region_sizes": parse_int_list(args.region_voxel_sizes),
        "proxies": parse_str_list(args.proxies),
        "thresholds": parse_float_list(args.thresholds),
        "alphas": parse_float_list(args.alphas),
        "directions": parse_str_list(args.directions),
        "seen_batches": results["seen_batches"],
        "outputs": {
            "variant_csv": str(variant_csv),
            "proxy_csv": str(proxy_csv),
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
    thresholds = parse_float_list(args.thresholds)
    alphas = parse_float_list(args.alphas)
    proxies = parse_str_list(args.proxies)
    directions = parse_str_list(args.directions)
    weak_classes = parse_names(args.weak_classes)
    if args.hard_class not in NAME_TO_ID or args.hard_counterpart not in NAME_TO_ID:
        raise ValueError("unknown hard class or counterpart")
    variants = build_variants(region_sizes, proxies, thresholds, alphas, directions)
    if args.dry_run:
        print(f"[dry-run] data_root={args.data_root}")
        print(f"[dry-run] region_sizes={region_sizes}")
        print(f"[dry-run] proxies={proxies}")
        print(f"[dry-run] directions={directions}")
        print(f"[dry-run] variants={len(variants)}")
        return
    cfg = load_config(args.repo_root / args.config)
    num_classes = int(cfg.data.num_classes)
    model = build_model(cfg, args.weight)
    results = eval_phrd(args, model, cfg, variants, region_sizes, proxies, weak_classes, num_classes)
    write_results(args, results, variants, region_sizes, proxies, weak_classes)


if __name__ == "__main__":
    main()
