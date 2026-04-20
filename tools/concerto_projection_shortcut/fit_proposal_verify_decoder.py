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
    threshold: float
    beta: float

    @property
    def name(self) -> str:
        t = str(self.threshold).replace(".", "p")
        b = str(self.beta).replace(".", "p")
        return f"pvd_thr{t}_b{b}"


class ProposalMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Proposal-then-Verify Decoder (PVD) pilot. "
            "Train a lightweight proposal classifier on fine voxel proposals, then "
            "use verified hard-class proposals to add conservative logit boosts."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=256)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--region-voxel-size", type=int, default=4)
    parser.add_argument("--positive-purity", type=float, default=0.8)
    parser.add_argument(
        "--proposal-classes",
        default="picture,counter,desk,sink,door,shower curtain,cabinet,table,wall",
    )
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--max-per-class", type=int, default=30000)
    parser.add_argument("--max-background", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--proposal-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--thresholds", default="0.5,0.7,0.9")
    parser.add_argument("--betas", default="0.25,0.5,1.0")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_proposal_verify_decoder"),
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


def expanded_grid_coord(batch: dict) -> torch.Tensor:
    grid = batch.get("grid_coord")
    if grid is None:
        raise RuntimeError("grid_coord is required for PVD")
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


def proposal_features(
    feat: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    grid: torch.Tensor,
    region_size: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = torch.div(grid, region_size, rounding_mode="floor")
    _, inv = torch.unique(keys, dim=0, return_inverse=True)
    region_count = torch.bincount(inv).float().clamp_min(1.0)
    r = int(region_count.numel())

    feat_norm = F.normalize(feat.float(), dim=1)
    feat_sum = torch.zeros((r, feat_norm.shape[1]), device=feat.device)
    feat_sum.index_add_(0, inv, feat_norm)
    feat_mean = F.normalize(feat_sum / region_count[:, None], dim=1)

    probs = torch.softmax(logits.float(), dim=1)
    prob_sum = torch.zeros((r, num_classes), device=logits.device)
    prob_sum.index_add_(0, inv, probs)
    prob_mean = prob_sum / region_count[:, None]
    top2 = prob_mean.topk(2, dim=1).values
    region_conf = top2[:, :1]
    region_gap = (top2[:, :1] - top2[:, 1:2]).clamp_min(0)
    entropy = -(prob_mean * prob_mean.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
    entropy_score = 1.0 - entropy / float(np.log(num_classes))
    count_feat = torch.log1p(region_count).unsqueeze(1) / 8.0

    pred = logits.argmax(dim=1)
    pred_onehot = F.one_hot(pred.long(), num_classes=num_classes).float()
    pred_count = torch.zeros((r, num_classes), device=logits.device)
    pred_count.index_add_(0, inv, pred_onehot)
    pred_agreement = (pred_count.max(dim=1).values / region_count).unsqueeze(1)

    gt_onehot = F.one_hot(target.long(), num_classes=num_classes).float()
    gt_count = torch.zeros((r, num_classes), device=target.device)
    gt_count.index_add_(0, inv, gt_onehot)

    x = torch.cat([feat_mean, prob_mean, region_conf, region_gap, entropy_score, count_feat, pred_agreement], dim=1)
    return x, inv, region_count, gt_count, pred_count, prob_mean


def proposal_labels(
    gt_count: torch.Tensor,
    region_count: torch.Tensor,
    proposal_classes: list[int],
    positive_purity: float,
) -> torch.Tensor:
    class_counts = gt_count[:, proposal_classes]
    class_purity = class_counts / region_count[:, None]
    best_purity, best_idx = class_purity.max(dim=1)
    bg = len(proposal_classes)
    labels = torch.full((gt_count.shape[0],), bg, dtype=torch.long, device=gt_count.device)
    labels[best_purity >= positive_purity] = best_idx[best_purity >= positive_purity]
    return labels


def append_capped(
    feats_by_label: dict[int, list[torch.Tensor]],
    labels_by_label: dict[int, list[torch.Tensor]],
    x: torch.Tensor,
    y: torch.Tensor,
    num_labels: int,
    max_per_class: int,
    max_background: int,
) -> None:
    for label in range(num_labels):
        cap = max_background if label == num_labels - 1 else max_per_class
        current = sum(t.shape[0] for t in feats_by_label[label])
        remaining = cap - current
        if remaining <= 0:
            continue
        idx = torch.nonzero(y == label, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        if idx.numel() > remaining:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:remaining]]
        feats_by_label[label].append(x[idx].detach().cpu())
        labels_by_label[label].append(y[idx].detach().cpu())


def collect_train_bank(args: argparse.Namespace, model, cfg, proposal_classes: list[int], num_classes: int):
    loader = build_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker)
    num_labels = len(proposal_classes) + 1
    feats_by_label: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_labels)}
    labels_by_label: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_labels)}
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
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
            x, _, region_count, gt_count, _, _ = proposal_features(
                feat, logits, target, grid, args.region_voxel_size, num_classes
            )
            y = proposal_labels(gt_count, region_count, proposal_classes, args.positive_purity)
            append_capped(
                feats_by_label,
                labels_by_label,
                x,
                y,
                num_labels,
                args.max_per_class,
                args.max_background,
            )
            seen += 1
            if (batch_idx + 1) % 50 == 0:
                counts = [sum(t.shape[0] for t in feats_by_label[i]) for i in range(num_labels)]
                print(f"[train-bank] batch={batch_idx+1} counts={counts}", flush=True)
    xs = []
    ys = []
    for i in range(num_labels):
        if feats_by_label[i]:
            xs.append(torch.cat(feats_by_label[i], dim=0))
            ys.append(torch.cat(labels_by_label[i], dim=0))
    if not xs:
        raise RuntimeError("no proposal training data collected")
    x_all = torch.cat(xs, dim=0).float()
    y_all = torch.cat(ys, dim=0).long()
    perm = torch.randperm(x_all.shape[0])
    return x_all[perm], y_all[perm], seen


def train_proposal_model(args: argparse.Namespace, x: torch.Tensor, y: torch.Tensor, num_labels: int) -> ProposalMLP:
    device = torch.device("cuda")
    model = ProposalMLP(x.shape[1], num_labels).to(device)
    counts = torch.bincount(y, minlength=num_labels).float()
    weights = (counts.sum() / counts.clamp_min(1.0)).sqrt()
    weights = weights / weights.mean()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.proposal_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weights.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(yb.numel())
            total += int(yb.numel())
            correct += int((logits.argmax(dim=1) == yb).sum().item())
        print(
            f"[proposal-train] epoch={epoch+1}/{args.epochs} loss={total_loss/max(total,1):.4f} acc={correct/max(total,1):.4f}",
            flush=True,
        )
    return model.eval()


def summary_row(name: str, conf: np.ndarray, base_summary: dict, weak_classes: list[int]) -> dict[str, float | str]:
    s = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
    pic = NAME_TO_ID["picture"]
    return {
        "variant": name,
        "mIoU": s["mIoU"],
        "delta_mIoU": s["mIoU"] - base_summary["mIoU"],
        "weak_mean_iou": weak_mean(s, weak_classes),
        "delta_weak_mean_iou": weak_mean(s, weak_classes) - weak_mean(base_summary, weak_classes),
        "picture_iou": float(s["iou"][pic]),
        "delta_picture_iou": float(s["iou"][pic] - base_summary["iou"][pic]),
        "picture_to_wall": picture_to_wall_from_conf(conf),
        "delta_picture_to_wall": picture_to_wall_from_conf(conf) - picture_to_wall_from_conf(base_summary["conf"]),
    }


def eval_pvd(args: argparse.Namespace, model, cfg, proposal_model: ProposalMLP, proposal_classes: list[int], weak_classes: list[int], variants: list[Variant], num_classes: int):
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    base_conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    variant_conf = {v.name: torch.zeros((num_classes, num_classes), dtype=torch.long) for v in variants}
    proposal_conf = torch.zeros((len(proposal_classes) + 1, len(proposal_classes) + 1), dtype=torch.long)
    seen = 0
    device = torch.device("cuda")
    class_tensor = torch.tensor(proposal_classes, device=device, dtype=torch.long)
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

            x, inv, region_count, gt_count, _, _ = proposal_features(
                feat, logits, target, grid, args.region_voxel_size, num_classes
            )
            prop_true = proposal_labels(gt_count, region_count, proposal_classes, args.positive_purity)
            prop_logits = proposal_model(x.to(device))
            prop_prob = torch.softmax(prop_logits, dim=1)
            prop_pred = prop_prob.argmax(dim=1)
            flat = prop_true.cpu() * prop_prob.shape[1] + prop_pred.cpu()
            proposal_conf += torch.bincount(flat, minlength=prop_prob.shape[1] ** 2).reshape(prop_prob.shape[1], prop_prob.shape[1])

            bg_idx = len(proposal_classes)
            hard_prob, hard_idx = prop_prob[:, :bg_idx].max(dim=1)
            bg_logit = prop_logits[:, bg_idx]
            hard_logit = prop_logits.gather(1, hard_idx[:, None]).squeeze(1)
            margin = (hard_logit - bg_logit).clamp_min(0.0)
            hard_class = class_tensor[hard_idx]

            for v in variants:
                selected = (hard_prob >= v.threshold) & (hard_idx != bg_idx) & (margin > 0)
                boost = torch.zeros((prop_prob.shape[0], num_classes), device=device, dtype=logits.dtype)
                if bool(selected.any()):
                    boost[selected, hard_class[selected]] = (v.beta * margin[selected]).to(logits.dtype)
                pred = (logits + boost[inv]).argmax(dim=1)
                update_confusion(variant_conf[v.name], pred.cpu(), target.cpu(), num_classes, -1)

            seen += 1
            if (batch_idx + 1) % 25 == 0:
                base = summarize_confusion(base_conf.numpy(), SCANNET20_CLASS_NAMES)
                print(f"[val] batch={batch_idx+1} base_mIoU={base['mIoU']:.4f}", flush=True)
    return {
        "seen_batches": seen,
        "base_conf": base_conf.numpy(),
        "variant_conf": {k: v.numpy() for k, v in variant_conf.items()},
        "proposal_conf": proposal_conf.numpy(),
    }


def write_results(args: argparse.Namespace, results: dict, variants: list[Variant], proposal_classes: list[int], weak_classes: list[int], train_counts: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_summary = summarize_confusion(results["base_conf"], SCANNET20_CLASS_NAMES)
    base_summary["conf"] = results["base_conf"]
    rows = [summary_row("base", results["base_conf"], base_summary, weak_classes)]
    for v in variants:
        rows.append(summary_row(v.name, results["variant_conf"][v.name], base_summary, weak_classes))
    rows_sorted = sorted(rows, key=lambda r: (r["variant"] == "base", -float(r["mIoU"])))
    variant_csv = args.output_dir / "proposal_verify_decoder_variants.csv"
    with variant_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    prefix = args.summary_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    prop_conf = results["proposal_conf"]
    prop_names = [SCANNET20_CLASS_NAMES[c] for c in proposal_classes] + ["background"]
    prop_rows = []
    for i, name in enumerate(prop_names):
        total = prop_conf[i].sum()
        prop_rows.append(
            {
                "proposal_label": name,
                "count": int(total),
                "acc": safe_ratio(float(prop_conf[i, i]), float(total)),
                "pred_background": safe_ratio(float(prop_conf[i, -1]), float(total)),
            }
        )
    prop_csv = args.output_dir / "proposal_classifier_confusion.csv"
    with prop_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prop_rows)

    pic = NAME_TO_ID["picture"]
    best_miou = max(rows, key=lambda r: float(r["mIoU"]))
    safe = [r for r in rows if float(r["mIoU"]) >= base_summary["mIoU"] - 0.002]
    best_safe_pic = max(safe, key=lambda r: float(r["picture_iou"])) if safe else None
    lines = []
    lines.append("# Proposal-then-Verify Decoder Pilot\n")
    lines.append("Lightweight PVD pilot: learn a hard-class proposal verifier on fine voxel proposals and fuse conservative proposal boosts into base decoder logits.\n")
    lines.append("## Setup\n")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Weight: `{args.weight}`")
    lines.append(f"- Region voxel size: `{args.region_voxel_size}`")
    lines.append(f"- Positive purity: `{args.positive_purity}`")
    lines.append(f"- Proposal classes: `{','.join(prop_names[:-1])}`")
    lines.append(f"- Train batches: `{train_counts['seen_batches']}`")
    lines.append(f"- Train proposal count: `{train_counts['train_proposals']}`")
    lines.append(f"- Seen val batches: `{results['seen_batches']}`")
    lines.append("")
    lines.append("## Headline\n")
    lines.append(
        f"- Base: mIoU={base_summary['mIoU']:.4f}, picture={base_summary['iou'][pic]:.4f}, "
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
    for i, r in enumerate(rows_sorted[:16], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {float(r['mIoU']):.4f} | {float(r['delta_mIoU']):+.4f} | "
            f"{float(r['weak_mean_iou']):.4f} | {float(r['delta_weak_mean_iou']):+.4f} | "
            f"{float(r['picture_iou']):.4f} | {float(r['delta_picture_iou']):+.4f} | "
            f"{float(r['picture_to_wall']):.4f} | {float(r['delta_picture_to_wall']):+.4f} |"
        )
    lines.append("")
    lines.append("## Proposal Classifier Diagnostics\n")
    lines.append("| label | count | proposal acc | pred background |")
    lines.append("|---|---:|---:|---:|")
    for r in prop_rows:
        lines.append(f"| `{r['proposal_label']}` | {int(r['count'])} | {float(r['acc']):.4f} | {float(r['pred_background']):.4f} |")
    lines.append("")
    lines.append("## Interpretation Gate\n")
    lines.append("- PVD go: picture improves by >=0.03 with mIoU >= base -0.002, or mIoU improves by >=0.003.")
    lines.append("- If proposal classifier is accurate but fusion is no-go, proposal selection exists but point fusion/object mask assignment is the bottleneck.")
    lines.append("- If proposal classifier itself is weak on picture, the proposal-first family needs stronger object proposal supervision or richer masks.")
    lines.append("")
    lines.append("## Files\n")
    lines.append(f"- Variant CSV: `{variant_csv.resolve()}`")
    lines.append(f"- Proposal classifier CSV: `{prop_csv.resolve()}`")
    md = prefix.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    (args.output_dir / "proposal_verify_decoder.md").write_text("\n".join(lines) + "\n")
    metadata = {
        "config": str(args.config),
        "weight": str(args.weight),
        "region_voxel_size": args.region_voxel_size,
        "positive_purity": args.positive_purity,
        "proposal_classes": prop_names,
        "train_counts": train_counts,
        "seen_val_batches": results["seen_batches"],
        "outputs": {"variant_csv": str(variant_csv), "proposal_csv": str(prop_csv), "summary_md": str(md)},
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
    proposal_classes = parse_names(args.proposal_classes)
    weak_classes = parse_names(args.weak_classes)
    thresholds = parse_float_list(args.thresholds)
    betas = parse_float_list(args.betas)
    variants = [Variant(t, b) for t in thresholds for b in betas]
    if args.dry_run:
        print(f"[dry-run] data_root={args.data_root}")
        print(f"[dry-run] proposal_classes={[SCANNET20_CLASS_NAMES[c] for c in proposal_classes]}")
        print(f"[dry-run] weak_classes={[SCANNET20_CLASS_NAMES[c] for c in weak_classes]}")
        print(f"[dry-run] variants={[v.name for v in variants]}")
        return
    cfg = load_config(args.repo_root / args.config)
    num_classes = int(cfg.data.num_classes)
    model = build_model(cfg, args.weight)
    x_train, y_train, seen_train = collect_train_bank(args, model, cfg, proposal_classes, num_classes)
    train_counts = {
        "seen_batches": seen_train,
        "train_proposals": int(x_train.shape[0]),
        "label_counts": torch.bincount(y_train, minlength=len(proposal_classes) + 1).tolist(),
    }
    print(f"[train-bank] final counts={train_counts}", flush=True)
    proposal_model = train_proposal_model(args, x_train, y_train, len(proposal_classes) + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": proposal_model.state_dict(),
            "proposal_classes": proposal_classes,
            "in_dim": int(x_train.shape[1]),
            "args": vars(args),
            "train_counts": train_counts,
        },
        args.output_dir / "proposal_mlp_last.pth",
    )
    results = eval_pvd(args, model, cfg, proposal_model, proposal_classes, weak_classes, variants, num_classes)
    write_results(args, results, variants, proposal_classes, weak_classes, train_counts)


if __name__ == "__main__":
    main()
