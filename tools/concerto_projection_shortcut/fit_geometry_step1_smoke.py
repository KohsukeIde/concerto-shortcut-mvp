#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from posthoc_frozen_utils import load_cache, ridge_regression

GEOM_KEY = "geom_local9"


def one_hot_labels(label: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = label >= 0
    y = torch.zeros(label.shape[0], num_classes, dtype=torch.float32)
    y[mask, label[mask].long()] = 1.0
    return y, mask


def standardize(train_x: torch.Tensor, val_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def compute_semseg_metrics(pred: torch.Tensor, label: torch.Tensor, num_classes: int) -> Dict[str, float]:
    valid = label >= 0
    pred = pred[valid].long()
    label = label[valid].long()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for gt, pd in zip(label.tolist(), pred.tolist()):
        conf[gt, pd] += 1
    tp = conf.diag().float()
    support = conf.sum(dim=1).float()
    pred_count = conf.sum(dim=0).float()
    union = support + pred_count - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1.0), torch.zeros_like(union))
    acc = torch.where(support > 0, tp / support.clamp_min(1.0), torch.zeros_like(support))
    present = support > 0
    miou = float(iou[present].mean().item()) if present.any() else 0.0
    macc = float(acc[present].mean().item()) if present.any() else 0.0
    allacc = float((pred == label).float().mean().item()) if label.numel() > 0 else 0.0
    return {
        "mIoU": miou,
        "mAcc": macc,
        "allAcc": allacc,
        "num_valid": int(label.numel()),
        "num_present_classes": int(present.sum().item()),
    }


def fit_linear_head(train_x: torch.Tensor, train_y_oh: torch.Tensor, train_mask: torch.Tensor, ridge: float) -> torch.Tensor:
    return ridge_regression(train_x[train_mask], train_y_oh[train_mask], ridge=ridge)


def eval_logits(logits: torch.Tensor, label: torch.Tensor) -> Dict[str, float]:
    valid = label >= 0
    if valid.any():
        num_classes = int(label[valid].max().item()) + 1
    else:
        num_classes = int(logits.shape[1])
    pred = logits.argmax(dim=1)
    return compute_semseg_metrics(pred, label, num_classes=num_classes)


def summarize_confusion_pairs(logits: torch.Tensor, label: torch.Tensor, topk: int = 20) -> List[Dict[str, int]]:
    valid = label >= 0
    logits = logits[valid]
    label = label[valid].long()
    pred = logits.argmax(dim=1)
    wrong = pred != label
    if not wrong.any():
        return []
    pair_count: Dict[Tuple[int, int], int] = {}
    for gt, pd in zip(label[wrong].tolist(), pred[wrong].tolist()):
        pair_count[(gt, pd)] = pair_count.get((gt, pd), 0) + 1
    items = sorted(pair_count.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:topk]
    return [{"gt": int(g), "pred": int(p), "count": int(c)} for (g, p), c in items]


def main():
    parser = argparse.ArgumentParser(description="Step 1 smoke: geometry-only / concat / global residual expert on frozen caches.")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--geometry-key", default=GEOM_KEY)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--residual-ridge", type=float, default=1e-2)
    parser.add_argument("--pass-threshold", type=float, default=0.003)
    args = parser.parse_args()

    train = load_cache(args.train_cache)
    val = load_cache(args.val_cache)
    if args.geometry_key not in train or args.geometry_key not in val:
        raise KeyError(f"geometry key {args.geometry_key!r} missing from cache")

    x_train = train["feat"].float()
    x_val = val["feat"].float()
    phi_train = train[args.geometry_key].float()
    phi_val = val[args.geometry_key].float()
    y_train = train["label"].long()
    y_val = val["label"].long()

    num_classes = int(y_train[y_train >= 0].max().item()) + 1
    y_train_oh, train_mask = one_hot_labels(y_train, num_classes)

    phi_train_std, phi_val_std, geom_mean, geom_std = standardize(phi_train, phi_val)

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    # Original baseline
    w0 = fit_linear_head(x_train, y_train_oh, train_mask, ridge=args.ridge)
    logits_orig_train = x_train @ w0
    logits_orig_val = x_val @ w0
    metrics_orig_train = eval_logits(logits_orig_train, y_train)
    metrics_orig_val = eval_logits(logits_orig_val, y_val)

    # Geometry-only
    wg = fit_linear_head(phi_train_std, y_train_oh, train_mask, ridge=args.ridge)
    logits_geo_train = phi_train_std @ wg
    logits_geo_val = phi_val_std @ wg
    metrics_geo_train = eval_logits(logits_geo_train, y_train)
    metrics_geo_val = eval_logits(logits_geo_val, y_val)

    # Concat
    xg_train = torch.cat([x_train, phi_train_std], dim=1)
    xg_val = torch.cat([x_val, phi_val_std], dim=1)
    wc = fit_linear_head(xg_train, y_train_oh, train_mask, ridge=args.ridge)
    logits_concat_train = xg_train @ wc
    logits_concat_val = xg_val @ wc
    metrics_concat_train = eval_logits(logits_concat_train, y_train)
    metrics_concat_val = eval_logits(logits_concat_val, y_val)

    # Residual expert with fixed base head
    residual_target = y_train_oh - logits_orig_train
    wa = ridge_regression(phi_train_std[train_mask], residual_target[train_mask], ridge=args.residual_ridge)
    logits_res_train = logits_orig_train + phi_train_std @ wa
    logits_res_val = logits_orig_val + phi_val_std @ wa
    metrics_res_train = eval_logits(logits_res_train, y_train)
    metrics_res_val = eval_logits(logits_res_val, y_val)

    rows = [
        {
            "method": "original",
            "train_mIoU": metrics_orig_train["mIoU"],
            "val_mIoU": metrics_orig_val["mIoU"],
            "train_mAcc": metrics_orig_train["mAcc"],
            "val_mAcc": metrics_orig_val["mAcc"],
            "train_allAcc": metrics_orig_train["allAcc"],
            "val_allAcc": metrics_orig_val["allAcc"],
            "delta_vs_original": 0.0,
        },
        {
            "method": "geometry_only",
            "train_mIoU": metrics_geo_train["mIoU"],
            "val_mIoU": metrics_geo_val["mIoU"],
            "train_mAcc": metrics_geo_train["mAcc"],
            "val_mAcc": metrics_geo_val["mAcc"],
            "train_allAcc": metrics_geo_train["allAcc"],
            "val_allAcc": metrics_geo_val["allAcc"],
            "delta_vs_original": metrics_geo_val["mIoU"] - metrics_orig_val["mIoU"],
        },
        {
            "method": "concat",
            "train_mIoU": metrics_concat_train["mIoU"],
            "val_mIoU": metrics_concat_val["mIoU"],
            "train_mAcc": metrics_concat_train["mAcc"],
            "val_mAcc": metrics_concat_val["mAcc"],
            "train_allAcc": metrics_concat_train["allAcc"],
            "val_allAcc": metrics_concat_val["allAcc"],
            "delta_vs_original": metrics_concat_val["mIoU"] - metrics_orig_val["mIoU"],
        },
        {
            "method": "residual_expert_global",
            "train_mIoU": metrics_res_train["mIoU"],
            "val_mIoU": metrics_res_val["mIoU"],
            "train_mAcc": metrics_res_train["mAcc"],
            "val_mAcc": metrics_res_val["mAcc"],
            "train_allAcc": metrics_res_train["allAcc"],
            "val_allAcc": metrics_res_val["allAcc"],
            "delta_vs_original": metrics_res_val["mIoU"] - metrics_orig_val["mIoU"],
        },
    ]

    pass_methods = [
        row["method"]
        for row in rows
        if row["method"] != "original" and row["delta_vs_original"] >= args.pass_threshold
    ]

    summary = {
        "geometry_key": args.geometry_key,
        "feature_dim": int(x_train.shape[1]),
        "geometry_dim": int(phi_train.shape[1]),
        "num_classes": int(num_classes),
        "ridge": args.ridge,
        "residual_ridge": args.residual_ridge,
        "pass_threshold": args.pass_threshold,
        "pass": len(pass_methods) > 0,
        "pass_methods": pass_methods,
        "top_confusion_pairs_original": summarize_confusion_pairs(logits_orig_val, y_val, topk=20),
        "top_confusion_pairs_residual": summarize_confusion_pairs(logits_res_val, y_val, topk=20),
        "rows": rows,
    }

    # Save machine-readable artifacts
    with (out / "step1_geometry_smoke.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out / "step1_geometry_smoke.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "train_mIoU", "val_mIoU", "train_mAcc", "val_mAcc", "train_allAcc", "val_allAcc", "delta_vs_original"],
        )
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        "# Step 1 geometry smoke",
        "",
        f"- geometry_key: `{args.geometry_key}`",
        f"- feature_dim: {x_train.shape[1]}",
        f"- geometry_dim: {phi_train.shape[1]}",
        f"- pass_threshold: {args.pass_threshold:.4f}",
        f"- pass: `{summary['pass']}`",
        f"- pass_methods: {', '.join(pass_methods) if pass_methods else '(none)' }",
        "",
        "| method | val mIoU | delta vs original | val mAcc | val allAcc |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['method']} | {row['val_mIoU']:.4f} | {row['delta_vs_original']:+.4f} | {row['val_mAcc']:.4f} | {row['val_allAcc']:.4f} |"
        )
    (out / "step1_geometry_smoke.md").write_text("\n".join(md_lines) + "\n")

    # Save useful state for Step 2
    torch.save(
        {
            "base_weight": w0.float().cpu(),
            "geom_weight": wg.float().cpu(),
            "concat_weight": wc.float().cpu(),
            "residual_weight": wa.float().cpu(),
            "geom_mean": geom_mean.float().cpu(),
            "geom_std": geom_std.float().cpu(),
            "geometry_key": args.geometry_key,
            "base_val_metrics": metrics_orig_val,
            "residual_val_metrics": metrics_res_val,
        },
        out / "step1_geometry_smoke_artifacts.pt",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
