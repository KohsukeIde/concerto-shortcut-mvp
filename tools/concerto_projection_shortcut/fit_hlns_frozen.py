#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from posthoc_frozen_utils import (
    apply_transform,
    center_features,
    fit_linear_classifier,
    load_cache,
    make_nuisance,
    nuisance_energy,
    projection_transform,
    ridge_regression,
    save_editor_checkpoint,
    task_safe_nuisance_basis,
)


def group_slices(feature_dim: int, num_groups: int):
    base = feature_dim // num_groups
    rem = feature_dim % num_groups
    start = 0
    out = []
    for g in range(num_groups):
        width = base + (1 if g < rem else 0)
        end = start + width
        out.append((start, end))
        start = end
    return out


def main():
    parser = argparse.ArgumentParser(description="Fit Head-Localized Nuisance Surgery editor on frozen features.")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nuisance", default="height+xyz", choices=["height", "xyz", "height+xyz", "zbin8", "zbin16"])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--classifier-ridge", type=float, default=1e-2)
    parser.add_argument("--num-groups", type=int, default=16)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--task-tradeoff", type=float, default=0.25)
    args = parser.parse_args()

    train = load_cache(args.train_cache)
    val = load_cache(args.val_cache)
    x_train = train["feat"].float()
    x_val = val["feat"].float()
    y_train = train["label"].long()
    y_val = val["label"].long()
    n_train = make_nuisance(train["coord_norm"].float(), args.nuisance)

    x_train_c, x_val_c, mean = center_features(x_train, x_val)
    feature_dim = x_train.shape[1]
    num_classes = int(y_train[y_train >= 0].max().item()) + 1
    y_train_oh = torch.zeros(y_train.shape[0], num_classes, dtype=torch.float32)
    mask_train = y_train >= 0
    y_train_oh[mask_train, y_train[mask_train]] = 1.0

    slices = group_slices(feature_dim, args.num_groups)
    block_transform = torch.eye(feature_dim)
    group_stats = []
    for gid, (s, e) in enumerate(slices):
        xg = x_train_c[:, s:e]
        beta_task = ridge_regression(xg[mask_train], y_train_oh[mask_train], ridge=args.ridge)
        beta_nui = ridge_regression(xg, n_train, ridge=args.ridge)
        _, u_harm, _ = task_safe_nuisance_basis(beta_task, beta_nui)
        nuisance_score = float((beta_nui ** 2).sum().item())
        task_score = float((beta_task ** 2).sum().item())
        score = nuisance_score / (task_score + args.task_tradeoff)
        group_stats.append(
            {
                "group": gid,
                "start": s,
                "end": e,
                "score": score,
                "nuisance_score": nuisance_score,
                "task_score": task_score,
                "harm_rank": int(u_harm.shape[1]),
            }
        )

    ranked = sorted(group_stats, key=lambda item: item["score"], reverse=True)
    selected = {item["group"] for item in ranked[: args.topk]}
    for item in group_stats:
        gid, s, e = item["group"], item["start"], item["end"]
        if gid not in selected:
            continue
        xg = x_train_c[:, s:e]
        beta_task = ridge_regression(xg[mask_train], y_train_oh[mask_train], ridge=args.ridge)
        beta_nui = ridge_regression(xg, n_train, ridge=args.ridge)
        _, u_harm, _ = task_safe_nuisance_basis(beta_task, beta_nui)
        block_transform[s:e, s:e] = projection_transform(e - s, u_harm, gamma=args.gamma)

    edited_train = apply_transform(x_train, mean, block_transform)
    edited_val = apply_transform(x_val, mean, block_transform)

    metrics_before = fit_linear_classifier(x_train, y_train, x_val, y_val, ridge=args.classifier_ridge)
    metrics_after = fit_linear_classifier(edited_train, y_train, edited_val, y_val, ridge=args.classifier_ridge)

    metadata = {
        "method": "hlns_frozen",
        "nuisance": args.nuisance,
        "gamma": args.gamma,
        "ridge": args.ridge,
        "feature_dim": int(feature_dim),
        "num_groups": args.num_groups,
        "topk": args.topk,
        "task_tradeoff": args.task_tradeoff,
        "selected_groups": sorted(selected),
        "group_stats": ranked,
        "classifier_before": metrics_before,
        "classifier_after": metrics_after,
        "global_selected_energy_before_train": nuisance_energy(x_train, mean, block_transform.new_zeros((feature_dim, 0))),
    }

    # Approximate nuisance-energy drop with the overall removed component.
    delta_train = edited_train - x_train
    delta_val = edited_val - x_val
    metadata["edit_energy_train"] = float((delta_train.pow(2).sum(dim=1) / (x_train.pow(2).sum(dim=1).clamp_min(1e-6))).mean().item())
    metadata["edit_energy_val"] = float((delta_val.pow(2).sum(dim=1) / (x_val.pow(2).sum(dim=1).clamp_min(1e-6))).mean().item())

    print(metadata)
    save_editor_checkpoint(args.output, transform=block_transform, mean=mean, metadata=metadata)


if __name__ == "__main__":
    main()
