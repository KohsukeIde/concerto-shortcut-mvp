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


def main():
    parser = argparse.ArgumentParser(description="Fit SPLICE-3D frozen-feature editor.")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nuisance", default="height+xyz", choices=["height", "xyz", "height+xyz", "zbin8", "zbin16"])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--classifier-ridge", type=float, default=1e-2)
    args = parser.parse_args()

    train = load_cache(args.train_cache)
    val = load_cache(args.val_cache)

    x_train = train["feat"].float()
    x_val = val["feat"].float()
    y_train = train["label"].long()
    y_val = val["label"].long()
    n_train = make_nuisance(train["coord_norm"].float(), args.nuisance)
    n_val = make_nuisance(val["coord_norm"].float(), args.nuisance)

    x_train_c, x_val_c, mean = center_features(x_train, x_val)
    num_classes = int(y_train[y_train >= 0].max().item()) + 1
    y_train_oh = torch.zeros(y_train.shape[0], num_classes, dtype=torch.float32)
    mask_train = y_train >= 0
    y_train_oh[mask_train, y_train[mask_train]] = 1.0

    beta_task = ridge_regression(x_train_c[mask_train], y_train_oh[mask_train], ridge=args.ridge)
    beta_nui = ridge_regression(x_train_c, n_train, ridge=args.ridge)
    u_task, u_harm, nuisance_resid = task_safe_nuisance_basis(beta_task, beta_nui)
    transform = projection_transform(x_train.shape[1], u_harm, gamma=args.gamma)

    edited_train = apply_transform(x_train, mean, transform)
    edited_val = apply_transform(x_val, mean, transform)

    metrics_before = fit_linear_classifier(x_train, y_train, x_val, y_val, ridge=args.classifier_ridge)
    metrics_after = fit_linear_classifier(edited_train, y_train, edited_val, y_val, ridge=args.classifier_ridge)

    metadata = {
        "method": "splice3d_frozen",
        "nuisance": args.nuisance,
        "gamma": args.gamma,
        "ridge": args.ridge,
        "feature_dim": int(x_train.shape[1]),
        "task_rank": int(u_task.shape[1]),
        "harm_rank": int(u_harm.shape[1]),
        "nuisance_energy_before_train": nuisance_energy(x_train, mean, u_harm),
        "nuisance_energy_after_train": nuisance_energy(edited_train, mean, u_harm),
        "nuisance_energy_before_val": nuisance_energy(x_val, mean, u_harm),
        "nuisance_energy_after_val": nuisance_energy(edited_val, mean, u_harm),
        "classifier_before": metrics_before,
        "classifier_after": metrics_after,
    }

    print(metadata)
    save_editor_checkpoint(args.output, transform=transform, mean=mean, metadata=metadata)


if __name__ == "__main__":
    main()
