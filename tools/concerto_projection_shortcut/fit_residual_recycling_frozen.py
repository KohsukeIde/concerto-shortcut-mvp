#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from posthoc_frozen_utils import (
    apply_residual_recycling,
    apply_transform,
    center_features,
    fit_linear_classifier,
    geometry_dim,
    load_cache,
    make_geometry,
    make_nuisance,
    nuisance_energy,
    projection_transform,
    ridge_regression,
    save_editor_checkpoint,
    standardize_geometry,
    task_safe_nuisance_basis,
)


def pad_columns(mat: torch.Tensor, width: int) -> torch.Tensor:
    if mat.shape[1] > width:
        return mat[:, :width]
    if mat.shape[1] == width:
        return mat
    pad = mat.new_zeros((mat.shape[0], width - mat.shape[1]))
    return torch.cat([mat, pad], dim=1)


def residual_coefficients(residual: torch.Tensor, class_effect: torch.Tensor, ridge: float) -> torch.Tensor:
    rank = class_effect.shape[0]
    if rank == 0:
        return residual.new_zeros((residual.shape[0], 0))
    gram = class_effect @ class_effect.T
    reg = ridge * torch.eye(rank, dtype=class_effect.dtype)
    return residual @ class_effect.T @ torch.linalg.inv(gram + reg)


def linear_logits(x: torch.Tensor, label: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_classes = int(label[label >= 0].max().item()) + 1
    y = torch.zeros(label.shape[0], num_classes, dtype=torch.float32)
    mask = label >= 0
    y[mask, label[mask].long()] = 1.0
    weight = ridge_regression(x[mask], y[mask], ridge=ridge)
    return x @ weight, weight, y


def main():
    parser = argparse.ArgumentParser(description="Fit residual-recycling frozen-feature editor.")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nuisance", default="height+xyz", choices=["height", "xyz", "height+xyz", "zbin8", "zbin16"])
    parser.add_argument("--geometry", default="coord9", choices=["height", "xyz", "height+xyz", "coord9", "coord10"])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--recycle-scale", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--classifier-ridge", type=float, default=1e-2)
    parser.add_argument("--recycle-ridge", type=float, default=1e-2)
    parser.add_argument("--coeff-ridge", type=float, default=1e-2)
    parser.add_argument("--max-rank", type=int, default=8)
    args = parser.parse_args()

    train = load_cache(args.train_cache)
    val = load_cache(args.val_cache)

    x_train = train["feat"].float()
    x_val = val["feat"].float()
    y_train = train["label"].long()
    y_val = val["label"].long()
    mask_train = y_train >= 0

    n_train = make_nuisance(train["coord_norm"].float(), args.nuisance)
    phi_train = make_geometry(train["coord_norm"].float(), args.geometry)
    phi_val = make_geometry(val["coord_norm"].float(), args.geometry)
    phi_train, phi_val, geom_mean, geom_std = standardize_geometry(phi_train, phi_val)

    x_train_c, x_val_c, mean = center_features(x_train, x_val)
    logits_orig, weight_orig, y_train_oh = linear_logits(x_train, y_train, args.classifier_ridge)
    num_classes = y_train_oh.shape[1]

    beta_task = ridge_regression(x_train_c[mask_train], y_train_oh[mask_train], ridge=args.ridge)
    beta_nui = ridge_regression(x_train_c, n_train, ridge=args.ridge)
    u_task, u_harm, nuisance_resid = task_safe_nuisance_basis(beta_task, beta_nui)
    if u_harm.shape[1] > args.max_rank:
        u_harm = u_harm[:, : args.max_rank]
    transform = projection_transform(x_train.shape[1], u_harm, gamma=args.gamma)

    deleted_train = apply_transform(x_train, mean, transform)
    deleted_val = apply_transform(x_val, mean, transform)
    logits_deleted, weight_deleted, _ = linear_logits(deleted_train, y_train, args.classifier_ridge)
    residual = y_train_oh - logits_deleted
    class_effect = u_harm.T @ weight_deleted
    coeff_target = residual_coefficients(residual[mask_train], class_effect, ridge=args.coeff_ridge)
    geom_weight = ridge_regression(phi_train[mask_train], coeff_target, ridge=args.recycle_ridge)

    recycled_train = apply_residual_recycling(
        x_train,
        mean,
        transform,
        phi_train,
        u_harm,
        geom_weight,
        recycle_scale=args.recycle_scale,
    )
    recycled_val = apply_residual_recycling(
        x_val,
        mean,
        transform,
        phi_val,
        u_harm,
        geom_weight,
        recycle_scale=args.recycle_scale,
    )

    metrics_before = fit_linear_classifier(x_train, y_train, x_val, y_val, ridge=args.classifier_ridge)
    metrics_deleted = fit_linear_classifier(deleted_train, y_train, deleted_val, y_val, ridge=args.classifier_ridge)
    metrics_recycled = fit_linear_classifier(recycled_train, y_train, recycled_val, y_val, ridge=args.classifier_ridge)
    train_resid_mse = float((residual[mask_train] ** 2).mean().item())
    recycled_logits = recycled_train @ weight_deleted
    recycled_resid_mse = float(((y_train_oh[mask_train] - recycled_logits[mask_train]) ** 2).mean().item())

    padded_basis = pad_columns(u_harm, args.max_rank)
    padded_weight = pad_columns(geom_weight, args.max_rank)
    metadata = {
        "method": "residual_recycling_frozen",
        "nuisance": args.nuisance,
        "geometry": args.geometry,
        "gamma": args.gamma,
        "recycle_scale": args.recycle_scale,
        "ridge": args.ridge,
        "classifier_ridge": args.classifier_ridge,
        "recycle_ridge": args.recycle_ridge,
        "coeff_ridge": args.coeff_ridge,
        "feature_dim": int(x_train.shape[1]),
        "geometry_dim": geometry_dim(args.geometry),
        "num_classes": int(num_classes),
        "task_rank": int(u_task.shape[1]),
        "harm_rank": int(u_harm.shape[1]),
        "max_rank": int(args.max_rank),
        "nuisance_energy_before_train": nuisance_energy(x_train, mean, u_harm),
        "nuisance_energy_deleted_train": nuisance_energy(deleted_train, mean, u_harm),
        "nuisance_energy_recycled_train": nuisance_energy(recycled_train, mean, u_harm),
        "nuisance_energy_before_val": nuisance_energy(x_val, mean, u_harm),
        "nuisance_energy_deleted_val": nuisance_energy(deleted_val, mean, u_harm),
        "nuisance_energy_recycled_val": nuisance_energy(recycled_val, mean, u_harm),
        "train_residual_mse_deleted": train_resid_mse,
        "train_residual_mse_recycled_with_deleted_head": recycled_resid_mse,
        "classifier_before": metrics_before,
        "classifier_deleted": metrics_deleted,
        "classifier_recycled": metrics_recycled,
        "orig_train_logit_norm": float(logits_orig[mask_train].norm(dim=1).mean().item()),
        "deleted_train_logit_norm": float(logits_deleted[mask_train].norm(dim=1).mean().item()),
    }
    print(metadata)
    save_editor_checkpoint(
        args.output,
        transform=transform,
        mean=mean,
        metadata=metadata,
        extra_state={
            "harm_basis": padded_basis,
            "geom_weight": padded_weight,
            "geom_mean": geom_mean,
            "geom_std": geom_std,
        },
    )


if __name__ == "__main__":
    main()
