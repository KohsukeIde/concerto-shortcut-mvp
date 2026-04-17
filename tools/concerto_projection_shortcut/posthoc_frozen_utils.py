#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def load_cache(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "feat" not in payload or "label" not in payload or "coord_norm" not in payload:
        raise ValueError(f"Malformed feature cache: {path}")
    return payload


def one_hot_labels(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    mask = label >= 0
    y = torch.zeros(label.shape[0], num_classes, dtype=torch.float32)
    valid = label[mask].long()
    y[mask, valid] = 1.0
    return y, mask


def make_nuisance(coord_norm: torch.Tensor, spec: str) -> torch.Tensor:
    z = coord_norm[:, 2:3]
    xyz = coord_norm[:, :3]
    if spec == "height":
        return z
    if spec == "xyz":
        return xyz
    if spec == "height+xyz":
        return torch.cat([z, xyz], dim=1)
    if spec.startswith("zbin"):
        bins = int(spec.replace("zbin", ""))
        q = torch.clamp(((z.squeeze(1) + 3.0) / 6.0 * bins).long(), 0, bins - 1)
        n = torch.zeros(z.shape[0], bins, dtype=torch.float32)
        n[torch.arange(z.shape[0]), q] = 1.0
        return n
    raise ValueError(f"Unsupported nuisance spec: {spec}")


def make_geometry(coord_norm: torch.Tensor, spec: str) -> torch.Tensor:
    x = coord_norm[:, 0:1]
    y = coord_norm[:, 1:2]
    z = coord_norm[:, 2:3]
    xyz = coord_norm[:, :3]
    if spec == "height":
        return z
    if spec == "xyz":
        return xyz
    if spec == "height+xyz":
        return torch.cat([z, xyz], dim=1)
    if spec == "coord9":
        return torch.cat([x, y, z, x * x, y * y, z * z, x * y, y * z, z * x], dim=1)
    if spec == "coord10":
        ones = torch.ones_like(z)
        return torch.cat([ones, x, y, z, x * x, y * y, z * z, x * y, y * z, z * x], dim=1)
    raise ValueError(f"Unsupported geometry spec: {spec}")


def geometry_dim(spec: str) -> int:
    return {
        "height": 1,
        "xyz": 3,
        "height+xyz": 4,
        "coord9": 9,
        "coord10": 10,
    }[spec]


def center_features(x_train: torch.Tensor, x_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    return x_train - mean, x_val - mean, mean.squeeze(0)


def ridge_regression(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    out_dtype = x.dtype
    x = x.double()
    y = y.double()
    d = x.shape[1]
    xtx = x.T @ x
    reg = ridge * torch.eye(d, dtype=x.dtype, device=x.device)
    xty = x.T @ y
    system = xtx + reg
    try:
        return torch.linalg.solve(system, xty).to(out_dtype)
    except torch._C._LinAlgError:
        for jitter in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            try:
                jitter_eye = jitter * torch.eye(d, dtype=x.dtype, device=x.device)
                return torch.linalg.solve(system + jitter_eye, xty).to(out_dtype)
            except torch._C._LinAlgError:
                continue
        return (torch.linalg.pinv(system) @ xty).to(out_dtype)


def orth_basis(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if mat.numel() == 0:
        return mat.new_zeros((mat.shape[0], 0))
    u, s, _ = torch.linalg.svd(mat, full_matrices=False)
    rank = int((s > eps).sum().item())
    return u[:, :rank]


def task_safe_nuisance_basis(beta_task: torch.Tensor, beta_nui: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u_task = orth_basis(beta_task)
    if u_task.numel() == 0:
        nuisance_resid = beta_nui
    else:
        nuisance_resid = beta_nui - u_task @ (u_task.T @ beta_nui)
    u_harm = orth_basis(nuisance_resid)
    return u_task, u_harm, nuisance_resid


def projection_transform(feature_dim: int, basis: torch.Tensor, gamma: float) -> torch.Tensor:
    eye = torch.eye(feature_dim, dtype=basis.dtype if basis.numel() else torch.float32)
    if basis.numel() == 0:
        return eye
    return eye - float(gamma) * (basis @ basis.T)


def apply_transform(x: torch.Tensor, mean: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    xc = x - mean
    y = xc @ transform.T
    return y + mean


def standardize_geometry(train_phi: torch.Tensor, val_phi: torch.Tensor):
    mean = train_phi.mean(dim=0, keepdim=True)
    std = train_phi.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_phi - mean) / std, (val_phi - mean) / std, mean.squeeze(0), std.squeeze(0)


def apply_residual_recycling(
    x: torch.Tensor,
    mean: torch.Tensor,
    transform: torch.Tensor,
    geom: torch.Tensor,
    harm_basis: torch.Tensor,
    geom_weight: torch.Tensor,
    recycle_scale: float,
) -> torch.Tensor:
    y = apply_transform(x, mean, transform)
    if harm_basis.numel() == 0 or geom_weight.numel() == 0:
        return y
    coeff = geom @ geom_weight
    return y + float(recycle_scale) * (coeff @ harm_basis.T)


def nuisance_energy(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> float:
    if basis.numel() == 0:
        return 0.0
    xc = x - mean
    proj = xc @ basis
    numer = (proj ** 2).sum(dim=1)
    denom = (xc ** 2).sum(dim=1).clamp_min(1e-6)
    return float((numer / denom).mean().item())


def fit_linear_classifier(train_x: torch.Tensor, train_label: torch.Tensor, val_x: torch.Tensor, val_label: torch.Tensor, ridge: float) -> Dict[str, float]:
    num_classes = int(train_label[train_label >= 0].max().item()) + 1
    y_train, mask_train = one_hot_labels(train_label, num_classes)
    train_w = ridge_regression(train_x[mask_train], y_train[mask_train], ridge=ridge)
    train_logits = train_x @ train_w
    val_logits = val_x @ train_w
    train_pred = train_logits.argmax(dim=1)
    val_pred = val_logits.argmax(dim=1)
    train_mask = train_label >= 0
    val_mask = val_label >= 0
    train_acc = (train_pred[train_mask] == train_label[train_mask]).float().mean().item()
    val_acc = (val_pred[val_mask] == val_label[val_mask]).float().mean().item()
    return {"train_acc": float(train_acc), "val_acc": float(val_acc), "num_classes": num_classes}


def save_editor_checkpoint(
    output_path: Path,
    transform: torch.Tensor,
    mean: torch.Tensor,
    metadata: dict,
    extra_state: Dict[str, torch.Tensor] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "transform": transform.float().cpu(),
        "mean": mean.float().cpu(),
        "bias": torch.zeros_like(mean).float().cpu(),
    }
    if extra_state:
        state_dict.update({key: value.float().cpu() for key, value in extra_state.items()})
    payload = {"state_dict": state_dict, "metadata": metadata}
    torch.save(payload, output_path)
    (output_path.with_suffix(output_path.suffix + ".json")).write_text(json.dumps(metadata, indent=2))
