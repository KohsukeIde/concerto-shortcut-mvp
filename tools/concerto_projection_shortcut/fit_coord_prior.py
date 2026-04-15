#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cfg(repo_root: Path, config_name: str):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pointcept.utils.config import Config

    return Config.fromfile(str(repo_root / "configs" / "concerto" / f"{config_name}.py"))


def move_batch_to_cuda(batch):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.cuda(non_blocking=True)
    return batch


def load_weight(model, weight_path: Path) -> None:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    bare_backbone = not any(
        key.startswith(("student.", "teacher.", "enc2d_", "patch_proj"))
        for key in cleaned
    )
    if bare_backbone and hasattr(model, "student") and "backbone" in model.student:
        student_info = model.student.backbone.load_state_dict(cleaned, strict=False)
        teacher_info = model.teacher.backbone.load_state_dict(cleaned, strict=False)
        print(
            "[info] load bare backbone",
            f"student_missing={len(student_info.missing_keys)}",
            f"student_unexpected={len(student_info.unexpected_keys)}",
            f"teacher_missing={len(teacher_info.missing_keys)}",
            f"teacher_unexpected={len(teacher_info.unexpected_keys)}",
        )
    else:
        info = model.load_state_dict(cleaned, strict=False)
        print(
            "[info] load_weight",
            f"missing={len(info.missing_keys)}",
            f"unexpected={len(info.unexpected_keys)}",
        )


def build_loader(cfg, split: str, data_root: Path | None, batch_size: int, num_worker: int):
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn

    cfg.data.train.split = [split]
    cfg.data.train.loop = 1
    if data_root is not None:
        cfg.data.train.data_root = str(data_root.resolve())
    dataset = build_dataset(cfg.data.train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        collate_fn=lambda batch: point_collate_fn(batch, mix_prob=0),
    )


def collect_cache(
    model,
    loader,
    split: str,
    max_batches: int,
    max_rows_per_batch: int,
    cache_path: Path,
    dry_run: bool,
    force: bool,
):
    if cache_path.exists() and not force and not dry_run:
        print(f"[cache] reuse {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    coords = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = move_batch_to_cuda(batch)
            extracted = model.extract_enc2d_coord_target(
                batch, max_rows=max_rows_per_batch
            )
            coord = extracted["coord"].detach().float().cpu()
            target = extracted["target"].detach().half().cpu()
            if coord.numel() == 0:
                continue
            coords.append(coord)
            targets.append(target)
            if (batch_idx + 1) % 50 == 0:
                rows = sum(item.shape[0] for item in coords)
                print(f"[cache] {split} batches={batch_idx + 1} rows={rows}")

    if not coords:
        raise RuntimeError(f"No rows collected for split={split}.")
    payload = dict(
        split=split,
        coord=torch.cat(coords, dim=0),
        target=torch.cat(targets, dim=0),
        target_shifted=bool(model.enc2d_cos_shift),
        coord_normalize=bool(model.shortcut_probe["coord_normalize"]),
        target_dim=int(targets[0].shape[1]),
    )
    print(
        f"[cache] {split} rows={payload['coord'].shape[0]} "
        f"target_dim={payload['target_dim']} dry_run={dry_run}"
    )
    if not dry_run:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
        print(f"[cache] wrote {cache_path}")
    return payload


def shifted_pred(pred: torch.Tensor, target_shifted: bool) -> torch.Tensor:
    if target_shifted:
        return pred - pred.mean(dim=-1, keepdim=True)
    return pred


def iter_minibatches(num_rows: int, batch_size: int, shuffle: bool):
    if shuffle:
        order = torch.randperm(num_rows)
    else:
        order = torch.arange(num_rows)
    for start in range(0, num_rows, batch_size):
        yield order[start : start + batch_size]


def evaluate_prior(prior, coord, target, target_shifted: bool, batch_size: int):
    prior.eval()
    losses = []
    target_energy = []
    residual_norm = []
    with torch.inference_mode():
        for idx in iter_minibatches(coord.shape[0], batch_size, shuffle=False):
            c = coord[idx].cuda(non_blocking=True)
            t = target[idx].cuda(non_blocking=True).float()
            pred = shifted_pred(prior(c), target_shifted)
            cos = F.cosine_similarity(pred, t, dim=1, eps=1e-6)
            u = F.normalize(pred, dim=1, eps=1e-6)
            projection = (t * u).sum(dim=1, keepdim=True) * u
            losses.append((1 - cos).detach().cpu())
            target_energy.append(
                (
                    projection.pow(2).sum(dim=1)
                    / t.pow(2).sum(dim=1).clamp_min(1e-6)
                )
                .detach()
                .cpu()
            )
            residual = t - projection
            residual_norm.append(
                (
                    residual.norm(dim=1)
                    / t.norm(dim=1).clamp_min(1e-6)
                )
                .detach()
                .cpu()
            )
    return dict(
        cosine_loss=float(torch.cat(losses).mean().item()),
        target_energy=float(torch.cat(target_energy).mean().item()),
        residual_norm=float(torch.cat(residual_norm).mean().item()),
    )


def fit_prior(
    arch: str,
    train_cache,
    val_cache,
    output_dir: Path,
    hidden_channels: int,
    epochs: int,
    batch_size: int,
    lr: float,
    dry_run: bool,
):
    from pointcept.models.concerto.concerto_v1m1_base import CoordPrior

    target_dim = int(train_cache["target_dim"])
    target_shifted = bool(train_cache["target_shifted"])
    prior = CoordPrior(arch, target_dim, hidden_channels=hidden_channels).cuda()
    optimizer = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=1e-4)
    train_coord = train_cache["coord"].float()
    train_target = train_cache["target"]
    val_coord = val_cache["coord"].float()
    val_target = val_cache["target"]
    for epoch in range(epochs):
        prior.train()
        epoch_losses = []
        for idx in iter_minibatches(train_coord.shape[0], batch_size, shuffle=True):
            coord = train_coord[idx].cuda(non_blocking=True)
            target = train_target[idx].cuda(non_blocking=True).float()
            pred = shifted_pred(prior(coord), target_shifted)
            loss = (1 - F.cosine_similarity(pred, target, dim=1, eps=1e-6)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        print(
            f"[fit] arch={arch} epoch={epoch + 1}/{epochs} "
            f"train_cosine_loss={sum(epoch_losses) / max(len(epoch_losses), 1):.6f}"
        )
    metrics = evaluate_prior(
        prior, val_coord, val_target, target_shifted=target_shifted, batch_size=batch_size
    )
    metrics["arch"] = arch
    metrics["target_dim"] = target_dim
    metrics["target_shifted"] = target_shifted
    metrics["coord_normalize"] = bool(train_cache["coord_normalize"])
    metrics["hidden_channels"] = int(hidden_channels)
    print(f"[fit] arch={arch} val_metrics={json.dumps(metrics, sort_keys=True)}")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                arch=arch,
                target_dim=target_dim,
                hidden_channels=int(hidden_channels),
                state_dict=prior.cpu().state_dict(),
                metadata=metrics,
            ),
            output_dir / "model_last.pth",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit coordinate prior for Concerto enc2d targets.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="pretrain-concerto-v1m1-0-arkit-full-continue")
    parser.add_argument("--weight", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--max-train-batches", type=int, default=4096)
    parser.add_argument("--max-val-batches", type=int, default=512)
    parser.add_argument("--max-rows-per-batch", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--prior-epochs", type=int, default=20)
    parser.add_argument("--prior-batch-size", type=int, default=8192)
    parser.add_argument("--prior-hidden-channels", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlp-min-improvement", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    repo_root = args.repo_root.resolve()
    if args.output_root is None:
        args.output_root = repo_root / "data" / "runs" / "projres_v1" / "priors"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pointcept.models.builder import build_model

    cfg = load_cfg(repo_root, args.config)
    model = build_model(cfg.model).cuda()
    weight_path = args.weight
    if weight_path is None:
        default_weight = repo_root / "data" / "weights" / "concerto" / "concerto_base_origin.pth"
        weight_path = default_weight if default_weight.exists() else None
    if weight_path is not None:
        load_weight(model, weight_path.resolve())
    train_loader = build_loader(
        cfg,
        "Training",
        args.data_root,
        args.extract_batch_size,
        args.num_worker,
    )
    val_loader = build_loader(
        cfg,
        "Validation",
        args.data_root,
        args.extract_batch_size,
        args.num_worker,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    train_cache = collect_cache(
        model,
        train_loader,
        "Training",
        args.max_train_batches,
        args.max_rows_per_batch,
        args.output_root / "cache" / "train.pt",
        args.dry_run,
        args.force_cache,
    )
    val_cache = collect_cache(
        model,
        val_loader,
        "Validation",
        args.max_val_batches,
        args.max_rows_per_batch,
        args.output_root / "cache" / "val.pt",
        args.dry_run,
        args.force_cache,
    )
    del model
    torch.cuda.empty_cache()

    metrics = {}
    for arch in ("linear", "mlp"):
        metrics[arch] = fit_prior(
            arch,
            train_cache,
            val_cache,
            args.output_root / arch,
            hidden_channels=args.prior_hidden_channels,
            epochs=1 if args.dry_run else args.prior_epochs,
            batch_size=args.prior_batch_size,
            lr=args.lr,
            dry_run=args.dry_run,
        )
    selected = "linear"
    improvement = metrics["linear"]["cosine_loss"] - metrics["mlp"]["cosine_loss"]
    if improvement >= args.mlp_min_improvement:
        selected = "mlp"
    summary = dict(
        selected=selected,
        selected_path=str((args.output_root / selected / "model_last.pth").resolve()),
        mlp_improvement=float(improvement),
        metrics=metrics,
    )
    print("[select]", json.dumps(summary, indent=2, sort_keys=True))
    if not args.dry_run:
        (args.output_root / "selected_prior.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
