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
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
    parse_pairs,
)


@dataclass
class PointCache:
    feat: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    class_counts: dict[str, int]
    seen_batches: int


class CoDAAdapter(nn.Module):
    """Small residual adapter from frozen decoder point features to 20-way logits."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
        dropout: float,
        use_class_bias: bool,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("feat_mean", feat_mean.float())
        self.register_buffer("feat_std", feat_std.float().clamp_min(1e-6))
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        if use_class_bias:
            self.class_bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter("class_bias", None)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        feat_z = (feat.float() - self.feat_mean.to(feat.device)) / self.feat_std.to(feat.device)
        delta = self.net(feat_z)
        if self.class_bias is not None:
            delta = delta + self.class_bias.to(delta.device)
        return delta


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CoDA: confusion-aware lightweight decoder adapter for the frozen "
            "concerto_base_origin ScanNet decoder-probe checkpoint. The frozen "
            "decoder point feature h and base logits z0 are cached; only a tiny "
            "residual adapter A(h) is trained, producing z = z0 + A(h)."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall",
    )
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--heldout-mod", type=int, default=5)
    parser.add_argument("--heldout-remainder", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=420)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-adapter-train-points", type=int, default=1200000)
    parser.add_argument("--max-heldout-points", type=int, default=1200000)
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sample-batch-size", type=int, default=8192)
    parser.add_argument("--weak-class-weight", type=float, default=3.0)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    parser.add_argument("--residual-l2", type=float, default=0.01)
    parser.add_argument("--train-tau", type=float, default=1.0)
    parser.add_argument("--eval-lambdas", default="0.1,0.2,0.5,1.0")
    parser.add_argument("--eval-taus", default="0.1,0.2,0.5,1.0")
    parser.add_argument("--center-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-class-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selection", default="miou_then_picture", choices=["miou_then_picture", "picture_safe"])
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Load a saved CoDA adapter and run validation evaluation without retraining.",
    )
    parser.add_argument(
        "--eval-all-val",
        action="store_true",
        help="Evaluate every lambda/tau variant on ScanNet val. Useful for checking heldout selection failure.",
    )
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_names(text: str) -> list[int]:
    ids = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {name}")
        ids.append(NAME_TO_ID[name])
    if not ids:
        raise ValueError("no classes provided")
    return ids


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_model(cfg, weight_path: Path):
    from pointcept.models.builder import build_model

    model = build_model(cfg.model).cuda().eval()
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    info = model.load_state_dict(cleaned, strict=False)
    print(
        f"[load] weight={weight_path} missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}",
        flush=True,
    )
    if info.missing_keys:
        print(f"[load] first missing={info.missing_keys[:8]}", flush=True)
    if info.unexpected_keys:
        print(f"[load] first unexpected={info.unexpected_keys[:8]}", flush=True)
    return model


def build_adapter_from_state(state_dict: dict[str, torch.Tensor], dropout: float) -> CoDAAdapter:
    feat_mean = state_dict["feat_mean"]
    feat_std = state_dict["feat_std"]
    hidden_dim, feat_dim = state_dict["net.0.weight"].shape
    num_classes = state_dict["net.8.bias"].numel()
    use_class_bias = "class_bias" in state_dict
    adapter = CoDAAdapter(
        feat_dim=feat_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        feat_mean=feat_mean,
        feat_std=feat_std,
        dropout=dropout,
        use_class_bias=use_class_bias,
    )
    adapter.load_state_dict(state_dict, strict=True)
    return adapter


def move_to_cuda(input_dict: dict) -> dict:
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.cuda(non_blocking=True)
    return input_dict


@torch.no_grad()
def forward_features(model, batch: dict):
    out = model(batch, return_point=True)
    point = out["point"]
    feat = point.feat.float()
    logits = out["seg_logits"].float()
    labels = batch["segment"].long()
    if feat.shape[0] != logits.shape[0] or feat.shape[0] != labels.shape[0]:
        raise RuntimeError(f"shape mismatch feat={feat.shape} logits={logits.shape} labels={labels.shape}")
    return feat, logits, labels, batch


def eval_tensors(feat: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, batch: dict):
    inverse = batch.get("inverse")
    origin_segment = batch.get("origin_segment")
    if inverse is not None and origin_segment is not None:
        return feat[inverse], logits[inverse], origin_segment.long()
    return feat, logits, labels


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    valid = target != ignore_index
    pred = pred[valid].long()
    target = target[valid].long()
    in_range = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[in_range]
    target = target[in_range]
    flat = target * num_classes + pred
    confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def summarize_confusion(conf: np.ndarray, names: list[str]):
    inter = np.diag(conf).astype(np.float64)
    target_sum = conf.sum(axis=1).astype(np.float64)
    pred_sum = conf.sum(axis=0).astype(np.float64)
    union = target_sum + pred_sum - inter
    iou = inter / (union + 1e-10)
    acc = inter / (target_sum + 1e-10)
    return {
        "mIoU": float(iou.mean()),
        "mAcc": float(acc.mean()),
        "allAcc": float(inter.sum() / (target_sum.sum() + 1e-10)),
        "iou": iou,
        "acc": acc,
        "target_sum": target_sum,
        "pred_sum": pred_sum,
        "intersection": inter,
        "union": union,
        "names": names,
    }


def append_to_cache(
    cache: dict,
    feat: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_points: int,
    max_per_class: int,
    num_classes: int,
) -> None:
    total = cache["total"]
    if total >= max_points:
        return
    valid = (labels >= 0) & (labels < num_classes)
    keep_indices = []
    for cls in range(num_classes):
        room_cls = max_per_class - cache["class_counts"][cls]
        if room_cls <= 0:
            continue
        idx = ((labels == cls) & valid).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        room_total = max_points - total - sum(x.numel() for x in keep_indices)
        if room_total <= 0:
            break
        cap = min(room_cls, room_total)
        if idx.numel() > cap:
            idx = idx[:cap]
        keep_indices.append(idx)
        cache["class_counts"][cls] += int(idx.numel())
    if keep_indices:
        keep = torch.cat(keep_indices, dim=0)
        cache["feat"].append(feat[keep].detach().cpu())
        cache["logits"].append(logits[keep].detach().cpu())
        cache["labels"].append(labels[keep].detach().cpu())
        cache["total"] += int(keep.numel())


def finalize_cache(raw: dict, num_classes: int, seen_batches: int) -> PointCache:
    if not raw["labels"]:
        raise RuntimeError("empty cache")
    return PointCache(
        feat=torch.cat(raw["feat"], dim=0),
        logits=torch.cat(raw["logits"], dim=0),
        labels=torch.cat(raw["labels"], dim=0).long(),
        class_counts={SCANNET20_CLASS_NAMES[k]: int(v) for k, v in raw["class_counts"].items()},
        seen_batches=seen_batches,
    )


def collect_train_heldout(args: argparse.Namespace, model, cfg, num_classes: int) -> tuple[PointCache, PointCache]:
    train_raw = {"feat": [], "logits": [], "labels": [], "class_counts": {i: 0 for i in range(num_classes)}, "total": 0}
    held_raw = {"feat": [], "logits": [], "labels": [], "class_counts": {i: 0 for i in range(num_classes)}, "total": 0}
    loader = build_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
                break
            if train_raw["total"] >= args.max_adapter_train_points and held_raw["total"] >= args.max_heldout_points:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, _ = forward_features(model, batch)
            is_heldout = (batch_idx % args.heldout_mod) == args.heldout_remainder
            if is_heldout:
                append_to_cache(held_raw, feat, logits, labels, args.max_heldout_points, args.max_per_class, num_classes)
            else:
                append_to_cache(train_raw, feat, logits, labels, args.max_adapter_train_points, args.max_per_class, num_classes)
            seen += 1
            if (batch_idx + 1) % 25 == 0:
                print(
                    f"[collect] batch={batch_idx + 1} train={train_raw['total']} heldout={held_raw['total']}",
                    flush=True,
                )
    train = finalize_cache(train_raw, num_classes, seen)
    heldout = finalize_cache(held_raw, num_classes, seen)
    print(f"[collect] done train={train.labels.numel()} heldout={heldout.labels.numel()} seen={seen}", flush=True)
    return train, heldout


def class_weights(labels: torch.Tensor, weak_classes: list[int], num_classes: int, weak_weight: float) -> torch.Tensor:
    weights = torch.ones(num_classes, dtype=torch.float32)
    for cls in weak_classes:
        weights[cls] = weak_weight
    return weights[labels.cpu()].float()


def center_and_clip_delta(delta: torch.Tensor, center_delta: bool, tau: float) -> torch.Tensor:
    if center_delta:
        delta = delta - delta.mean(dim=1, keepdim=True)
    if tau > 0 and tau < 900:
        delta = delta.clamp(min=-tau, max=tau)
    return delta


def pairwise_aux_loss(final_logits: torch.Tensor, labels: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    losses = []
    for a, b in pairs:
        mask = (labels == a) | (labels == b)
        if not mask.any():
            continue
        pair_logits = final_logits[mask][:, [a, b]]
        pair_target = (labels[mask] == b).long()
        losses.append(F.cross_entropy(pair_logits, pair_target))
    if not losses:
        return final_logits.sum() * 0.0
    return torch.stack(losses).mean()


def train_adapter(
    args: argparse.Namespace,
    cache: PointCache,
    pairs: list[tuple[int, int]],
    weak_classes: list[int],
    num_classes: int,
):
    device = torch.device("cuda")
    feat = cache.feat.to(device)
    logits = cache.logits.to(device)
    labels = cache.labels.to(device)
    weights = class_weights(labels.detach().cpu(), weak_classes, num_classes, args.weak_class_weight).to(device)
    feat_mean = feat.mean(dim=0, keepdim=True).detach().cpu()
    feat_std = feat.std(dim=0, keepdim=True).clamp_min(1e-6).detach().cpu()
    adapter = CoDAAdapter(
        feat_dim=feat.shape[1],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        feat_mean=feat_mean,
        feat_std=feat_std,
        dropout=args.dropout,
        use_class_bias=args.use_class_bias,
    ).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = labels.numel()
    history = []
    for step in range(args.steps):
        idx = torch.randint(0, n, (min(args.sample_batch_size, n),), device=device)
        feat_b = feat[idx]
        logits_b = logits[idx]
        labels_b = labels[idx]
        weights_b = weights[idx]
        delta = adapter(feat_b)
        delta = center_and_clip_delta(delta, args.center_delta, args.train_tau)
        final = logits_b + delta
        ce = F.cross_entropy(final, labels_b, reduction="none")
        loss_ce = (ce * weights_b).sum() / weights_b.sum().clamp_min(1e-6)
        loss_pair = pairwise_aux_loss(final, labels_b, pairs)
        loss = loss_ce + args.pair_weight * loss_pair
        if args.kl_weight > 0:
            t = args.kl_temperature
            base_prob = (logits_b / t).softmax(dim=1)
            log_prob = (final / t).log_softmax(dim=1)
            loss_kl = F.kl_div(log_prob, base_prob, reduction="batchmean") * (t * t)
            loss = loss + args.kl_weight * loss_kl
        else:
            loss_kl = final.sum() * 0.0
        if args.residual_l2 > 0:
            loss_l2 = delta.pow(2).mean()
            loss = loss + args.residual_l2 * loss_l2
        else:
            loss_l2 = final.sum() * 0.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
        opt.step()
        if (step + 1) % max(args.steps // 10, 1) == 0 or step == 0:
            with torch.no_grad():
                base_pred = logits_b.argmax(dim=1)
                pred = final.argmax(dim=1)
                row = {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "ce": float(loss_ce.item()),
                    "pair": float(loss_pair.item()),
                    "kl": float(loss_kl.item()),
                    "delta_l2": float(loss_l2.item()),
                    "base_acc": float((base_pred == labels_b).float().mean().item()),
                    "adapter_acc": float((pred == labels_b).float().mean().item()),
                    "delta_rms": float(delta.pow(2).mean().sqrt().item()),
                }
            history.append(row)
            print(
                f"[train] step={row['step']} loss={row['loss']:.4f} ce={row['ce']:.4f} "
                f"pair={row['pair']:.4f} kl={row['kl']:.4f} "
                f"base={row['base_acc']:.4f} acc={row['adapter_acc']:.4f} delta={row['delta_rms']:.4f}",
                flush=True,
            )
    return adapter, {
        "history": history,
        "train_points": int(labels.numel()),
        "feat_dim": int(feat.shape[1]),
        "num_trainable": int(sum(p.numel() for p in adapter.parameters() if p.requires_grad)),
    }


def corrected_logits(adapter: CoDAAdapter, feat: torch.Tensor, logits: torch.Tensor, lam: float, tau: float, center_delta: bool):
    delta = adapter(feat)
    delta = center_and_clip_delta(delta, center_delta, tau)
    return logits + lam * delta


def evaluate_cache_variants(
    cache: PointCache,
    adapter: CoDAAdapter,
    num_classes: int,
    ignore_index: int,
    lambdas: list[float],
    taus: list[float],
    center_delta: bool,
):
    device = torch.device("cuda")
    feat = cache.feat.to(device)
    logits = cache.logits.to(device)
    labels = cache.labels.to(device)
    variants = [("base", None)]
    for lam in lambdas:
        for tau in taus:
            name = f"lam{lam:g}_tau{tau:g}".replace(".", "p")
            variants.append((name, (lam, tau)))
    confusions = {name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device) for name, _ in variants}
    batch = 200000
    adapter.eval()
    with torch.inference_mode():
        for start in range(0, labels.numel(), batch):
            end = min(start + batch, labels.numel())
            feat_b = feat[start:end]
            logits_b = logits[start:end]
            labels_b = labels[start:end]
            for name, spec in variants:
                if spec is None:
                    pred = logits_b.argmax(dim=1)
                else:
                    pred = corrected_logits(adapter, feat_b, logits_b, spec[0], spec[1], center_delta).argmax(dim=1)
                update_confusion(confusions[name], pred, labels_b, num_classes, ignore_index)
    summaries = {name: summarize_confusion(conf.detach().cpu().numpy(), SCANNET20_CLASS_NAMES) for name, conf in confusions.items()}
    return summaries, confusions


def evaluate_val_stream(
    args: argparse.Namespace,
    model,
    cfg,
    adapter: CoDAAdapter,
    num_classes: int,
    ignore_index: int,
    specs: dict[str, tuple[float, float]],
):
    device = torch.device("cuda")
    variants = {"base": None, **specs}
    confusions = {name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device) for name in variants}
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
    adapter.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, batch = forward_features(model, batch)
            feat_e, logits_e, labels_e = eval_tensors(feat, logits, labels, batch)
            for name, spec in variants.items():
                if spec is None:
                    pred = logits_e.argmax(dim=1)
                else:
                    pred = corrected_logits(adapter, feat_e, logits_e, spec[0], spec[1], args.center_delta).argmax(dim=1)
                update_confusion(confusions[name], pred, labels_e, num_classes, ignore_index)
            seen += 1
            if (batch_idx + 1) % 25 == 0:
                base_sum = summarize_confusion(confusions["base"].detach().cpu().numpy(), SCANNET20_CLASS_NAMES)
                print(f"[val] batch={batch_idx + 1} base_mIoU={base_sum['mIoU']:.4f}", flush=True)
    summaries = {name: summarize_confusion(conf.detach().cpu().numpy(), SCANNET20_CLASS_NAMES) for name, conf in confusions.items()}
    return summaries, confusions, seen


def row_for_summary(name: str, summary: dict, base: dict, conf: np.ndarray) -> dict:
    picture = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    picture_target = max(summary["target_sum"][picture], 1.0)
    return {
        "variant": name,
        "mIoU": summary["mIoU"],
        "delta_mIoU": summary["mIoU"] - base["mIoU"],
        "mAcc": summary["mAcc"],
        "allAcc": summary["allAcc"],
        "picture_iou": summary["iou"][picture],
        "picture_delta_iou": summary["iou"][picture] - base["iou"][picture],
        "picture_to_wall_frac": conf[picture, wall] / picture_target,
        "counter_iou": summary["iou"][NAME_TO_ID["counter"]],
        "counter_delta_iou": summary["iou"][NAME_TO_ID["counter"]] - base["iou"][NAME_TO_ID["counter"]],
        "desk_iou": summary["iou"][NAME_TO_ID["desk"]],
        "desk_delta_iou": summary["iou"][NAME_TO_ID["desk"]] - base["iou"][NAME_TO_ID["desk"]],
        "sink_iou": summary["iou"][NAME_TO_ID["sink"]],
        "sink_delta_iou": summary["iou"][NAME_TO_ID["sink"]] - base["iou"][NAME_TO_ID["sink"]],
        "cabinet_iou": summary["iou"][NAME_TO_ID["cabinet"]],
        "cabinet_delta_iou": summary["iou"][NAME_TO_ID["cabinet"]] - base["iou"][NAME_TO_ID["cabinet"]],
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in row.items()})


def parse_spec(variant: str) -> tuple[float, float]:
    parts = variant.split("_")
    lam = float(parts[0][3:].replace("p", "."))
    tau = float(parts[1][3:].replace("p", "."))
    return lam, tau


def select_specs(heldout_rows: list[dict], selection: str) -> dict[str, tuple[float, float]]:
    nonbase = [row for row in heldout_rows if row["variant"] != "base"]
    base = next(row for row in heldout_rows if row["variant"] == "base")
    best_miou = max(nonbase, key=lambda r: (r["mIoU"], r["picture_iou"]))
    safe = [row for row in nonbase if r_safe(row, base)]
    if safe:
        best_picture = max(safe, key=lambda r: (r["picture_iou"], r["mIoU"]))
    else:
        best_picture = max(nonbase, key=lambda r: (r["picture_iou"], r["mIoU"]))
    if selection == "picture_safe":
        primary = best_picture
        secondary = best_miou
    else:
        primary = best_miou
        secondary = best_picture
    specs = {
        f"selected_primary__{primary['variant']}": parse_spec(primary["variant"]),
    }
    if secondary["variant"] != primary["variant"]:
        specs[f"selected_secondary__{secondary['variant']}"] = parse_spec(secondary["variant"])
    return specs


def r_safe(row: dict, base: dict) -> bool:
    return row["mIoU"] >= base["mIoU"] - 0.001


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.config = str((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if args.adapter_path is not None:
        args.adapter_path = (repo_root / args.adapter_path).resolve() if not args.adapter_path.is_absolute() else args.adapter_path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    cfg = load_config(Path(args.config))
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    pairs = parse_pairs(args.class_pairs)
    weak_classes = parse_names(args.weak_classes)
    lambdas = parse_float_list(args.eval_lambdas)
    taus = parse_float_list(args.eval_taus)
    if args.dry_run:
        loader = build_loader(cfg, args.val_split, args.data_root, 1, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] pairs={args.class_pairs}", flush=True)
        print(f"[dry] lambdas={lambdas} taus={taus}", flush=True)
        return 0

    model = build_model(cfg, args.weight)
    if args.adapter_path is not None:
        payload = torch.load(args.adapter_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        adapter = build_adapter_from_state(state_dict, args.dropout).cuda().eval()
        if args.eval_all_val:
            specs = {
                f"lam{lam:g}_tau{tau:g}".replace(".", "p"): (lam, tau)
                for lam in lambdas
                for tau in taus
            }
        else:
            metadata = payload.get("metadata", {})
            selected = metadata.get("selected_specs", {}) if isinstance(metadata, dict) else {}
            specs = {key: tuple(value) for key, value in selected.items()}
            if not specs:
                specs = {"selected_default__lam1_tau1": (1.0, 1.0)}
        val_summaries, val_conf, seen_val = evaluate_val_stream(
            args, model, cfg, adapter, num_classes, ignore_index, specs
        )
        val_base = val_summaries["base"]
        val_rows = [
            row_for_summary(name, summary, val_base, val_conf[name].detach().cpu().numpy())
            for name, summary in val_summaries.items()
        ]
        val_rows = sorted(val_rows, key=lambda r: (r["mIoU"], r["picture_iou"]), reverse=True)
        fields = [
            "variant",
            "mIoU",
            "delta_mIoU",
            "mAcc",
            "allAcc",
            "picture_iou",
            "picture_delta_iou",
            "picture_to_wall_frac",
            "counter_iou",
            "counter_delta_iou",
            "desk_iou",
            "desk_delta_iou",
            "sink_iou",
            "sink_delta_iou",
            "cabinet_iou",
            "cabinet_delta_iou",
        ]
        write_csv(args.output_dir / "coda_val_all.csv", val_rows, fields)
        lines = [
            "# CoDA Adapter Validation Sweep",
            "",
            f"- adapter: `{args.adapter_path}`",
            f"- config: `{args.config}`",
            f"- weight: `{args.weight}`",
            f"- val batches seen: {seen_val}",
            "",
            "| variant | mIoU | delta | picture IoU | picture delta | picture->wall | counter delta | desk delta | sink delta | cabinet delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in val_rows:
            lines.append(
                f"| {row['variant']} | {row['mIoU']:.4f} | {row['delta_mIoU']:+.4f} | "
                f"{row['picture_iou']:.4f} | {row['picture_delta_iou']:+.4f} | "
                f"{row['picture_to_wall_frac']:.4f} | {row['counter_delta_iou']:+.4f} | "
                f"{row['desk_delta_iou']:+.4f} | {row['sink_delta_iou']:+.4f} | "
                f"{row['cabinet_delta_iou']:+.4f} |"
            )
        (args.output_dir / "coda_val_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[done] wrote val sweep {args.output_dir}", flush=True)
        return 0

    train_cache, heldout_cache = collect_train_heldout(args, model, cfg, num_classes)
    adapter, train_meta = train_adapter(args, train_cache, pairs, weak_classes, num_classes)
    heldout_summaries, heldout_conf = evaluate_cache_variants(
        heldout_cache, adapter, num_classes, ignore_index, lambdas, taus, args.center_delta
    )
    heldout_base = heldout_summaries["base"]
    heldout_rows = [
        row_for_summary(name, summary, heldout_base, heldout_conf[name].detach().cpu().numpy())
        for name, summary in heldout_summaries.items()
    ]
    heldout_rows = sorted(heldout_rows, key=lambda r: (r["mIoU"], r["picture_iou"]), reverse=True)
    selected_specs = select_specs(heldout_rows, args.selection)
    print(f"[select] {selected_specs}", flush=True)
    val_summaries, val_conf, seen_val = evaluate_val_stream(args, model, cfg, adapter, num_classes, ignore_index, selected_specs)
    val_base = val_summaries["base"]
    val_rows = [
        row_for_summary(name, summary, val_base, val_conf[name].detach().cpu().numpy())
        for name, summary in val_summaries.items()
    ]
    val_rows = sorted(val_rows, key=lambda r: (r["mIoU"], r["picture_iou"]), reverse=True)

    fields = [
        "variant",
        "mIoU",
        "delta_mIoU",
        "mAcc",
        "allAcc",
        "picture_iou",
        "picture_delta_iou",
        "picture_to_wall_frac",
        "counter_iou",
        "counter_delta_iou",
        "desk_iou",
        "desk_delta_iou",
        "sink_iou",
        "sink_delta_iou",
        "cabinet_iou",
        "cabinet_delta_iou",
    ]
    write_csv(args.output_dir / "coda_heldout_sweep.csv", heldout_rows, fields)
    write_csv(args.output_dir / "coda_val_selected.csv", val_rows, fields)
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "metadata": {
                "train_meta": train_meta,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
        },
        args.output_dir / "coda_adapter.pt",
    )
    metadata = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train": {
            "num_points": int(train_cache.labels.numel()),
            "class_counts": train_cache.class_counts,
            "seen_batches": train_cache.seen_batches,
        },
        "heldout": {
            "num_points": int(heldout_cache.labels.numel()),
            "class_counts": heldout_cache.class_counts,
            "seen_batches": heldout_cache.seen_batches,
        },
        "train_meta": train_meta,
        "selected_specs": {k: list(v) for k, v in selected_specs.items()},
        "val_seen_batches": seen_val,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# CoDA Decoder Adapter",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- train points: {train_cache.labels.numel()}",
        f"- heldout train points: {heldout_cache.labels.numel()}",
        f"- val batches seen: {seen_val}",
        f"- trainable params: {train_meta['num_trainable']}",
        f"- weak class weight: `{args.weak_class_weight}`",
        f"- pair weight: `{args.pair_weight}`",
        f"- KL weight: `{args.kl_weight}`",
        f"- residual L2: `{args.residual_l2}`",
        f"- train tau: `{args.train_tau}`",
        f"- eval lambdas: `{args.eval_lambdas}`",
        f"- eval taus: `{args.eval_taus}`",
        "",
        "## Training Trace",
        "",
        "| step | loss | CE | pair | KL | delta L2 | base acc | adapter acc | delta RMS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in train_meta["history"]:
        lines.append(
            f"| {row['step']} | {row['loss']:.4f} | {row['ce']:.4f} | {row['pair']:.4f} | "
            f"{row['kl']:.4f} | {row['delta_l2']:.4f} | {row['base_acc']:.4f} | "
            f"{row['adapter_acc']:.4f} | {row['delta_rms']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Heldout Selection",
            "",
            "| variant | mIoU | delta | picture IoU | picture delta | picture->wall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in heldout_rows[:12]:
        lines.append(
            f"| {row['variant']} | {row['mIoU']:.4f} | {row['delta_mIoU']:+.4f} | "
            f"{row['picture_iou']:.4f} | {row['picture_delta_iou']:+.4f} | "
            f"{row['picture_to_wall_frac']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Selected specs: `{selected_specs}`",
            "",
            "## ScanNet Val",
            "",
            "| variant | mIoU | delta | picture IoU | picture delta | picture->wall | counter delta | desk delta | sink delta | cabinet delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in val_rows:
        lines.append(
            f"| {row['variant']} | {row['mIoU']:.4f} | {row['delta_mIoU']:+.4f} | "
            f"{row['picture_iou']:.4f} | {row['picture_delta_iou']:+.4f} | "
            f"{row['picture_to_wall_frac']:.4f} | {row['counter_delta_iou']:+.4f} | "
            f"{row['desk_delta_iou']:+.4f} | {row['sink_delta_iou']:+.4f} | "
            f"{row['cabinet_delta_iou']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{args.output_dir / 'coda_heldout_sweep.csv'}`",
            f"- `{args.output_dir / 'coda_val_selected.csv'}`",
            f"- `{args.output_dir / 'coda_adapter.pt'}`",
            f"- `{args.output_dir / 'metadata.json'}`",
            "",
        ]
    )
    (args.output_dir / "coda_decoder_adapter.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
