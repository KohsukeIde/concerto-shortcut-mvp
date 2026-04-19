#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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


class ConstrainedSetDecoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int,
        class_embed_dim: int,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
        logit_mean: torch.Tensor,
        logit_std: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("feat_mean", feat_mean.float())
        self.register_buffer("feat_std", feat_std.float().clamp_min(1e-6))
        self.register_buffer("logit_mean", logit_mean.float())
        self.register_buffer("logit_std", logit_std.float().clamp_min(1e-6))
        self.register_buffer("prototypes", F.normalize(prototypes.float(), dim=1))
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.scalar_proj = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + class_embed_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor, logits: torch.Tensor, graph_neighbor: torch.Tensor) -> torch.Tensor:
        feat_z = (feat.float() - self.feat_mean.to(feat.device)) / self.feat_std.to(feat.device)
        logits = logits.float()
        logit_z = (logits - self.logit_mean.to(logits.device)) / self.logit_std.to(logits.device)
        prob = logits.softmax(dim=1)
        top = logits.topk(k=2, dim=1)
        top1 = top.values[:, :1]
        top2_gap = top.values[:, :1] - top.values[:, 1:2]
        margin_to_top = (logits - top1) / self.logit_std.to(logits.device)
        entropy = -(prob.clamp_min(1e-9) * prob.clamp_min(1e-9).log()).sum(dim=1, keepdim=True)
        proto_cos = F.normalize(feat.float(), dim=1) @ self.prototypes.to(feat.device).t()
        ranks = torch.empty_like(logits, dtype=torch.float32)
        order = logits.argsort(dim=1, descending=True)
        rank_values = torch.arange(self.num_classes, device=logits.device, dtype=torch.float32).unsqueeze(0).expand_as(order)
        ranks.scatter_(1, order, rank_values)
        rank_norm = ranks / max(self.num_classes - 1, 1)
        scalar = torch.stack(
            [
                logit_z,
                prob,
                margin_to_top,
                proto_cos,
                rank_norm,
                graph_neighbor.float(),
                entropy.expand_as(logits) / math.log(self.num_classes),
                top2_gap.expand_as(logits) / self.logit_std.to(logits.device),
            ],
            dim=-1,
        )
        feat_h = self.feat_proj(feat_z).unsqueeze(1).expand(-1, self.num_classes, -1)
        cls_h = self.class_embed(torch.arange(self.num_classes, device=feat.device)).unsqueeze(0).expand(
            feat.shape[0], -1, -1
        )
        scalar_h = self.scalar_proj(scalar)
        return self.scorer(torch.cat([feat_h, cls_h, scalar_h], dim=-1)).squeeze(-1)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-aware constrained top-K set decoder for the frozen "
            "concerto_base_origin decoder-probe checkpoint. Trains a tiny "
            "candidate-set reranker on a scene split of ScanNet train, selects "
            "constraints on a held-out train split, then evaluates on ScanNet val."
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
    parser.add_argument("--max-train-batches", type=int, default=320)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-rerank-train-points", type=int, default=500000)
    parser.add_argument("--max-heldout-points", type=int, default=250000)
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--train-top-k", type=int, default=5)
    parser.add_argument("--eval-top-ks", default="2,3,5")
    parser.add_argument("--lambdas", default="0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--taus", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--trust-gaps", default="999.0,2.0,1.0,0.5")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--class-embed-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sample-batch-size", type=int, default=8192)
    parser.add_argument("--residual-l2", type=float, default=5e-3)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--weak-class-weight", type=float, default=2.0)
    parser.add_argument("--selection", default="miou_then_picture", choices=["miou_then_picture", "picture_safe"])
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


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


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


def build_neighbor_mask(pairs: list[tuple[int, int]], num_classes: int) -> torch.Tensor:
    mask = torch.zeros((num_classes, num_classes), dtype=torch.bool)
    for a, b in pairs:
        mask[a, b] = True
        mask[b, a] = True
    return mask


def candidate_mask(logits: torch.Tensor, top_k: int, neighbor_mask: torch.Tensor | None = None) -> torch.Tensor:
    top_k = min(top_k, logits.shape[1])
    top_idx = logits.topk(k=top_k, dim=1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, top_idx, True)
    if neighbor_mask is not None:
        mask |= neighbor_mask.to(logits.device)[top_idx].any(dim=1)
    return mask


def graph_neighbor_features(logits: torch.Tensor, neighbor_mask: torch.Tensor, top_k: int) -> torch.Tensor:
    top_idx = logits.topk(k=min(top_k, logits.shape[1]), dim=1).indices
    return neighbor_mask.to(logits.device)[top_idx].any(dim=1)


def center_candidate_scores(scores: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    count = candidate.sum(dim=1, keepdim=True).clamp_min(1)
    mean = scores.masked_fill(~candidate, 0.0).sum(dim=1, keepdim=True) / count
    return (scores - mean).masked_fill(~candidate, 0.0)


def apply_correction(
    decoder: ConstrainedSetDecoder,
    feat: torch.Tensor,
    logits: torch.Tensor,
    neighbor_mask: torch.Tensor,
    top_k: int,
    lam: float,
    tau: float,
    trust_gap: float,
) -> torch.Tensor:
    candidate = candidate_mask(logits, top_k, neighbor_mask)
    graph = graph_neighbor_features(logits, neighbor_mask, top_k)
    raw = decoder(feat, logits, graph)
    delta = center_candidate_scores(raw, candidate).clamp(min=-tau, max=tau)
    if trust_gap < 900:
        vals = logits.topk(k=2, dim=1).values
        confident = (vals[:, 0] - vals[:, 1]) > trust_gap
        delta[confident] = 0
    corrected = logits.clone()
    corrected[candidate] = corrected[candidate] + lam * delta[candidate]
    return corrected


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
            if train_raw["total"] >= args.max_rerank_train_points and held_raw["total"] >= args.max_heldout_points:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, _ = forward_features(model, batch)
            is_heldout = (batch_idx % args.heldout_mod) == args.heldout_remainder
            if is_heldout:
                append_to_cache(held_raw, feat, logits, labels, args.max_heldout_points, args.max_per_class, num_classes)
            else:
                append_to_cache(train_raw, feat, logits, labels, args.max_rerank_train_points, args.max_per_class, num_classes)
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


def compute_prototypes(cache: PointCache, num_classes: int) -> torch.Tensor:
    feat_dim = cache.feat.shape[1]
    prototypes = torch.zeros((num_classes, feat_dim), dtype=torch.float32)
    global_mean = cache.feat.mean(dim=0)
    for cls in range(num_classes):
        mask = cache.labels == cls
        if mask.any():
            prototypes[cls] = cache.feat[mask].mean(dim=0)
        else:
            prototypes[cls] = global_mean
    return prototypes


def train_decoder(args: argparse.Namespace, cache: PointCache, pairs: list[tuple[int, int]], weak_classes: list[int], num_classes: int):
    device = torch.device("cuda")
    neighbor_mask = build_neighbor_mask(pairs, num_classes).to(device)
    feat = cache.feat.to(device)
    logits = cache.logits.to(device)
    labels = cache.labels.to(device)
    candidate = candidate_mask(logits, args.train_top_k, neighbor_mask)
    valid = candidate.gather(1, labels[:, None]).squeeze(1)
    valid_idx = valid.nonzero(as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        raise RuntimeError("no train samples have ground truth in candidate set")
    feat = feat[valid_idx]
    logits = logits[valid_idx]
    labels = labels[valid_idx]
    candidate = candidate[valid_idx]
    weights = class_weights(labels.detach().cpu(), weak_classes, num_classes, args.weak_class_weight).to(device)
    feat_mean = feat.mean(dim=0, keepdim=True).detach().cpu()
    feat_std = feat.std(dim=0, keepdim=True).clamp_min(1e-6).detach().cpu()
    logit_mean = logits.mean(dim=0, keepdim=True).detach().cpu()
    logit_std = logits.std(dim=0, keepdim=True).clamp_min(1e-6).detach().cpu()
    prototypes = compute_prototypes(cache, num_classes)
    decoder = ConstrainedSetDecoder(
        feat_dim=feat.shape[1],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        class_embed_dim=args.class_embed_dim,
        feat_mean=feat_mean,
        feat_std=feat_std,
        logit_mean=logit_mean,
        logit_std=logit_std,
        prototypes=prototypes,
    ).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = labels.numel()
    history = []
    for step in range(args.steps):
        idx = torch.randint(0, n, (min(args.sample_batch_size, n),), device=device)
        feat_b = feat[idx]
        logits_b = logits[idx]
        labels_b = labels[idx]
        cand_b = candidate[idx]
        weights_b = weights[idx]
        graph = graph_neighbor_features(logits_b, neighbor_mask, args.train_top_k)
        raw = decoder(feat_b, logits_b, graph)
        delta = center_candidate_scores(raw, cand_b)
        final = (logits_b + delta).masked_fill(~cand_b, -1e9)
        ce = F.cross_entropy(final, labels_b, reduction="none")
        loss = (ce * weights_b).sum() / weights_b.sum().clamp_min(1e-6)
        if args.residual_l2 > 0:
            loss = loss + args.residual_l2 * delta[cand_b].pow(2).mean()
        if args.kl_weight > 0:
            base_prob = logits_b.masked_fill(~cand_b, -1e9).softmax(dim=1)
            log_prob = final.log_softmax(dim=1)
            kl = F.kl_div(log_prob, base_prob, reduction="batchmean")
            loss = loss + args.kl_weight * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        opt.step()
        if (step + 1) % max(args.steps // 10, 1) == 0 or step == 0:
            with torch.no_grad():
                base_pred = logits_b.masked_fill(~cand_b, -1e9).argmax(dim=1)
                pred = final.argmax(dim=1)
                row = {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "candidate_base_acc": float((base_pred == labels_b).float().mean().item()),
                    "candidate_train_acc": float((pred == labels_b).float().mean().item()),
                    "delta_rms": float(delta[cand_b].pow(2).mean().sqrt().item()),
                }
            history.append(row)
            print(
                f"[train] step={row['step']} loss={row['loss']:.4f} "
                f"base={row['candidate_base_acc']:.4f} acc={row['candidate_train_acc']:.4f} "
                f"delta={row['delta_rms']:.4f}",
                flush=True,
            )
    return decoder, {
        "history": history,
        "train_candidate_coverage": float(valid.float().mean().item()),
        "valid_train_points": int(labels.numel()),
    }


def evaluate_cache_variants(
    cache: PointCache,
    decoder: ConstrainedSetDecoder,
    pairs: list[tuple[int, int]],
    num_classes: int,
    ignore_index: int,
    top_ks: list[int],
    lambdas: list[float],
    taus: list[float],
    trust_gaps: list[float],
):
    device = torch.device("cuda")
    feat = cache.feat.to(device)
    logits = cache.logits.to(device)
    labels = cache.labels.to(device)
    neighbor_mask = build_neighbor_mask(pairs, num_classes).to(device)
    variants = [("base", None)]
    for k in top_ks:
        for lam in lambdas:
            for tau in taus:
                for gap in trust_gaps:
                    name = f"k{k}_lam{lam:g}_tau{tau:g}_gap{gap:g}".replace(".", "p")
                    variants.append((name, (k, lam, tau, gap)))
    confusions = {name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device) for name, _ in variants}
    batch = 200000
    decoder.eval()
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
                    pred = apply_correction(decoder, feat_b, logits_b, neighbor_mask, *spec).argmax(dim=1)
                update_confusion(confusions[name], pred, labels_b, num_classes, ignore_index)
    summaries = {name: summarize_confusion(conf.detach().cpu().numpy(), SCANNET20_CLASS_NAMES) for name, conf in confusions.items()}
    return summaries, confusions


def evaluate_val_stream(
    args: argparse.Namespace,
    model,
    cfg,
    decoder: ConstrainedSetDecoder,
    pairs: list[tuple[int, int]],
    num_classes: int,
    ignore_index: int,
    specs: dict[str, tuple[int, float, float, float]],
):
    device = torch.device("cuda")
    neighbor_mask = build_neighbor_mask(pairs, num_classes).to(device)
    variants = {"base": None, **specs}
    confusions = {name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device) for name in variants}
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
    decoder.eval()
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
                    pred = apply_correction(decoder, feat_e, logits_e, neighbor_mask, *spec).argmax(dim=1)
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
        "desk_iou": summary["iou"][NAME_TO_ID["desk"]],
        "sink_iou": summary["iou"][NAME_TO_ID["sink"]],
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in row.items()})


def select_specs(heldout_rows: list[dict], selection: str) -> dict[str, tuple[int, float, float, float]]:
    def parse_spec(variant: str) -> tuple[int, float, float, float]:
        parts = variant.split("_")
        k = int(parts[0][1:])
        lam = float(parts[1][3:].replace("p", "."))
        tau = float(parts[2][3:].replace("p", "."))
        gap = float(parts[3][3:].replace("p", "."))
        return k, lam, tau, gap

    nonbase = [row for row in heldout_rows if row["variant"] != "base"]
    best_miou = max(nonbase, key=lambda r: (r["mIoU"], r["picture_iou"]))
    base = next(row for row in heldout_rows if row["variant"] == "base")
    safe = [row for row in nonbase if row["mIoU"] >= base["mIoU"] - 0.001]
    if safe:
        best_picture = max(safe, key=lambda r: (r["picture_iou"], r["mIoU"]))
    else:
        best_picture = max(nonbase, key=lambda r: (r["picture_iou"], r["mIoU"]))
    specs = {
        f"selected_best_miou__{best_miou['variant']}": parse_spec(best_miou["variant"]),
        f"selected_best_picture_safe__{best_picture['variant']}": parse_spec(best_picture["variant"]),
    }
    return specs


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.config = str((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    cfg = load_config(Path(args.config))
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    pairs = parse_pairs(args.class_pairs)
    weak_classes = parse_names(args.weak_classes)
    top_ks = parse_int_list(args.eval_top_ks)
    lambdas = parse_float_list(args.lambdas)
    taus = parse_float_list(args.taus)
    trust_gaps = parse_float_list(args.trust_gaps)
    if args.dry_run:
        loader = build_loader(cfg, args.val_split, args.data_root, 1, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] pairs={args.class_pairs}", flush=True)
        print(f"[dry] top_ks={top_ks} lambdas={lambdas} taus={taus} gaps={trust_gaps}", flush=True)
        return 0

    model = build_model(cfg, args.weight)
    train_cache, heldout_cache = collect_train_heldout(args, model, cfg, num_classes)
    decoder, train_meta = train_decoder(args, train_cache, pairs, weak_classes, num_classes)
    heldout_summaries, heldout_conf = evaluate_cache_variants(
        heldout_cache, decoder, pairs, num_classes, ignore_index, top_ks, lambdas, taus, trust_gaps
    )
    heldout_base = heldout_summaries["base"]
    heldout_rows = [
        row_for_summary(name, summary, heldout_base, heldout_conf[name].detach().cpu().numpy())
        for name, summary in heldout_summaries.items()
    ]
    heldout_rows = sorted(heldout_rows, key=lambda r: (r["mIoU"], r["picture_iou"]), reverse=True)
    selected_specs = select_specs(heldout_rows, args.selection)
    print(f"[select] {selected_specs}", flush=True)
    val_summaries, val_conf, seen_val = evaluate_val_stream(args, model, cfg, decoder, pairs, num_classes, ignore_index, selected_specs)
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
        "desk_iou",
        "sink_iou",
    ]
    write_csv(args.output_dir / "constrained_topk_heldout_sweep.csv", heldout_rows, fields)
    write_csv(args.output_dir / "constrained_topk_val_selected.csv", val_rows, fields)
    torch.save({"state_dict": decoder.state_dict(), "metadata": train_meta}, args.output_dir / "constrained_topk_set_decoder.pt")
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

    best_heldout = heldout_rows[0]
    best_val = val_rows[0]
    lines = [
        "# Constrained Top-K Set Decoder",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- train points: {train_cache.labels.numel()}",
        f"- heldout train points: {heldout_cache.labels.numel()}",
        f"- val batches seen: {seen_val}",
        f"- train top-K: `{args.train_top_k}`",
        f"- eval top-Ks: `{args.eval_top_ks}`",
        f"- lambdas: `{args.lambdas}`",
        f"- taus: `{args.taus}`",
        f"- trust gaps: `{args.trust_gaps}`",
        "",
        "## Heldout Selection",
        "",
        "| variant | mIoU | delta | picture IoU | picture delta | picture->wall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in heldout_rows[:10]:
        lines.append(
            "| {variant} | {mIoU:.4f} | {delta_mIoU:+.4f} | {picture_iou:.4f} | {picture_delta_iou:+.4f} | {picture_to_wall_frac:.4f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## ScanNet Val Selected Variants",
        "",
        "| variant | mIoU | delta | picture IoU | picture delta | picture->wall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in val_rows:
        lines.append(
            "| {variant} | {mIoU:.4f} | {delta_mIoU:+.4f} | {picture_iou:.4f} | {picture_delta_iou:+.4f} | {picture_to_wall_frac:.4f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Training Trace",
        "",
        "| step | loss | candidate base acc | candidate train acc | delta RMS |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in train_meta["history"]:
        lines.append(
            f"| {row['step']} | {row['loss']:.4f} | {row['candidate_base_acc']:.4f} | "
            f"{row['candidate_train_acc']:.4f} | {row['delta_rms']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- The base decoder checkpoint is fixed and was not retrained.",
        "- The ScanNet train split is used only to train and select the reranker constraints; ScanNet val is used once for selected variants.",
        "- `trust_gap=999` means no confident-top1 skip. Smaller values skip corrections when the base top1-top2 gap exceeds that threshold.",
    ]
    (args.output_dir / "constrained_topk_set_decoder.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[best-heldout] {best_heldout}", flush=True)
    print(f"[best-val] {best_val}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
