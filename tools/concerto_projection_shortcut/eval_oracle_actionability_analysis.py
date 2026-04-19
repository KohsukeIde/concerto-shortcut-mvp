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
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
    parse_pairs,
    pair_name,
)


@dataclass
class TrainCache:
    feat: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    class_counts: dict[str, int]
    seen_batches: int


@dataclass
class PairProbe:
    pair: tuple[int, int]
    mean: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    train_bal_acc: float


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle/actionability analysis for the origin decoder-probe "
            "checkpoint. Measures whether weak-class errors are reachable by "
            "readout-family methods before trying another method."
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
        "--weak-classes",
        default="picture,counter,desk,sink,cabinet,shower curtain,door",
    )
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall",
    )
    parser.add_argument("--top-ks", default="1,2,3,5,10,20")
    parser.add_argument("--oracle-top-ks", default="2,3,5,10")
    parser.add_argument("--graph-top-ks", default="1,2,3,5")
    parser.add_argument("--prior-alphas", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=256)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-train-points", type=int, default=600000)
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--max-geometry-per-class", type=int, default=60000)
    parser.add_argument("--pair-probe-steps", type=int, default=800)
    parser.add_argument("--pair-probe-lr", type=float, default=0.05)
    parser.add_argument("--pair-probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--bias-steps", type=int, default=1000)
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_names(text: str) -> list[int]:
    ids: list[int] = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {name}")
        ids.append(NAME_TO_ID[name])
    if not ids:
        raise ValueError("no classes were provided")
    return ids


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


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


def collect_train_cache(args: argparse.Namespace, model, cfg, num_classes: int) -> TrainCache:
    feats: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    class_counts = {idx: 0 for idx in range(num_classes)}
    total_points = 0
    loader = build_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker)
    seen_batches = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
                break
            if total_points >= args.max_train_points:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, _ = forward_features(model, batch)
            seen_batches += 1
            valid = (labels >= 0) & (labels < num_classes)
            if not valid.any():
                continue
            keep_indices = []
            for cls in range(num_classes):
                room = args.max_per_class - class_counts[cls]
                if room <= 0:
                    continue
                idx = ((labels == cls) & valid).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                remaining_total = args.max_train_points - total_points - sum(i.numel() for i in keep_indices)
                if remaining_total <= 0:
                    break
                cap = min(room, remaining_total)
                if idx.numel() > cap:
                    idx = idx[:cap]
                keep_indices.append(idx)
                class_counts[cls] += int(idx.numel())
            if keep_indices:
                keep = torch.cat(keep_indices, dim=0)
                feats.append(feat[keep].detach().cpu())
                logits_list.append(logits[keep].detach().cpu())
                labels_list.append(labels[keep].detach().cpu())
                total_points += int(keep.numel())
            if (batch_idx + 1) % 25 == 0:
                counts = " ".join(f"{SCANNET20_CLASS_NAMES[k]}={v}" for k, v in class_counts.items() if v)
                print(f"[collect-train] batch={batch_idx + 1} total={total_points} {counts}", flush=True)
    if not labels_list:
        raise RuntimeError("no train cache collected")
    cache = TrainCache(
        feat=torch.cat(feats, dim=0),
        logits=torch.cat(logits_list, dim=0),
        labels=torch.cat(labels_list, dim=0).long(),
        class_counts={SCANNET20_CLASS_NAMES[k]: int(v) for k, v in class_counts.items()},
        seen_batches=seen_batches,
    )
    print(f"[collect-train] done points={cache.labels.numel()} seen_batches={seen_batches}", flush=True)
    return cache


def fit_pair_probe(pair: tuple[int, int], cache: TrainCache, args: argparse.Namespace) -> PairProbe:
    pos_cls, neg_cls = pair
    pos_idx = (cache.labels == pos_cls).nonzero(as_tuple=False).flatten()
    neg_idx = (cache.labels == neg_cls).nonzero(as_tuple=False).flatten()
    n = min(pos_idx.numel(), neg_idx.numel())
    if n == 0:
        raise RuntimeError(f"cannot fit pair probe with no samples: {pair_name(pair)}")
    x = torch.cat([cache.feat[pos_idx[:n]], cache.feat[neg_idx[:n]]], dim=0).cuda()
    y = torch.cat([torch.ones(n), torch.zeros(n)], dim=0).cuda()
    perm = torch.randperm(x.shape[0], device=x.device)
    x = x[perm]
    y = y[perm]
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    xz = (x - mean) / std
    weight = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    bias = torch.zeros((), device=x.device, requires_grad=True)
    opt = torch.optim.AdamW([weight, bias], lr=args.pair_probe_lr, weight_decay=args.pair_probe_weight_decay)
    for step in range(args.pair_probe_steps):
        opt.zero_grad(set_to_none=True)
        score = xz @ weight + bias
        loss = F.binary_cross_entropy_with_logits(score, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        score = xz @ weight + bias
        pred = score >= 0
        pos_acc = (pred[y == 1] == 1).float().mean().item()
        neg_acc = (pred[y == 0] == 0).float().mean().item()
        bal = (pos_acc + neg_acc) / 2.0
    print(f"[pair-probe] {pair_name(pair)} train_bal_acc={bal:.4f} n={n}", flush=True)
    return PairProbe(
        pair=pair,
        mean=mean.detach().cpu(),
        std=std.detach().cpu(),
        weight=weight.detach().cpu(),
        bias=bias.detach().cpu(),
        train_bal_acc=bal,
    )


def fit_bias(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, args: argparse.Namespace, balanced: bool) -> tuple[torch.Tensor, list[dict]]:
    device = torch.device("cuda")
    logits = logits.to(device)
    labels = labels.to(device)
    bias = torch.zeros(num_classes, device=device, requires_grad=True)
    opt = torch.optim.AdamW([bias], lr=args.bias_lr, weight_decay=args.bias_weight_decay)
    class_weight = None
    if balanced:
        counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
        class_weight = (counts.sum() / (num_classes * counts)).to(device)
        class_weight = class_weight / class_weight.mean()
    history = []
    for step in range(args.bias_steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits + bias, labels, weight=class_weight)
        loss.backward()
        opt.step()
        if (step + 1) % max(args.bias_steps // 5, 1) == 0 or step == 0:
            with torch.no_grad():
                acc = ((logits + bias).argmax(dim=1) == labels).float().mean().item()
            history.append({"step": step + 1, "loss": float(loss.item()), "acc": acc})
    return bias.detach().cpu(), history


def cosine_distance_matrix(centroids: torch.Tensor) -> torch.Tensor:
    c = F.normalize(centroids.float(), dim=1)
    return c @ c.t()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def binomial_ci(count: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = count / total
    se = (p * (1.0 - p) / total) ** 0.5
    return max(0.0, p - z * se), min(1.0, p + z * se)


def eval_pair_probe_predictions(feat: torch.Tensor, logits: torch.Tensor, probes: list[PairProbe]) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    top2 = logits.topk(k=2, dim=1).indices
    for probe in probes:
        pos_cls, neg_cls = probe.pair
        has_pos = (top2 == pos_cls).any(dim=1)
        has_neg = (top2 == neg_cls).any(dim=1)
        mask = has_pos & has_neg
        if not mask.any():
            continue
        x = (feat[mask] - probe.mean.to(feat.device)) / probe.std.to(feat.device)
        score = x @ probe.weight.to(feat.device) + probe.bias.to(feat.device)
        pred[mask] = torch.where(score >= 0, torch.tensor(pos_cls, device=feat.device), torch.tensor(neg_cls, device=feat.device))
    return pred


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.config = str((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    weak_classes = parse_names(args.weak_classes)
    pairs = parse_pairs(args.class_pairs)
    top_ks = parse_int_list(args.top_ks)
    oracle_top_ks = parse_int_list(args.oracle_top_ks)
    graph_top_ks = parse_int_list(args.graph_top_ks)
    prior_alphas = parse_float_list(args.prior_alphas)
    cfg = load_config(Path(args.config))
    names = list(cfg.data.names)
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    if args.dry_run:
        loader = build_loader(cfg, args.val_split, args.data_root, 1, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] weak={[SCANNET20_CLASS_NAMES[i] for i in weak_classes]}", flush=True)
        print(f"[dry] pairs={[pair_name(pair) for pair in pairs]}", flush=True)
        return 0

    model = build_model(cfg, args.weight)
    train_cache = collect_train_cache(args, model, cfg, num_classes)
    pair_probes = [fit_pair_probe(pair, train_cache, args) for pair in pairs]
    bias_unweighted, bias_hist_unweighted = fit_bias(train_cache.logits, train_cache.labels, num_classes, args, balanced=False)
    bias_balanced, bias_hist_balanced = fit_bias(train_cache.logits, train_cache.labels, num_classes, args, balanced=True)
    train_counts = torch.bincount(train_cache.labels, minlength=num_classes).float().clamp_min(1.0)
    log_prior = (train_counts / train_counts.sum()).log()
    neighbor_mask = build_neighbor_mask(pairs, num_classes).cuda()

    variants: dict[str, torch.Tensor] = {
        "base": torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda"),
        "pair_probe_top2": torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda"),
        "bias_unweighted": torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda"),
        "bias_balanced": torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda"),
    }
    for k in oracle_top_ks:
        variants[f"oracle_top{k}"] = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")
    for k in graph_top_ks:
        variants[f"oracle_graph_top{k}"] = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")
    for alpha in prior_alphas:
        variants[f"prior_alpha{str(alpha).replace('.', 'p')}"] = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")

    hit_counts = {(cls, k): 0 for cls in weak_classes for k in top_ks}
    graph_hit_counts = {(cls, k): 0 for cls in weak_classes for k in graph_top_ks}
    target_counts = {cls: 0 for cls in weak_classes}
    top3_counts = {(cls, pred_cls): 0 for cls in weak_classes for pred_cls in range(num_classes)}
    geom_feats = {cls: [] for cls in range(num_classes)}
    geom_counts = {cls: 0 for cls in range(num_classes)}

    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    seen_val_batches = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, batch = forward_features(model, batch)
            feat_e, logits_e, labels_e = eval_tensors(feat, logits, labels, batch)
            valid = (labels_e >= 0) & (labels_e < num_classes) & (labels_e != ignore_index)
            if not valid.any():
                continue
            feat_e = feat_e[valid]
            logits_e = logits_e[valid]
            labels_e = labels_e[valid]
            base_pred = logits_e.argmax(dim=1)
            update_confusion(variants["base"], base_pred, labels_e, num_classes, ignore_index)
            update_confusion(
                variants["pair_probe_top2"],
                eval_pair_probe_predictions(feat_e, logits_e, pair_probes),
                labels_e,
                num_classes,
                ignore_index,
            )
            update_confusion(variants["bias_unweighted"], (logits_e + bias_unweighted.to(logits_e.device)).argmax(dim=1), labels_e, num_classes, ignore_index)
            update_confusion(variants["bias_balanced"], (logits_e + bias_balanced.to(logits_e.device)).argmax(dim=1), labels_e, num_classes, ignore_index)
            for alpha in prior_alphas:
                pred = (logits_e - alpha * log_prior.to(logits_e.device)).argmax(dim=1)
                update_confusion(variants[f"prior_alpha{str(alpha).replace('.', 'p')}"], pred, labels_e, num_classes, ignore_index)
            for k in oracle_top_ks:
                cand = candidate_mask(logits_e, k)
                pred = torch.where(cand.gather(1, labels_e[:, None]).squeeze(1), labels_e, base_pred)
                update_confusion(variants[f"oracle_top{k}"], pred, labels_e, num_classes, ignore_index)
            for k in graph_top_ks:
                cand = candidate_mask(logits_e, k, neighbor_mask)
                pred = torch.where(cand.gather(1, labels_e[:, None]).squeeze(1), labels_e, base_pred)
                update_confusion(variants[f"oracle_graph_top{k}"], pred, labels_e, num_classes, ignore_index)

            max_k = max(top_ks)
            top_idx = logits_e.topk(k=min(max_k, num_classes), dim=1).indices
            for cls in weak_classes:
                mask = labels_e == cls
                n = int(mask.sum().item())
                if n == 0:
                    continue
                target_counts[cls] += n
                top_cls = top_idx[mask]
                for k in top_ks:
                    hit_counts[(cls, k)] += int((top_cls[:, : min(k, num_classes)] == cls).any(dim=1).sum().item())
                top3 = top_cls[:, : min(3, num_classes)]
                for pred_cls in range(num_classes):
                    top3_counts[(cls, pred_cls)] += int((top3 == pred_cls).any(dim=1).sum().item())
                for k in graph_top_ks:
                    cand = candidate_mask(logits_e[mask], k, neighbor_mask)
                    graph_hit_counts[(cls, k)] += int(cand[:, cls].sum().item())
            for cls in range(num_classes):
                room = args.max_geometry_per_class - geom_counts[cls]
                if room <= 0:
                    continue
                idx = (labels_e == cls).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                if idx.numel() > room:
                    idx = idx[:room]
                geom_feats[cls].append(feat_e[idx].detach().cpu())
                geom_counts[cls] += int(idx.numel())
            seen_val_batches += 1
            if (batch_idx + 1) % 25 == 0:
                base_summary = summarize_confusion(variants["base"].detach().cpu().numpy(), names)
                print(f"[eval] batch={batch_idx + 1} base_mIoU={base_summary['mIoU']:.4f}", flush=True)

    summaries = {name: summarize_confusion(conf.detach().cpu().numpy(), names) for name, conf in variants.items()}
    base = summaries["base"]

    topk_rows = []
    for cls in weak_classes:
        denom = max(target_counts[cls], 1)
        for k in top_ks:
            count = hit_counts[(cls, k)]
            ci_low, ci_high = binomial_ci(count, denom)
            topk_rows.append(
                {
                    "class_id": cls,
                    "class_name": SCANNET20_CLASS_NAMES[cls],
                    "kind": "topk",
                    "k": k,
                    "hit_count": count,
                    "hit_rate": count / denom,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "target_count": target_counts[cls],
                }
            )
        for k in graph_top_ks:
            count = graph_hit_counts[(cls, k)]
            ci_low, ci_high = binomial_ci(count, denom)
            topk_rows.append(
                {
                    "class_id": cls,
                    "class_name": SCANNET20_CLASS_NAMES[cls],
                    "kind": "topk_plus_confusion_graph",
                    "k": k,
                    "hit_count": count,
                    "hit_rate": count / denom,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "target_count": target_counts[cls],
                }
            )

    variant_rows = []
    for name, summary in summaries.items():
        row = {
            "variant": name,
            "mIoU": summary["mIoU"],
            "mAcc": summary["mAcc"],
            "allAcc": summary["allAcc"],
            "delta_mIoU": summary["mIoU"] - base["mIoU"],
        }
        for cls in weak_classes:
            cname = SCANNET20_CLASS_NAMES[cls].replace(" ", "_")
            row[f"{cname}_iou"] = summary["iou"][cls]
            row[f"{cname}_delta_iou"] = summary["iou"][cls] - base["iou"][cls]
        variant_rows.append(row)

    confusion_rows = []
    base_conf = variants["base"].detach().cpu().numpy()
    for cls in weak_classes:
        denom = max(base_conf[cls].sum(), 1)
        for pred_id in np.argsort(base_conf[cls])[::-1]:
            count = int(base_conf[cls, pred_id])
            if count == 0:
                continue
            confusion_rows.append(
                {
                    "target_id": cls,
                    "target_name": SCANNET20_CLASS_NAMES[cls],
                    "pred_id": int(pred_id),
                    "pred_name": SCANNET20_CLASS_NAMES[pred_id],
                    "count": count,
                    "fraction_of_target": count / denom,
                }
            )

    top3_rows = []
    for cls in weak_classes:
        denom = max(target_counts[cls], 1)
        for pred_id in range(num_classes):
            count = top3_counts[(cls, pred_id)]
            if count == 0:
                continue
            top3_rows.append(
                {
                    "target_id": cls,
                    "target_name": SCANNET20_CLASS_NAMES[cls],
                    "top3_class_id": pred_id,
                    "top3_class_name": SCANNET20_CLASS_NAMES[pred_id],
                    "count": count,
                    "fraction_of_target": count / denom,
                }
            )

    centroid = []
    within_var = []
    for cls in range(num_classes):
        if geom_feats[cls]:
            x = torch.cat(geom_feats[cls], dim=0).float()
            c = x.mean(dim=0)
            cn = F.normalize(c, dim=0)
            sims = F.normalize(x, dim=1) @ cn
            centroid.append(c)
            within_var.append(float((1.0 - sims).mean().item()))
        else:
            centroid.append(torch.zeros_like(next(v[0] for v in geom_feats.values() if v)))
            within_var.append(float("nan"))
    centroids = torch.stack(centroid, dim=0)
    cos = cosine_distance_matrix(centroids).numpy()
    geometry_rows = []
    for cls in weak_classes:
        order = np.argsort(cos[cls])[::-1]
        rank = 0
        for other in order:
            if other == cls:
                continue
            rank += 1
            geometry_rows.append(
                {
                    "class_id": cls,
                    "class_name": SCANNET20_CLASS_NAMES[cls],
                    "other_id": int(other),
                    "other_name": SCANNET20_CLASS_NAMES[int(other)],
                    "centroid_cosine": float(cos[cls, other]),
                    "rank": rank,
                    "class_within_cos_distance": within_var[cls],
                    "other_within_cos_distance": within_var[int(other)],
                    "class_count": geom_counts[cls],
                    "other_count": geom_counts[int(other)],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "oracle_topk_hit_rates.csv",
        topk_rows,
        ["class_id", "class_name", "kind", "k", "hit_count", "hit_rate", "ci_low", "ci_high", "target_count"],
    )
    variant_fields = ["variant", "mIoU", "mAcc", "allAcc", "delta_mIoU"]
    for cls in weak_classes:
        cname = SCANNET20_CLASS_NAMES[cls].replace(" ", "_")
        variant_fields += [f"{cname}_iou", f"{cname}_delta_iou"]
    write_csv(args.output_dir / "oracle_variants.csv", variant_rows, variant_fields)
    write_csv(
        args.output_dir / "oracle_confusion_distribution.csv",
        confusion_rows,
        ["target_id", "target_name", "pred_id", "pred_name", "count", "fraction_of_target"],
    )
    write_csv(
        args.output_dir / "oracle_top3_distribution.csv",
        top3_rows,
        ["target_id", "target_name", "top3_class_id", "top3_class_name", "count", "fraction_of_target"],
    )
    write_csv(
        args.output_dir / "oracle_feature_geometry.csv",
        geometry_rows,
        [
            "class_id",
            "class_name",
            "other_id",
            "other_name",
            "centroid_cosine",
            "rank",
            "class_within_cos_distance",
            "other_within_cos_distance",
            "class_count",
            "other_count",
        ],
    )
    probe_rows = [
        {
            "pair": pair_name(probe.pair),
            "positive_class": SCANNET20_CLASS_NAMES[probe.pair[0]],
            "negative_class": SCANNET20_CLASS_NAMES[probe.pair[1]],
            "train_bal_acc": probe.train_bal_acc,
        }
        for probe in pair_probes
    ]
    write_csv(args.output_dir / "oracle_pair_probe_train.csv", probe_rows, ["pair", "positive_class", "negative_class", "train_bal_acc"])

    metadata = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "weak_classes": [SCANNET20_CLASS_NAMES[i] for i in weak_classes],
        "pairs": [pair_name(pair) for pair in pairs],
        "train": {
            "seen_batches": train_cache.seen_batches,
            "class_counts": train_cache.class_counts,
            "num_points": int(train_cache.labels.numel()),
        },
        "val": {"seen_batches": seen_val_batches, "target_counts": {SCANNET20_CLASS_NAMES[k]: int(v) for k, v in target_counts.items()}},
        "bias_history_unweighted": bias_hist_unweighted,
        "bias_history_balanced": bias_hist_balanced,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    picture = NAME_TO_ID.get("picture")
    wall = NAME_TO_ID.get("wall")
    best_variant = max((r for r in variant_rows if r["variant"] != "base"), key=lambda r: r["mIoU"])
    best_picture = max((r for r in variant_rows if r["variant"] != "base"), key=lambda r: r.get("picture_iou", -1.0))
    lines = [
        "# Oracle Actionability Analysis",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- weak classes: `{args.weak_classes}`",
        f"- class pairs: `{args.class_pairs}`",
        f"- train batches seen: {train_cache.seen_batches}",
        f"- val batches seen: {seen_val_batches}",
        "",
        "## Aggregate Variants",
        "",
        "| variant | mIoU | delta mIoU | picture IoU | picture delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(variant_rows, key=lambda r: r["mIoU"], reverse=True)[:12]:
        lines.append(
            "| {variant} | {mIoU:.4f} | {delta_mIoU:+.4f} | {picture_iou:.4f} | {picture_delta_iou:+.4f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Top-K Hit Rates",
        "",
        "| class | kind | K | hit rate | 95% CI | target count |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in topk_rows:
        if row["class_name"] in {"picture", "counter", "desk", "sink"} or row["k"] in {1, 2, 5}:
            lines.append(
                f"| {row['class_name']} | {row['kind']} | {row['k']} | {row['hit_rate']:.4f} | "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['target_count']} |"
            )
    if picture is not None and wall is not None:
        picture_wall_frac = float(base_conf[picture, wall] / max(base_conf[picture].sum(), 1))
    else:
        picture_wall_frac = float("nan")
    lines += [
        "",
        "## Key Readout Headroom",
        "",
        f"- base mIoU: `{base['mIoU']:.4f}`",
        f"- base picture IoU: `{base['iou'][picture]:.4f}`",
        f"- base picture -> wall fraction: `{picture_wall_frac:.4f}`",
        f"- best non-base mIoU variant: `{best_variant['variant']}` "
        f"(`{best_variant['mIoU']:.4f}`, delta `{best_variant['delta_mIoU']:+.4f}`)",
        f"- best non-base picture variant: `{best_picture['variant']}` "
        f"(`{best_picture['picture_iou']:.4f}`, delta `{best_picture['picture_delta_iou']:+.4f}`)",
        "",
        "## Output Files",
        "",
        "- `oracle_topk_hit_rates.csv`",
        "- `oracle_variants.csv`",
        "- `oracle_confusion_distribution.csv`",
        "- `oracle_top3_distribution.csv`",
        "- `oracle_feature_geometry.csv`",
        "- `oracle_pair_probe_train.csv`",
        "- `metadata.json`",
        "",
        "## Interpretation Notes",
        "",
        "- `oracle_topK` variants are upper bounds: if the ground-truth class is in the candidate set, prediction is replaced by the ground truth.",
        "- `oracle_graph_topK` expands top-K by the predefined confusion graph before applying the same oracle rule.",
        "- `pair_probe_top2` is a learned readout-family variant: when a configured pair appears as the top-2 classes, a train-fitted binary point-feature probe chooses between them.",
        "- `prior_alpha*`, `bias_unweighted`, and `bias_balanced` are train-derived calibration variants, not val-tuned oracle variants.",
    ]
    md_path = args.output_dir / "oracle_actionability_analysis.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {md_path}", flush=True)
    print(f"[best-variant] {best_variant}", flush=True)
    print(f"[best-picture] {best_picture}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
