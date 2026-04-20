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
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)


@dataclass
class PointBank:
    feat: torch.Tensor
    labels: torch.Tensor
    logits: torch.Tensor
    class_counts: dict[str, int]
    seen_batches: int


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieval/prototype readout baselines for the concerto_base_origin "
            "decoder-probe checkpoint. This is an eval-only method-family test: "
            "use frozen decoder features, build a train datastore/prototype set, "
            "and interpolate local evidence with the base decoder logits."
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
    parser.add_argument("--max-bank-points", type=int, default=200000)
    parser.add_argument("--max-per-class", type=int, default=10000)
    parser.add_argument("--max-proto-points-per-class", type=int, default=20000)
    parser.add_argument("--val-chunk-size", type=int, default=2048)
    parser.add_argument("--knn-ks", default="5,10,20,50")
    parser.add_argument("--knn-lambdas", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--knn-taus", default="0.05,0.1")
    parser.add_argument("--prototype-counts", default="1,4,8")
    parser.add_argument("--prototype-lambdas", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--prototype-taus", default="0.05,0.1,0.2")
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--summary-prefix", type=Path, default=Path("tools/concerto_projection_shortcut/results_retrieval_prototype_readout"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


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
    if not ids:
        raise ValueError("no weak classes provided")
    return ids


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


def append_to_bank(raw: dict, feat: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, max_points: int, max_per_class: int, num_classes: int) -> None:
    total = raw["total"]
    if total >= max_points:
        return
    valid = (labels >= 0) & (labels < num_classes)
    keep_indices = []
    for cls in range(num_classes):
        room_cls = max_per_class - raw["class_counts"][cls]
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
        raw["class_counts"][cls] += int(idx.numel())
    if keep_indices:
        keep = torch.cat(keep_indices, dim=0)
        raw["feat"].append(feat[keep].detach().cpu())
        raw["logits"].append(logits[keep].detach().cpu())
        raw["labels"].append(labels[keep].detach().cpu())
        raw["total"] += int(keep.numel())


def collect_bank(args: argparse.Namespace, model, cfg, num_classes: int) -> PointBank:
    raw = {"feat": [], "logits": [], "labels": [], "class_counts": {i: 0 for i in range(num_classes)}, "total": 0}
    loader = build_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
                break
            if raw["total"] >= args.max_bank_points:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, _ = forward_features(model, batch)
            append_to_bank(raw, feat, logits, labels, args.max_bank_points, args.max_per_class, num_classes)
            seen += 1
            if (batch_idx + 1) % 25 == 0:
                counts = " ".join(f"{SCANNET20_CLASS_NAMES[k]}={v}" for k, v in raw["class_counts"].items() if v)
                print(f"[bank] batch={batch_idx + 1} total={raw['total']} {counts}", flush=True)
    if not raw["labels"]:
        raise RuntimeError("empty retrieval bank")
    bank = PointBank(
        feat=torch.cat(raw["feat"], dim=0).float(),
        logits=torch.cat(raw["logits"], dim=0).float(),
        labels=torch.cat(raw["labels"], dim=0).long(),
        class_counts={SCANNET20_CLASS_NAMES[k]: int(v) for k, v in raw["class_counts"].items()},
        seen_batches=seen,
    )
    print(f"[bank] done points={bank.labels.numel()} seen_batches={seen}", flush=True)
    return bank


def class_iou(summary: dict, name: str) -> float:
    return float(summary["iou"][NAME_TO_ID[name]])


def picture_to_wall(conf: np.ndarray, summary: dict) -> float:
    picture = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    return float(conf[picture, wall] / max(summary["target_sum"][picture], 1.0))


def weak_mean(summary: dict, weak_classes: list[int]) -> float:
    return float(np.mean([summary["iou"][cls] for cls in weak_classes]))


def normalize_feature(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def knn_probs(query: torch.Tensor, bank_feat: torch.Tensor, bank_labels: torch.Tensor, k: int, tau: float, num_classes: int) -> torch.Tensor:
    sims = query @ bank_feat.t()
    vals, idx = sims.topk(k=min(k, bank_feat.shape[0]), dim=1)
    weights = (vals / tau).softmax(dim=1)
    labels = bank_labels[idx]
    out = torch.zeros((query.shape[0], num_classes), device=query.device, dtype=torch.float32)
    out.scatter_add_(1, labels, weights)
    return out.clamp_min(1e-12)


def entropy_lambda(base_prob: torch.Tensor, lam: float) -> torch.Tensor:
    num_classes = base_prob.shape[1]
    ent = -(base_prob.clamp_min(1e-12) * base_prob.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
    return (lam * ent / math.log(num_classes)).clamp(0.0, lam)


def class_score_from_prototypes(query: torch.Tensor, proto_feat: torch.Tensor, proto_labels: torch.Tensor, num_classes: int, tau: float) -> torch.Tensor:
    sim = query @ proto_feat.t()
    scores = torch.full((query.shape[0], num_classes), -1e4, device=query.device, dtype=torch.float32)
    for cls in range(num_classes):
        mask = proto_labels == cls
        if mask.any():
            scores[:, cls] = sim[:, mask].max(dim=1).values
    return (scores / tau).softmax(dim=1).clamp_min(1e-12)


def make_single_prototypes(bank: PointBank, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    feat = normalize_feature(bank.feat)
    protos = []
    labels = []
    global_mean = normalize_feature(feat.mean(dim=0, keepdim=True))[0]
    for cls in range(num_classes):
        mask = bank.labels == cls
        if mask.any():
            proto = normalize_feature(feat[mask].mean(dim=0, keepdim=True))[0]
        else:
            proto = global_mean
        protos.append(proto)
        labels.append(cls)
    return torch.stack(protos, dim=0), torch.tensor(labels, dtype=torch.long)


def kmeans_class(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    if x.shape[0] <= k:
        return x.clone()
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g, device=x.device)[:k]
    centers = x[perm].clone()
    for _ in range(iters):
        assign = (x @ centers.t()).argmax(dim=1)
        new_centers = []
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers.append(normalize_feature(x[mask].mean(dim=0, keepdim=True))[0])
            else:
                new_centers.append(centers[j])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-4, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    return centers


def make_multi_prototypes(bank: PointBank, num_classes: int, n_proto: int, max_per_class: int, iters: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    feat = normalize_feature(bank.feat)
    protos = []
    labels = []
    for cls in range(num_classes):
        idx = (bank.labels == cls).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        if idx.numel() > max_per_class:
            idx = idx[:max_per_class]
        x = feat[idx].cuda()
        centers = kmeans_class(x, min(n_proto, x.shape[0]), iters, seed + cls).cpu()
        protos.append(centers)
        labels.extend([cls] * centers.shape[0])
    return torch.cat(protos, dim=0), torch.tensor(labels, dtype=torch.long)


def init_confusions(names: list[str], num_classes: int) -> dict[str, torch.Tensor]:
    return {name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda") for name in names}


def variant_names(knn_ks: list[int], knn_lambdas: list[float], knn_taus: list[float], proto_counts: list[int], proto_lambdas: list[float], proto_taus: list[float]) -> list[str]:
    names = ["base"]
    for k in knn_ks:
        for tau in knn_taus:
            for lam in knn_lambdas:
                names.append(f"knn_k{k}_tau{tau:g}_lam{lam:g}".replace(".", "p"))
                names.append(f"knn_adapt_k{k}_tau{tau:g}_lam{lam:g}".replace(".", "p"))
    for n_proto in proto_counts:
        for tau in proto_taus:
            for lam in proto_lambdas:
                prefix = "proto" if n_proto == 1 else f"multiproto{n_proto}"
                names.append(f"{prefix}_tau{tau:g}_lam{lam:g}".replace(".", "p"))
                names.append(f"{prefix}_adapt_tau{tau:g}_lam{lam:g}".replace(".", "p"))
    return names


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{row[key]:.8f}" if isinstance(row.get(key), float) else row.get(key, "")) for key in fields})


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.config = str((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    cfg = load_config(Path(args.config))
    names = list(cfg.data.names)
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    weak_classes = parse_names(args.weak_classes)
    knn_ks = parse_int_list(args.knn_ks)
    knn_lambdas = parse_float_list(args.knn_lambdas)
    knn_taus = parse_float_list(args.knn_taus)
    proto_counts = parse_int_list(args.prototype_counts)
    proto_lambdas = parse_float_list(args.prototype_lambdas)
    proto_taus = parse_float_list(args.prototype_taus)

    if args.dry_run:
        loader = build_loader(cfg, args.val_split, args.data_root, 1, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] variants={len(variant_names(knn_ks, knn_lambdas, knn_taus, proto_counts, proto_lambdas, proto_taus))}", flush=True)
        return 0

    model = build_model(cfg, args.weight)
    bank = collect_bank(args, model, cfg, num_classes)
    bank_feat = normalize_feature(bank.feat).cuda()
    bank_labels = bank.labels.cuda()

    proto_sets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    if 1 in proto_counts:
        proto_sets[1] = make_single_prototypes(bank, num_classes)
    for n_proto in proto_counts:
        if n_proto == 1:
            continue
        proto_sets[n_proto] = make_multi_prototypes(
            bank,
            num_classes,
            n_proto=n_proto,
            max_per_class=args.max_proto_points_per_class,
            iters=args.kmeans_iters,
            seed=args.seed,
        )
        print(f"[proto] n={n_proto} total_prototypes={proto_sets[n_proto][0].shape[0]}", flush=True)
    proto_sets = {k: (v[0].cuda(), v[1].cuda()) for k, v in proto_sets.items()}

    names_all = variant_names(knn_ks, knn_lambdas, knn_taus, proto_counts, proto_lambdas, proto_taus)
    confusions = init_confusions(names_all, num_classes)
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
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
            feat_e = normalize_feature(feat_e[valid])
            logits_e = logits_e[valid].float()
            labels_e = labels_e[valid].long()
            for start in range(0, labels_e.numel(), args.val_chunk_size):
                end = min(start + args.val_chunk_size, labels_e.numel())
                feat_b = feat_e[start:end]
                logits_b = logits_e[start:end]
                labels_b = labels_e[start:end]
                base_prob = logits_b.softmax(dim=1).clamp_min(1e-12)
                update_confusion(confusions["base"], base_prob.argmax(dim=1), labels_b, num_classes, ignore_index)
                for k in knn_ks:
                    for tau in knn_taus:
                        p_knn = knn_probs(feat_b, bank_feat, bank_labels, k, tau, num_classes)
                        for lam in knn_lambdas:
                            fixed = (1.0 - lam) * base_prob + lam * p_knn
                            name = f"knn_k{k}_tau{tau:g}_lam{lam:g}".replace(".", "p")
                            update_confusion(confusions[name], fixed.argmax(dim=1), labels_b, num_classes, ignore_index)
                            lam_i = entropy_lambda(base_prob, lam)
                            adapt = (1.0 - lam_i) * base_prob + lam_i * p_knn
                            name = f"knn_adapt_k{k}_tau{tau:g}_lam{lam:g}".replace(".", "p")
                            update_confusion(confusions[name], adapt.argmax(dim=1), labels_b, num_classes, ignore_index)
                for n_proto in proto_counts:
                    proto_feat, proto_labels = proto_sets[n_proto]
                    for tau in proto_taus:
                        p_proto = class_score_from_prototypes(feat_b, proto_feat, proto_labels, num_classes, tau)
                        prefix = "proto" if n_proto == 1 else f"multiproto{n_proto}"
                        for lam in proto_lambdas:
                            fixed = (1.0 - lam) * base_prob + lam * p_proto
                            name = f"{prefix}_tau{tau:g}_lam{lam:g}".replace(".", "p")
                            update_confusion(confusions[name], fixed.argmax(dim=1), labels_b, num_classes, ignore_index)
                            lam_i = entropy_lambda(base_prob, lam)
                            adapt = (1.0 - lam_i) * base_prob + lam_i * p_proto
                            name = f"{prefix}_adapt_tau{tau:g}_lam{lam:g}".replace(".", "p")
                            update_confusion(confusions[name], adapt.argmax(dim=1), labels_b, num_classes, ignore_index)
            seen += 1
            if (batch_idx + 1) % 25 == 0:
                base_sum = summarize_confusion(confusions["base"].detach().cpu().numpy(), names)
                print(f"[val] batch={batch_idx + 1} base_mIoU={base_sum['mIoU']:.4f}", flush=True)

    summaries = {name: summarize_confusion(conf.detach().cpu().numpy(), names) for name, conf in confusions.items()}
    base = summaries["base"]
    rows = []
    class_rows = []
    for name, summary in summaries.items():
        conf_np = confusions[name].detach().cpu().numpy()
        row = {
            "variant": name,
            "mIoU": summary["mIoU"],
            "delta_mIoU": summary["mIoU"] - base["mIoU"],
            "mAcc": summary["mAcc"],
            "allAcc": summary["allAcc"],
            "weak_mean_iou": weak_mean(summary, weak_classes),
            "delta_weak_mean_iou": weak_mean(summary, weak_classes) - weak_mean(base, weak_classes),
            "picture_iou": class_iou(summary, "picture"),
            "delta_picture_iou": class_iou(summary, "picture") - class_iou(base, "picture"),
            "picture_to_wall_frac": picture_to_wall(conf_np, summary),
            "delta_picture_to_wall_frac": picture_to_wall(conf_np, summary) - picture_to_wall(confusions["base"].detach().cpu().numpy(), base),
            "counter_iou": class_iou(summary, "counter"),
            "desk_iou": class_iou(summary, "desk"),
            "sink_iou": class_iou(summary, "sink"),
            "cabinet_iou": class_iou(summary, "cabinet"),
            "door_iou": class_iou(summary, "door"),
            "shower_curtain_iou": class_iou(summary, "shower curtain"),
        }
        rows.append(row)
        for cls, cls_name in enumerate(SCANNET20_CLASS_NAMES):
            class_rows.append(
                {
                    "variant": name,
                    "class_id": cls,
                    "class_name": cls_name,
                    "iou": float(summary["iou"][cls]),
                    "delta_iou": float(summary["iou"][cls] - base["iou"][cls]),
                    "acc": float(summary["acc"][cls]),
                    "target_sum": float(summary["target_sum"][cls]),
                    "pred_sum": float(summary["pred_sum"][cls]),
                }
            )
    rows.sort(key=lambda r: (r["mIoU"], r["picture_iou"]), reverse=True)
    safe_picture = [
        r for r in rows if r["mIoU"] >= base["mIoU"] - 0.002 and r["variant"] != "base"
    ]
    best_safe_picture = max(safe_picture, key=lambda r: (r["picture_iou"], r["mIoU"])) if safe_picture else None
    best_miou = rows[0]
    best_picture = max([r for r in rows if r["variant"] != "base"], key=lambda r: (r["picture_iou"], r["mIoU"]))

    fields = [
        "variant",
        "mIoU",
        "delta_mIoU",
        "mAcc",
        "allAcc",
        "weak_mean_iou",
        "delta_weak_mean_iou",
        "picture_iou",
        "delta_picture_iou",
        "picture_to_wall_frac",
        "delta_picture_to_wall_frac",
        "counter_iou",
        "desk_iou",
        "sink_iou",
        "cabinet_iou",
        "door_iou",
        "shower_curtain_iou",
    ]
    write_csv(args.output_dir / "retrieval_variants.csv", rows, fields)
    write_csv(args.summary_prefix.with_suffix(".csv"), rows, fields)
    write_csv(
        args.output_dir / "retrieval_class_metrics.csv",
        class_rows,
        ["variant", "class_id", "class_name", "iou", "delta_iou", "acc", "target_sum", "pred_sum"],
    )
    metadata = {
        "config": str(args.config),
        "weight": str(args.weight),
        "data_root": str(args.data_root),
        "seen_val_batches": seen,
        "bank_points": int(bank.labels.numel()),
        "bank_class_counts": bank.class_counts,
        "max_train_batches": args.max_train_batches,
        "max_bank_points": args.max_bank_points,
        "max_per_class": args.max_per_class,
        "knn_ks": knn_ks,
        "knn_lambdas": knn_lambdas,
        "knn_taus": knn_taus,
        "prototype_counts": proto_counts,
        "prototype_lambdas": proto_lambdas,
        "prototype_taus": proto_taus,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def fmt(row: dict | None) -> str:
        if row is None:
            return "n/a"
        return (
            f"`{row['variant']}` mIoU={row['mIoU']:.4f} "
            f"(Δ{row['delta_mIoU']:+.4f}), picture={row['picture_iou']:.4f} "
            f"(Δ{row['delta_picture_iou']:+.4f}), p->wall={row['picture_to_wall_frac']:.4f} "
            f"(Δ{row['delta_picture_to_wall_frac']:+.4f})"
        )

    md = [
        "# Retrieval / Prototype Readout",
        "",
        "Eval-only frozen decoder-feature baselines for the `concerto_base_origin` decoder probe. "
        "This tests whether local nonparametric evidence can recover the oracle/actionability headroom "
        "that fixed-logit rerankers and pair-emphasis adapters failed to recover.",
        "",
        "## Setup",
        "",
        f"- Config: `{args.config}`",
        f"- Weight: `{args.weight}`",
        f"- Bank points: `{bank.labels.numel()}`",
        f"- Seen val batches: `{seen}`",
        f"- Bank caps: max_train_batches `{args.max_train_batches}`, max_bank_points `{args.max_bank_points}`, max_per_class `{args.max_per_class}`",
        "",
        "## Headline",
        "",
        f"- Base: {fmt(next(r for r in rows if r['variant'] == 'base'))}",
        f"- Best mIoU: {fmt(best_miou)}",
        f"- Best picture: {fmt(best_picture)}",
        f"- Best safe picture (mIoU >= base - 0.002): {fmt(best_safe_picture)}",
        "",
        "## Top Variants",
        "",
        "| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows[:20], start=1):
        md.append(
            f"| {rank} | `{row['variant']}` | {row['mIoU']:.4f} | {row['delta_mIoU']:+.4f} | "
            f"{row['weak_mean_iou']:.4f} | {row['delta_weak_mean_iou']:+.4f} | "
            f"{row['picture_iou']:.4f} | {row['delta_picture_iou']:+.4f} | "
            f"{row['picture_to_wall_frac']:.4f} | {row['delta_picture_to_wall_frac']:+.4f} |"
        )
    md.extend(
        [
            "",
            "## Interpretation Gate",
            "",
            "- Promising if a variant improves mIoU by >= +0.003, or improves picture by >= +0.02 while keeping mIoU within -0.002.",
            "- If no variant passes that gate, retrieval/prototype readout is treated as no-go under this protocol and the next line should be LP-FT / class-safe LoRA.",
            "",
            "## Files",
            "",
            f"- Variant CSV: `{args.output_dir / 'retrieval_variants.csv'}`",
            f"- Class CSV: `{args.output_dir / 'retrieval_class_metrics.csv'}`",
            f"- Metadata: `{args.output_dir / 'metadata.json'}`",
        ]
    )
    text = "\n".join(md) + "\n"
    (args.output_dir / "retrieval_prototype_readout.md").write_text(text, encoding="utf-8")
    args.summary_prefix.with_suffix(".md").write_text(text, encoding="utf-8")
    print(f"[done] wrote {args.summary_prefix.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
