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
    parse_pairs,
)
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    build_loader,
    build_model,
    eval_tensors,
    forward_features,
    load_config,
    move_to_cuda,
    picture_to_wall,
    summarize_confusion,
    update_confusion,
    weak_mean,
)


@dataclass
class Cache:
    feat: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    seen_batches: int
    class_counts: dict[str, int]


@dataclass
class SubCenters:
    feat: torch.Tensor
    labels: torch.Tensor
    local_ids: torch.Tensor
    class_to_confusion: dict[int, int]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Latent subgroup diagnostic and targeted sub-center decoder pilot "
            "for the concerto_base_origin decoder-probe checkpoint."
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
    parser.add_argument("--max-train-batches", type=int, default=420)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-points", type=int, default=1200000)
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--heldout-mod", type=int, default=5)
    parser.add_argument("--heldout-remainder", type=int, default=0)
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall",
    )
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument(
        "--subgroups",
        default="picture:4,wall:8,counter:4,cabinet:8,desk:4,table:8,sink:4,door:4,shower curtain:4",
    )
    parser.add_argument("--default-k", type=int, default=1)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--score-taus", default="0.05,0.1,0.2")
    parser.add_argument("--mix-lambdas", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--logsumexp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_latent_subgroup_decoder"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_names(text: str) -> list[int]:
    ids = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown class: {name}")
        ids.append(NAME_TO_ID[name])
    return ids


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_subgroups(text: str, default_k: int, num_classes: int) -> dict[int, int]:
    out = {i: default_k for i in range(num_classes)}
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, k = raw.split(":", 1)
        name = name.strip()
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown subgroup class: {name}")
        out[NAME_TO_ID[name]] = int(k)
    return out


def split_dataset_cfg(cfg, split: str, data_root: Path):
    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg.split = split
    ds_cfg.data_root = str(data_root)
    ds_cfg.test_mode = False
    return ds_cfg


def build_split_loader(cfg, split: str, data_root: Path, batch_size: int, num_worker: int):
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


def normalize_feature(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def append_cache(raw: dict, feat: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, max_points: int, max_per_class: int, num_classes: int) -> None:
    if raw["total"] >= max_points:
        return
    valid = (labels >= 0) & (labels < num_classes)
    keep_indices = []
    current = raw["total"]
    for cls in range(num_classes):
        room_cls = max_per_class - raw["class_counts"][cls]
        if room_cls <= 0:
            continue
        idx = ((labels == cls) & valid).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        room_total = max_points - current - sum(x.numel() for x in keep_indices)
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


def finalize_cache(raw: dict, seen: int) -> Cache:
    if not raw["labels"]:
        raise RuntimeError("empty cache")
    return Cache(
        feat=torch.cat(raw["feat"], dim=0).float(),
        logits=torch.cat(raw["logits"], dim=0).float(),
        labels=torch.cat(raw["labels"], dim=0).long(),
        seen_batches=seen,
        class_counts={SCANNET20_CLASS_NAMES[k]: int(v) for k, v in raw["class_counts"].items()},
    )


def collect_train_heldout(args: argparse.Namespace, model, cfg, num_classes: int) -> tuple[Cache, Cache]:
    def empty_raw():
        return {"feat": [], "logits": [], "labels": [], "class_counts": {i: 0 for i in range(num_classes)}, "total": 0}

    train_raw = empty_raw()
    held_raw = empty_raw()
    loader = build_split_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
                break
            if train_raw["total"] >= args.max_points and held_raw["total"] >= args.max_points:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, _ = forward_features(model, batch)
            is_heldout = (batch_idx % args.heldout_mod) == args.heldout_remainder
            raw = held_raw if is_heldout else train_raw
            append_cache(raw, feat, logits, labels, args.max_points, args.max_per_class, num_classes)
            seen += 1
            if (batch_idx + 1) % 50 == 0:
                print(f"[collect] batch={batch_idx+1} train={train_raw['total']} heldout={held_raw['total']}", flush=True)
    return finalize_cache(train_raw, seen), finalize_cache(held_raw, seen)


def kmeans_class(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    x = normalize_feature(x)
    if x.shape[0] <= k:
        return x.clone()
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    centers = x[torch.randperm(x.shape[0], generator=gen, device=x.device)[:k]].clone()
    for _ in range(iters):
        assign = (x @ centers.t()).argmax(dim=1)
        new = []
        for j in range(k):
            mask = assign == j
            new.append(normalize_feature(x[mask].mean(dim=0, keepdim=True))[0] if mask.any() else centers[j])
        new = torch.stack(new, dim=0)
        if torch.allclose(new, centers, atol=1e-4, rtol=1e-4):
            centers = new
            break
        centers = new
    return centers


def make_subcenters(train: Cache, num_classes: int, k_by_class: dict[int, int], iters: int, seed: int, pairs: list[tuple[int, int]]) -> SubCenters:
    feat = normalize_feature(train.feat)
    centers = []
    labels = []
    local_ids = []
    for cls in range(num_classes):
        idx = (train.labels == cls).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        k = min(k_by_class.get(cls, 1), idx.numel())
        cls_centers = kmeans_class(feat[idx].cuda(), k, iters, seed + cls).cpu()
        centers.append(cls_centers)
        labels.extend([cls] * cls_centers.shape[0])
        local_ids.extend(list(range(cls_centers.shape[0])))
    class_to_confusion = {}
    for a, b in pairs:
        class_to_confusion[a] = b
    return SubCenters(
        feat=torch.cat(centers, dim=0).float(),
        labels=torch.tensor(labels, dtype=torch.long),
        local_ids=torch.tensor(local_ids, dtype=torch.long),
        class_to_confusion=class_to_confusion,
    )


def subcenter_probs(query: torch.Tensor, centers: SubCenters, num_classes: int, tau: float, logsumexp: bool) -> torch.Tensor:
    query = normalize_feature(query)
    cfeat = centers.feat.to(query.device)
    clabel = centers.labels.to(query.device)
    sim = query @ cfeat.t()
    scores = torch.full((query.shape[0], num_classes), -1e4, device=query.device)
    for cls in range(num_classes):
        mask = clabel == cls
        if not mask.any():
            continue
        cls_sim = sim[:, mask] / tau
        scores[:, cls] = torch.logsumexp(cls_sim, dim=1) if logsumexp else cls_sim.max(dim=1).values
    return scores.softmax(dim=1).clamp_min(1e-12)


def assign_same_class(feat: torch.Tensor, labels: torch.Tensor, centers: SubCenters) -> torch.Tensor:
    feat = normalize_feature(feat)
    cfeat = centers.feat.to(feat.device)
    clabel = centers.labels.to(feat.device)
    local = centers.local_ids.to(feat.device)
    out = torch.full_like(labels, -1)
    for cls in labels.unique().tolist():
        mask = labels == cls
        cmask = clabel == cls
        if not cmask.any():
            continue
        sim = feat[mask] @ cfeat[cmask].t()
        out[mask] = local[cmask][sim.argmax(dim=1)]
    return out


def init_confusions(names: list[str], num_classes: int) -> dict[str, torch.Tensor]:
    return {name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda") for name in names}


def eval_cache_for_diagnostic(cache: Cache, domain: str, centers: SubCenters, pairs: list[tuple[int, int]]) -> list[dict]:
    rows = []
    feat = cache.feat.cuda()
    logits = cache.logits.cuda()
    labels = cache.labels.cuda()
    pred = logits.argmax(dim=1)
    assign = assign_same_class(feat, labels, centers)
    for target, counterpart in pairs:
        mask_t = labels == target
        if not mask_t.any():
            continue
        for cluster_id in assign[mask_t].unique().tolist():
            if cluster_id < 0:
                continue
            mask = mask_t & (assign == cluster_id)
            n = int(mask.sum().item())
            if n == 0:
                continue
            target_top1 = float((pred[mask] == target).float().mean().item())
            confusion_rate = float((pred[mask] == counterpart).float().mean().item())
            margin = float((logits[mask, target] - logits[mask, counterpart]).mean().item())
            rows.append(
                {
                    "domain": domain,
                    "class_id": target,
                    "class_name": SCANNET20_CLASS_NAMES[target],
                    "counterpart_id": counterpart,
                    "counterpart_name": SCANNET20_CLASS_NAMES[counterpart],
                    "cluster": int(cluster_id),
                    "count": n,
                    "target_top1": target_top1,
                    "counterpart_top1": confusion_rate,
                    "mean_target_minus_counterpart_logit": margin,
                }
            )
    return rows


def update_cluster_diag(rows: list[dict], domain: str, feat: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, centers: SubCenters, pairs: list[tuple[int, int]]):
    pred = logits.argmax(dim=1)
    assign = assign_same_class(feat, labels, centers)
    for target, counterpart in pairs:
        mask_t = labels == target
        if not mask_t.any():
            continue
        for cluster_id in assign[mask_t].unique().tolist():
            if cluster_id < 0:
                continue
            mask = mask_t & (assign == cluster_id)
            n = int(mask.sum().item())
            rows.append(
                {
                    "domain": domain,
                    "class_id": target,
                    "class_name": SCANNET20_CLASS_NAMES[target],
                    "counterpart_id": counterpart,
                    "counterpart_name": SCANNET20_CLASS_NAMES[counterpart],
                    "cluster": int(cluster_id),
                    "count": n,
                    "target_top1": float((pred[mask] == target).float().mean().item()),
                    "counterpart_top1": float((pred[mask] == counterpart).float().mean().item()),
                    "mean_target_minus_counterpart_logit": float((logits[mask, target] - logits[mask, counterpart]).mean().item()),
                }
            )


def entropy_lambda(base_prob: torch.Tensor, lam: float) -> torch.Tensor:
    ncls = base_prob.shape[1]
    ent = -(base_prob.clamp_min(1e-12) * base_prob.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
    return (lam * ent / math.log(ncls)).clamp(0.0, lam)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{row[k]:.8f}" if isinstance(row.get(k), float) else row.get(k, "")) for k in fields})


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
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    weak_classes = parse_names(args.weak_classes)
    pairs = parse_pairs(args.class_pairs)
    score_taus = parse_float_list(args.score_taus)
    mix_lambdas = parse_float_list(args.mix_lambdas)
    k_by_class = parse_subgroups(args.subgroups, args.default_k, num_classes)
    if args.dry_run:
        loader = build_split_loader(cfg, args.val_split, args.data_root, 1, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] pairs={[(SCANNET20_CLASS_NAMES[a], SCANNET20_CLASS_NAMES[b]) for a,b in pairs]}", flush=True)
        return 0

    model = build_model(cfg, args.weight)
    train, heldout = collect_train_heldout(args, model, cfg, num_classes)
    centers = make_subcenters(train, num_classes, k_by_class, args.kmeans_iters, args.seed, pairs)
    print(f"[centers] total={centers.labels.numel()}", flush=True)
    train_diag = eval_cache_for_diagnostic(train, "train", centers, pairs)
    held_diag = eval_cache_for_diagnostic(heldout, "heldout", centers, pairs)

    variant_names = ["base"]
    for tau in score_taus:
        for lam in mix_lambdas:
            variant_names.append(f"subcenter_tau{tau:g}_lam{lam:g}".replace(".", "p"))
            variant_names.append(f"subcenter_adapt_tau{tau:g}_lam{lam:g}".replace(".", "p"))
    confusions = init_confusions(variant_names, num_classes)
    val_diag = []
    loader = build_split_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
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
            feat_e = feat_e[valid].float()
            logits_e = logits_e[valid].float()
            labels_e = labels_e[valid].long()
            update_cluster_diag(val_diag, "val", feat_e, logits_e, labels_e, centers, pairs)
            base_prob = logits_e.softmax(dim=1).clamp_min(1e-12)
            update_confusion(confusions["base"], base_prob.argmax(dim=1), labels_e, num_classes, ignore_index)
            for tau in score_taus:
                p_sub = subcenter_probs(feat_e, centers, num_classes, tau, args.logsumexp)
                for lam in mix_lambdas:
                    fixed = (1.0 - lam) * base_prob + lam * p_sub
                    name = f"subcenter_tau{tau:g}_lam{lam:g}".replace(".", "p")
                    update_confusion(confusions[name], fixed.argmax(dim=1), labels_e, num_classes, ignore_index)
                    lam_i = entropy_lambda(base_prob, lam)
                    adapt = (1.0 - lam_i) * base_prob + lam_i * p_sub
                    name = f"subcenter_adapt_tau{tau:g}_lam{lam:g}".replace(".", "p")
                    update_confusion(confusions[name], adapt.argmax(dim=1), labels_e, num_classes, ignore_index)
            seen += 1
            if seen % 25 == 0:
                base_sum = summarize_confusion(confusions["base"].detach().cpu().numpy(), list(cfg.data.names))
                print(f"[val] batches={seen} base_mIoU={base_sum['mIoU']:.4f}", flush=True)

    names = list(cfg.data.names)
    summaries = {name: summarize_confusion(conf.detach().cpu().numpy(), names) for name, conf in confusions.items()}
    base = summaries["base"]
    base_conf = confusions["base"].detach().cpu().numpy()
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
            "picture_iou": float(summary["iou"][NAME_TO_ID["picture"]]),
            "delta_picture_iou": float(summary["iou"][NAME_TO_ID["picture"]] - base["iou"][NAME_TO_ID["picture"]]),
            "picture_to_wall_frac": picture_to_wall(conf_np, summary),
            "delta_picture_to_wall_frac": picture_to_wall(conf_np, summary) - picture_to_wall(base_conf, base),
            "counter_iou": float(summary["iou"][NAME_TO_ID["counter"]]),
            "desk_iou": float(summary["iou"][NAME_TO_ID["desk"]]),
            "sink_iou": float(summary["iou"][NAME_TO_ID["sink"]]),
            "cabinet_iou": float(summary["iou"][NAME_TO_ID["cabinet"]]),
            "door_iou": float(summary["iou"][NAME_TO_ID["door"]]),
            "shower_curtain_iou": float(summary["iou"][NAME_TO_ID["shower curtain"]]),
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
    write_csv(args.output_dir / "latent_subgroup_variants.csv", rows, fields)
    write_csv(args.summary_prefix.with_suffix(".csv"), rows, fields)
    write_csv(
        args.output_dir / "latent_subgroup_class_metrics.csv",
        class_rows,
        ["variant", "class_id", "class_name", "iou", "delta_iou", "acc", "target_sum", "pred_sum"],
    )
    diag_rows = train_diag + held_diag + val_diag
    write_csv(
        args.output_dir / "latent_subgroup_cluster_diagnostic.csv",
        diag_rows,
        [
            "domain",
            "class_id",
            "class_name",
            "counterpart_id",
            "counterpart_name",
            "cluster",
            "count",
            "target_top1",
            "counterpart_top1",
            "mean_target_minus_counterpart_logit",
        ],
    )
    meta = {
        "config": str(args.config),
        "weight": str(args.weight),
        "data_root": str(args.data_root),
        "train_points": int(train.labels.numel()),
        "heldout_points": int(heldout.labels.numel()),
        "seen_val_batches": seen,
        "train_class_counts": train.class_counts,
        "heldout_class_counts": heldout.class_counts,
        "subgroups": {SCANNET20_CLASS_NAMES[k]: int(v) for k, v in k_by_class.items() if v != args.default_k},
        "score_taus": score_taus,
        "mix_lambdas": mix_lambdas,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_row = next(r for r in rows if r["variant"] == "base")
    best_miou = rows[0]
    best_picture = max([r for r in rows if r["variant"] != "base"], key=lambda r: (r["picture_iou"], r["mIoU"]))
    safe = [r for r in rows if r["variant"] != "base" and r["mIoU"] >= base_row["mIoU"] - 0.002]
    best_safe_picture = max(safe, key=lambda r: (r["picture_iou"], r["mIoU"])) if safe else None

    def fmt(row: dict | None) -> str:
        if row is None:
            return "n/a"
        return (
            f"`{row['variant']}` mIoU={row['mIoU']:.4f} "
            f"(Δ{row['delta_mIoU']:+.4f}), picture={row['picture_iou']:.4f} "
            f"(Δ{row['delta_picture_iou']:+.4f}), p->wall={row['picture_to_wall_frac']:.4f} "
            f"(Δ{row['delta_picture_to_wall_frac']:+.4f})"
        )

    def top_cluster_lines() -> list[str]:
        out = []
        rows_sorted = sorted(
            [r for r in diag_rows if r["domain"] == "val"],
            key=lambda r: (r["class_name"] != "picture", -r["count"], r["counterpart_top1"]),
        )
        for r in rows_sorted[:16]:
            out.append(
                f"| {r['domain']} | {r['class_name']} | {r['counterpart_name']} | {r['cluster']} | "
                f"{r['count']} | {r['target_top1']:.4f} | {r['counterpart_top1']:.4f} | "
                f"{r['mean_target_minus_counterpart_logit']:.4f} |"
            )
        return out

    md = [
        "# Latent Subgroup Decoder",
        "",
        "Latent subgroup diagnostic and targeted sub-center readout pilot for the "
        "`concerto_base_origin` decoder-probe checkpoint.",
        "",
        "## Setup",
        "",
        f"- Config: `{args.config}`",
        f"- Weight: `{args.weight}`",
        f"- Train points: `{train.labels.numel()}`",
        f"- Heldout points: `{heldout.labels.numel()}`",
        f"- Seen val batches: `{seen}`",
        "",
        "## Headline",
        "",
        f"- Base: {fmt(base_row)}",
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
            "## Val Cluster Diagnostic",
            "",
            "| domain | class | counterpart | cluster | count | target top1 | counterpart top1 | target-counterpart margin |",
            "|---|---|---|---:|---:|---:|---:|---:|",
            *top_cluster_lines(),
            "",
            "## Interpretation Gate",
            "",
            "- Promising if a variant improves mIoU by >= +0.003, or improves `picture` by >= +0.02 while keeping mIoU within -0.002.",
            "- If no variant passes, targeted latent-subgroup readout is not recovering the oracle/actionability headroom under this offline protocol.",
            "",
            "## Files",
            "",
            f"- Variant CSV: `{args.output_dir / 'latent_subgroup_variants.csv'}`",
            f"- Class CSV: `{args.output_dir / 'latent_subgroup_class_metrics.csv'}`",
            f"- Cluster diagnostic CSV: `{args.output_dir / 'latent_subgroup_cluster_diagnostic.csv'}`",
            f"- Metadata: `{args.output_dir / 'metadata.json'}`",
        ]
    )
    text = "\n".join(md) + "\n"
    (args.output_dir / "latent_subgroup_decoder.md").write_text(text, encoding="utf-8")
    args.summary_prefix.with_suffix(".md").write_text(text, encoding="utf-8")
    print(f"[done] wrote {args.summary_prefix.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
