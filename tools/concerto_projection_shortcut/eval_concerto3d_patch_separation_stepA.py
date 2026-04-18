#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from torch.utils.data import DataLoader


SCANNET20_CLASS_NAMES = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step A companion: measure picture/wall separability from Concerto "
            "3D encoder patch-pooled features on ScanNet image-point data."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="pretrain-concerto-v1m1-2-large-video")
    parser.add_argument(
        "--weight",
        type=Path,
        default=Path("data/weights/concerto/pretrain-concerto-v1m1-2-large-video.pth"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/concerto_scannet_imagepoint_absmeta"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--no-segment-alias",
        action="store_true",
        help=(
            "Do not create lightweight segment.npy -> segment20.npy aliases. "
            "Concerto image-point ScanNet stores segment20.npy, while "
            "DefaultImagePointDataset only loads segment.npy."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=128)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--max-sem-per-class", type=int, default=8000)
    parser.add_argument("--min-points-per-patch", type=int, default=4)
    parser.add_argument("--majority-threshold", type=float, default=0.6)
    parser.add_argument("--picture-class", type=int, default=10)
    parser.add_argument("--wall-class", type=int, default=0)
    parser.add_argument("--logreg-steps", type=int, default=600)
    parser.add_argument("--logreg-lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_repo_on_path(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_cfg(repo_root: Path, config_name: str):
    ensure_repo_on_path(repo_root)
    from pointcept.utils.config import Config

    return Config.fromfile(str(repo_root / "configs" / "concerto" / f"{config_name}.py"))


def ensure_segment_aliases(data_root: Path, splits: list[str]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for split in splits:
        split_path = data_root / "splits" / f"{split}.json"
        if not split_path.is_file():
            continue
        records = json.loads(split_path.read_text(encoding="utf-8"))
        made = 0
        for record in records.values():
            point_root = Path(record["pointclouds"])
            alias = point_root / "segment.npy"
            source = point_root / "segment20.npy"
            if alias.exists() or not source.is_file():
                continue
            try:
                os.symlink("segment20.npy", alias)
                made += 1
            except FileExistsError:
                pass
        summary[split] = made
    return summary


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scannet_dataset_cfg(cfg, data_root: Path, split: str):
    transform = copy.deepcopy(cfg.transform)
    for item in transform:
        if item.get("type") == "MultiViewGenerator":
            view_keys = list(
                item.get(
                    "view_keys",
                    ("coord", "origin_coord", "color", "normal", "correspondence"),
                )
            )
            if "segment" not in view_keys:
                view_keys.append("segment")
            item["view_keys"] = tuple(view_keys)
        if item.get("type") == "Collect":
            keys = list(item["keys"])
            if "global_segment" not in keys:
                keys.append("global_segment")
            item["keys"] = tuple(keys)
    return dict(
        type="DefaultImagePointDataset",
        crop_h=int(cfg.crop_h),
        crop_w=int(cfg.crop_w),
        patch_size=int(cfg.patch_size),
        split=[split],
        data_root=str(data_root.resolve()),
        transform=transform,
        test_mode=False,
        loop=1,
    )


def build_loader(cfg, data_root: Path, split: str, batch_size: int, num_worker: int):
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn

    dataset = build_dataset(make_scannet_dataset_cfg(cfg, data_root, split))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        collate_fn=lambda batch: point_collate_fn(batch, mix_prob=0),
    )


def load_weight(model, weight_path: Path) -> None:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    info = model.load_state_dict(cleaned, strict=False)
    print(
        "[load_weight]",
        f"missing={len(info.missing_keys)}",
        f"unexpected={len(info.unexpected_keys)}",
        flush=True,
    )


def move_batch_to_cuda(batch: dict) -> dict:
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.cuda(non_blocking=True)
    return batch


def offset2batch(offset: torch.Tensor) -> torch.Tensor:
    counts = torch.diff(torch.cat([offset.new_zeros(1), offset]))
    return torch.repeat_interleave(torch.arange(len(counts), device=offset.device), counts)


@torch.no_grad()
def extract_batch_patch_features(model, batch: dict, min_points: int, majority_threshold: float):
    from pointcept.models.utils.structure import Point

    global_point = Point(
        feat=batch["global_feat"],
        coord=batch["global_coord"],
        origin_coord=batch["global_origin_coord"],
        offset=batch["global_offset"],
        grid_size=batch["grid_size"][0],
    )
    point = model.student.backbone(global_point)
    point = model.up_cast(point)
    point = model.up_cast(point, upcast_level=model.enc2d_upcast_level - model.up_cast_level)
    to_feature = model.pool_corr(point, batch["global_correspondence"])

    data_dict_global_offset = torch.cat([torch.tensor([0], device=point.feat.device), to_feature["offset"]], dim=0)
    enc2d_count = (
        data_dict_global_offset[1 : len(data_dict_global_offset) : model.num_global_view]
        - data_dict_global_offset[0 : len(data_dict_global_offset) - 1 : model.num_global_view]
    )
    enc2d_offset = torch.cat([torch.tensor([0], device=point.feat.device), torch.cumsum(enc2d_count, dim=0)])
    enc2d_mask = torch.cat(
        [
            torch.arange(0, c, device=enc2d_count.device) + data_dict_global_offset[i * model.num_global_view]
            for i, c in enumerate(enc2d_count)
        ],
        dim=0,
    )
    batch_points_3d = offset2batch(enc2d_offset[1:])
    correspondence = to_feature["correspondence"][enc2d_mask]
    corr_mask = torch.any(correspondence != torch.tensor([-1, -1], device=correspondence.device), dim=2)
    valid_index = torch.where(corr_mask)
    if valid_index[0].numel() == 0:
        return None

    offset_img_num = torch.cat([torch.tensor([0], device=point.feat.device), torch.cumsum(batch["img_num"], dim=0)])
    batch_index = batch_points_3d[valid_index[0]]
    batch_img_num = offset_img_num[:-1][batch_index]
    feature_index = torch.cat(
        [
            batch_img_num.unsqueeze(-1),
            valid_index[1].unsqueeze(-1),
            correspondence[valid_index],
        ],
        dim=-1,
    ).long()
    feature_index = (
        feature_index[:, 0] * model.patch_h * model.patch_w
        + feature_index[:, 1] * model.patch_h * model.patch_w
        + feature_index[:, 2] * model.patch_w
        + feature_index[:, 3]
    )
    feature_index, inverse_index = torch.unique(feature_index, sorted=True, return_inverse=True)
    encoder_feat = to_feature["feat"][enc2d_mask][valid_index[0]]
    encoder_feat = torch_scatter.scatter_mean(
        encoder_feat,
        inverse_index,
        dim=0,
        dim_size=feature_index.shape[0],
    )
    projected_feat = model.patch_proj(encoder_feat)

    labels = batch["global_segment"][enc2d_mask][valid_index[0]].long()
    valid_label = (labels >= 0) & (labels < 20)
    if not valid_label.any():
        return None
    counts = torch.zeros((feature_index.shape[0], 20), device=labels.device, dtype=torch.float32)
    counts.index_add_(0, inverse_index[valid_label], F.one_hot(labels[valid_label], 20).float())
    totals = counts.sum(dim=1)
    top_count, top_label = counts.max(dim=1)
    confidence = top_count / totals.clamp_min(1.0)
    keep = (totals >= min_points) & (confidence >= majority_threshold)
    if not keep.any():
        return None
    patch_row = (feature_index % (model.patch_h * model.patch_w)) // model.patch_w
    patch_col = feature_index % model.patch_w
    pos = torch.stack(
        [
            patch_row.float() / max(model.patch_h - 1, 1),
            patch_col.float() / max(model.patch_w - 1, 1),
        ],
        dim=1,
    )
    return {
        "encoder": encoder_feat[keep].float().cpu(),
        "projected": projected_feat[keep].float().cpu(),
        "label": top_label[keep].cpu().long(),
        "points": totals[keep].cpu().long(),
        "confidence": confidence[keep].cpu().float(),
        "pos": pos[keep].cpu().float(),
    }


def collect_split(args: argparse.Namespace, model, loader, split: str, max_batches: int):
    features = {"encoder_pooled": [], "patch_proj": []}
    labels = []
    positions = []
    class_counts = {args.wall_class: 0, args.picture_class: 0}
    seen_batches = 0
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches >= 0 and batch_idx >= max_batches:
                break
            batch = move_batch_to_cuda(batch)
            out = extract_batch_patch_features(
                model,
                batch,
                min_points=args.min_points_per_patch,
                majority_threshold=args.majority_threshold,
            )
            seen_batches += 1
            if out is None:
                continue
            for cls in (args.wall_class, args.picture_class):
                class_mask = out["label"] == cls
                if not class_mask.any():
                    continue
                room = args.max_sem_per_class - class_counts[cls]
                if room <= 0:
                    continue
                idx = class_mask.nonzero(as_tuple=False).flatten()
                if idx.numel() > room:
                    idx = idx[:room]
                features["encoder_pooled"].append(out["encoder"][idx])
                features["patch_proj"].append(out["projected"][idx])
                labels.append(torch.full((idx.numel(),), 1 if cls == args.picture_class else 0, dtype=torch.float32))
                positions.append(out["pos"][idx])
                class_counts[cls] += int(idx.numel())
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[collect] split={split} batch={batch_idx + 1} "
                    f"wall={class_counts[args.wall_class]} picture={class_counts[args.picture_class]}",
                    flush=True,
                )
            if all(v >= args.max_sem_per_class for v in class_counts.values()):
                break
    if not labels:
        raise RuntimeError(f"no picture/wall patch samples collected for split={split}")
    payload = {
        "encoder_pooled": torch.cat(features["encoder_pooled"], dim=0),
        "patch_proj": torch.cat(features["patch_proj"], dim=0),
        "label": torch.cat(labels, dim=0),
        "pos": torch.cat(positions, dim=0),
        "seen_batches": seen_batches,
        "class_counts": {
            SCANNET20_CLASS_NAMES[k]: int(v) for k, v in class_counts.items()
        },
    }
    print(
        f"[collect] split={split} done samples={payload['label'].numel()} "
        f"counts={payload['class_counts']}",
        flush=True,
    )
    return payload


def standardize(train: torch.Tensor, val: torch.Tensor):
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (val - mean) / std


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.detach().cpu().long()
    scores = scores.detach().cpu()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())
    return float(auc.item())


def fit_binary_probe(train_x, train_y, val_x, val_y, args):
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    val_x = val_x.cuda()
    val_y = val_y.cuda()
    weight = torch.zeros(train_x.shape[1], device="cuda", requires_grad=True)
    bias = torch.zeros((), device="cuda", requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=args.logreg_lr, weight_decay=args.weight_decay)
    with torch.enable_grad():
        for _ in range(args.logreg_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = train_x @ weight + bias
            loss = F.binary_cross_entropy_with_logits(logits, train_y)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = val_x @ weight + bias
        pred = (logits >= 0).float()
        acc = (pred == val_y).float().mean().item()
        pos_mask = val_y == 1
        neg_mask = val_y == 0
        picture_acc = (pred[pos_mask] == val_y[pos_mask]).float().mean().item() if pos_mask.any() else float("nan")
        wall_acc = (pred[neg_mask] == val_y[neg_mask]).float().mean().item() if neg_mask.any() else float("nan")
        bal_acc = float(np.nanmean([picture_acc, wall_acc]))
        auc = binary_auc(logits, val_y)
        train_loss = F.binary_cross_entropy_with_logits(train_x @ weight + bias, train_y).item()
    return {
        "picture_wall_acc": acc,
        "picture_wall_bal_acc": bal_acc,
        "picture_acc": picture_acc,
        "wall_acc": wall_acc,
        "picture_wall_auc": auc,
        "picture_wall_train_loss": train_loss,
    }


def fit_position_probe(train_x, train_y, val_x, val_y, ridge: float):
    ones_train = torch.ones((train_x.shape[0], 1), dtype=train_x.dtype)
    ones_val = torch.ones((val_x.shape[0], 1), dtype=val_x.dtype)
    train_aug = torch.cat([train_x.cpu(), ones_train], dim=1)
    val_aug = torch.cat([val_x.cpu(), ones_val], dim=1)
    eye = torch.eye(train_aug.shape[1], dtype=train_aug.dtype)
    eye[-1, -1] = 0.0
    weight = torch.linalg.solve(train_aug.T @ train_aug + ridge * eye, train_aug.T @ train_y.cpu())
    pred = val_aug @ weight
    ss_res = ((val_y.cpu() - pred) ** 2).sum(dim=0)
    ss_tot = ((val_y.cpu() - val_y.cpu().mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return {
        "position_r2_row": float(r2[0].item()),
        "position_r2_col": float(r2[1].item()),
        "position_r2_mean": float(r2.mean().item()),
    }


def evaluate_variant(name: str, train_payload: dict, val_payload: dict, args: argparse.Namespace):
    train_x, val_x = standardize(train_payload[name], val_payload[name])
    metrics = {"variant": name}
    metrics.update(fit_binary_probe(train_x, train_payload["label"], val_x, val_payload["label"], args))
    metrics.update(fit_position_probe(train_x, train_payload["pos"], val_x, val_payload["pos"], args.ridge))
    metrics["train_samples"] = int(train_payload["label"].numel())
    metrics["val_samples"] = int(val_payload["label"].numel())
    metrics["train_picture"] = int((train_payload["label"] == 1).sum().item())
    metrics["val_picture"] = int((val_payload["label"] == 1).sum().item())
    return metrics


def write_outputs(args: argparse.Namespace, rows: list[dict], metadata: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "concerto3d_patch_separation_stepA.csv"
    fields = [
        "variant",
        "picture_wall_acc",
        "picture_wall_bal_acc",
        "picture_acc",
        "wall_acc",
        "picture_wall_auc",
        "picture_wall_train_loss",
        "position_r2_mean",
        "position_r2_row",
        "position_r2_col",
        "train_samples",
        "val_samples",
        "train_picture",
        "val_picture",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    md_path = args.output_dir / "concerto3d_patch_separation_stepA.md"
    lines = [
        "# Concerto 3D Patch Separation Step A",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- train batches seen: {metadata['train']['seen_batches']}",
        f"- val batches seen: {metadata['val']['seen_batches']}",
        f"- train class counts: {metadata['train']['class_counts']}",
        f"- val class counts: {metadata['val']['class_counts']}",
        "",
        "## Results",
        "",
        "| variant | pic/wall bal acc | pic acc | wall acc | AUC | pos R2 mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {picture_wall_bal_acc:.4f} | {picture_acc:.4f} | "
            "{wall_acc:.4f} | {picture_wall_auc:.4f} | {position_r2_mean:.4f} |".format(**row)
        )
    lines += [
        "",
        "## Interpretation Guide",
        "- `encoder_pooled` is the Concerto 3D encoder feature pooled to image patches through point-pixel correspondences.",
        "- `patch_proj` is the feature after Concerto's enc2d patch projection head.",
        "- Compare picture/wall balanced accuracy to DINO Step A' to see whether 3D alignment preserves or loses this 2D semantic separation.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {csv_path}", flush=True)
    print(f"[write] {md_path}", flush=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ensure_repo_on_path(repo_root)
    seed_everything(args.seed)
    cfg = load_cfg(repo_root, args.config)
    if not args.no_segment_alias:
        alias_summary = ensure_segment_aliases(args.data_root, [args.train_split, args.val_split])
        print(f"[segment_alias] {alias_summary}", flush=True)

    if args.dry_run:
        train_loader = build_loader(cfg, args.data_root, args.train_split, 1, 0)
        batch = next(iter(train_loader))
        print(f"[dry] keys={sorted(batch.keys())}")
        for key in ["global_feat", "global_segment", "global_correspondence", "images", "img_num"]:
            value = batch.get(key)
            print(f"[dry] {key}: {tuple(value.shape) if torch.is_tensor(value) else value}")
        return 0

    from pointcept.models.builder import build_model

    print(f"[info] building model config={args.config}", flush=True)
    model = build_model(cfg.model).cuda().eval()
    load_weight(model, (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight)
    if hasattr(model, "enc2d_model"):
        model.enc2d_model.cpu()
    torch.cuda.empty_cache()

    train_loader = build_loader(cfg, args.data_root, args.train_split, args.batch_size, args.num_worker)
    val_loader = build_loader(cfg, args.data_root, args.val_split, args.batch_size, args.num_worker)
    train = collect_split(args, model, train_loader, args.train_split, args.max_train_batches)
    val = collect_split(args, model, val_loader, args.val_split, args.max_val_batches)
    rows = [evaluate_variant("encoder_pooled", train, val, args), evaluate_variant("patch_proj", train, val, args)]
    metadata = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train": {
            "seen_batches": train["seen_batches"],
            "class_counts": train["class_counts"],
        },
        "val": {
            "seen_batches": val["seen_batches"],
            "class_counts": val["class_counts"],
        },
    }
    write_outputs(args, rows, metadata)
    for row in rows:
        print(
            "[result] {variant} picwall_bal={picture_wall_bal_acc:.4f} "
            "auc={picture_wall_auc:.4f} pos_r2={position_r2_mean:.4f}".format(**row),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
