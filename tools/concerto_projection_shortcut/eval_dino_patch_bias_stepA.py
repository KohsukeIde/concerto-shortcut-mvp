#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step A': probe DINOv2 ScanNet patch features for picture/wall "
            "separability and patch-position predictability."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/concerto_scannet_imagepoint_absmeta"),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/dinov2-with-registers-giant")
    parser.add_argument("--crop-h", type=int, default=518)
    parser.add_argument("--crop-w", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-images", type=int, default=256)
    parser.add_argument("--max-val-images", type=int, default=128)
    parser.add_argument("--max-sem-per-class", type=int, default=6000)
    parser.add_argument("--max-pos-patches", type=int, default=20000)
    parser.add_argument("--pos-patches-per-image", type=int, default=64)
    parser.add_argument("--sem-patches-per-image-per-class", type=int, default=8)
    parser.add_argument("--min-points-per-patch", type=int, default=4)
    parser.add_argument("--majority-threshold", type=float, default=0.6)
    parser.add_argument("--picture-class", type=int, default=10)
    parser.add_argument("--wall-class", type=int, default=0)
    parser.add_argument("--logreg-steps", type=int, default=600)
    parser.add_argument("--logreg-lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_split(data_root: Path, split: str) -> list[tuple[str, dict]]:
    path = data_root / "splits" / f"{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.items())


def center_crop_box(width: int, height: int, patch_h: int, patch_w: int) -> tuple[int, int, int, int]:
    div_w = width // patch_w
    div_h = height // patch_h
    div_min = max(min(div_w, div_h), 1)
    crop_img_width = div_min * patch_w
    crop_img_height = div_min * patch_h
    left = int((width - crop_img_width) / 2)
    top = int((height - crop_img_height) / 2)
    right = int((width + crop_img_width) / 2)
    bottom = int((height + crop_img_height) / 2)
    return left, top, right, bottom


def resize_correspondence_info(
    correspondence: np.ndarray,
    size: tuple[int, int],
    size0: tuple[int, int],
    crop_size: tuple[int, int, int, int],
    alignment: int,
) -> np.ndarray:
    h, w = size
    img_h0, img_w0 = size0
    del img_h0, img_w0
    left, top, right, bottom = crop_size
    crop_h = bottom - top
    crop_w = right - left
    mask_crop = (
        (correspondence[:, 1] >= top)
        & (correspondence[:, 1] < bottom)
        & (correspondence[:, 0] >= left)
        & (correspondence[:, 0] < right)
    )
    correspondence = correspondence[mask_crop].copy()
    if correspondence.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    correspondence[:, 1] -= top
    correspondence[:, 0] -= left
    correspondence[:, 1] = (correspondence[:, 1] * h / crop_h // alignment).astype(np.int32)
    correspondence[:, 0] = (correspondence[:, 0] * w / crop_w // alignment).astype(np.int32)
    correspondence = correspondence[:, [1, 0, 2]]
    return np.unique(correspondence, axis=0).astype(np.int32)


def load_image_tensor(
    image_path: str,
    crop_h: int,
    crop_w: int,
    patch_h: int,
    patch_w: int,
    patch_size: int,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int, int, int]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    crop = center_crop_box(width, height, patch_h, patch_w)
    image = image.crop(crop)
    transform = T.Compose(
        [
            T.Resize((patch_h * patch_size, patch_w * patch_size)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transform(image), (height, width), crop


def iter_record_images(records: list[tuple[str, dict]], max_images: int, rng: random.Random) -> Iterable[tuple[str, dict, int]]:
    order = list(range(len(records)))
    rng.shuffle(order)
    yielded = 0
    for rec_idx in order:
        name, record = records[rec_idx]
        image_ids = list(range(len(record["images"])))
        rng.shuffle(image_ids)
        for image_id in image_ids:
            yield name, record, image_id
            yielded += 1
            if yielded >= max_images:
                return


def patch_labels_from_correspondence(
    record: dict,
    image_id: int,
    size0: tuple[int, int],
    crop: tuple[int, int, int, int],
    patch_h: int,
    patch_w: int,
    patch_size: int,
    min_points: int,
    majority_threshold: float,
) -> dict[int, tuple[int, int, float]]:
    point_root = Path(record["pointclouds"])
    seg_path = point_root / "segment20.npy"
    if not seg_path.is_file():
        seg_path = point_root / "segment.npy"
    if not seg_path.is_file():
        return {}
    segment = np.load(seg_path).reshape(-1).astype(np.int64)
    corr = np.load(record["correspondences"][image_id]).astype(np.int32)
    if np.array_equal(corr, -np.ones((1, 3), dtype=np.int32)):
        return {}
    corr = resize_correspondence_info(
        corr,
        (patch_h * patch_size, patch_w * patch_size),
        size0,
        crop,
        patch_size,
    )
    if corr.size == 0:
        return {}

    patch_counts: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(20, dtype=np.int64))
    for row, col, point_idx in corr:
        if row < 0 or row >= patch_h or col < 0 or col >= patch_w:
            continue
        if point_idx < 0 or point_idx >= segment.shape[0]:
            continue
        label = int(segment[point_idx])
        if label < 0 or label >= 20:
            continue
        patch_counts[int(row) * patch_w + int(col)][label] += 1

    labels = {}
    for patch_index, counts in patch_counts.items():
        total = int(counts.sum())
        if total < min_points:
            continue
        top_label = int(counts.argmax())
        top_count = int(counts[top_label])
        confidence = float(top_count / max(total, 1))
        if confidence < majority_threshold:
            continue
        labels[patch_index] = (top_label, total, confidence)
    return labels


@torch.no_grad()
def dino_forward(model: torch.nn.Module, images: torch.Tensor, patch_count: int) -> torch.Tensor:
    outputs = model(images)
    features = outputs.last_hidden_state[:, -patch_count:, :]
    return features.float().cpu()


def sample_split_features(
    args: argparse.Namespace,
    split: str,
    max_images: int,
    rng: random.Random,
    model: torch.nn.Module,
):
    records = load_split(args.data_root, split)
    patch_h = args.crop_h // args.patch_size
    patch_w = args.crop_w // args.patch_size
    patch_count = patch_h * patch_w

    sem_feats: list[torch.Tensor] = []
    sem_labels: list[int] = []
    sem_meta: list[dict] = []
    pos_feats: list[torch.Tensor] = []
    pos_targets: list[tuple[float, float]] = []
    pos_meta: list[dict] = []
    class_counts = {args.wall_class: 0, args.picture_class: 0}
    pos_count = 0
    pending_images: list[torch.Tensor] = []
    pending_infos: list[tuple[str, dict, int, tuple[int, int], tuple[int, int, int, int]]] = []

    def flush() -> None:
        nonlocal pending_images, pending_infos, pos_count
        if not pending_images:
            return
        print(
            f"[extract] split={split} images={seen_images} "
            f"sem_wall={class_counts[args.wall_class]} "
            f"sem_picture={class_counts[args.picture_class]} pos={pos_count}",
            flush=True,
        )
        batch = torch.stack(pending_images).to(args.device, non_blocking=True)
        features = dino_forward(model, batch, patch_count)
        for local_idx, (name, record, image_id, size0, crop) in enumerate(pending_infos):
            patch_labels = patch_labels_from_correspondence(
                record=record,
                image_id=image_id,
                size0=size0,
                crop=crop,
                patch_h=patch_h,
                patch_w=patch_w,
                patch_size=args.patch_size,
                min_points=args.min_points_per_patch,
                majority_threshold=args.majority_threshold,
            )
            per_class_taken = {args.wall_class: 0, args.picture_class: 0}
            label_items = list(patch_labels.items())
            rng.shuffle(label_items)
            for patch_index, (label, total, confidence) in label_items:
                if label not in (args.wall_class, args.picture_class):
                    continue
                if class_counts[label] >= args.max_sem_per_class:
                    continue
                if per_class_taken[label] >= args.sem_patches_per_image_per_class:
                    continue
                sem_feats.append(features[local_idx, patch_index].clone())
                sem_labels.append(1 if label == args.picture_class else 0)
                sem_meta.append(
                    {
                        "scene": name,
                        "image_id": image_id,
                        "patch_index": patch_index,
                        "label": label,
                        "points": total,
                        "confidence": confidence,
                    }
                )
                class_counts[label] += 1
                per_class_taken[label] += 1

            if pos_count < args.max_pos_patches:
                available = list(range(patch_count))
                rng.shuffle(available)
                take = min(args.pos_patches_per_image, args.max_pos_patches - pos_count)
                for patch_index in available[:take]:
                    row = patch_index // patch_w
                    col = patch_index % patch_w
                    pos_feats.append(features[local_idx, patch_index].clone())
                    pos_targets.append(
                        (
                            float(row / max(patch_h - 1, 1)),
                            float(col / max(patch_w - 1, 1)),
                        )
                    )
                    pos_meta.append({"scene": name, "image_id": image_id, "patch_index": patch_index})
                pos_count += take
        pending_images = []
        pending_infos = []

    seen_images = 0
    for name, record, image_id in iter_record_images(records, max_images=max_images, rng=rng):
        image_path = record["images"][image_id]
        corr_path = record["correspondences"][image_id]
        if not Path(image_path).is_file() or not Path(corr_path).is_file():
            continue
        image_tensor, size0, crop = load_image_tensor(
            image_path=image_path,
            crop_h=args.crop_h,
            crop_w=args.crop_w,
            patch_h=patch_h,
            patch_w=patch_w,
            patch_size=args.patch_size,
        )
        pending_images.append(image_tensor)
        pending_infos.append((name, record, image_id, size0, crop))
        seen_images += 1
        if len(pending_images) >= args.batch_size:
            flush()
        if (
            class_counts[args.wall_class] >= args.max_sem_per_class
            and class_counts[args.picture_class] >= args.max_sem_per_class
            and pos_count >= args.max_pos_patches
        ):
            break
    flush()

    if len(sem_feats) == 0 or len(pos_feats) == 0:
        raise RuntimeError(f"no usable samples collected for split={split}")
    return {
        "sem_features": torch.stack(sem_feats),
        "sem_labels": torch.tensor(sem_labels, dtype=torch.float32),
        "pos_features": torch.stack(pos_feats),
        "pos_targets": torch.tensor(pos_targets, dtype=torch.float32),
        "sem_meta": sem_meta,
        "pos_meta": pos_meta,
        "seen_images": seen_images,
        "class_counts": {
            SCANNET20_CLASS_NAMES[k]: int(v) for k, v in class_counts.items()
        },
    }


def standardize(train: torch.Tensor, val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (val - mean) / std, mean.squeeze(0), std.squeeze(0)


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


def fit_binary_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> dict:
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    weight = torch.zeros(train_x.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=lr, weight_decay=weight_decay)
    with torch.enable_grad():
        for _ in range(steps):
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


def fit_position_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    ridge: float,
) -> tuple[dict, torch.Tensor]:
    ones_train = torch.ones((train_x.shape[0], 1), dtype=train_x.dtype)
    ones_val = torch.ones((val_x.shape[0], 1), dtype=val_x.dtype)
    train_aug = torch.cat([train_x.cpu(), ones_train], dim=1)
    val_aug = torch.cat([val_x.cpu(), ones_val], dim=1)
    eye = torch.eye(train_aug.shape[1], dtype=train_aug.dtype)
    eye[-1, -1] = 0.0
    xtx = train_aug.T @ train_aug + ridge * eye
    xty = train_aug.T @ train_y.cpu()
    weight = torch.linalg.solve(xtx, xty)
    pred = val_aug @ weight
    ss_res = ((val_y.cpu() - pred) ** 2).sum(dim=0)
    ss_tot = ((val_y.cpu() - val_y.cpu().mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot
    mse = ((val_y.cpu() - pred) ** 2).mean(dim=0)
    return (
        {
            "position_r2_row": float(r2[0].item()),
            "position_r2_col": float(r2[1].item()),
            "position_r2_mean": float(r2.mean().item()),
            "position_mse_row": float(mse[0].item()),
            "position_mse_col": float(mse[1].item()),
        },
        weight[:-1].clone(),
    )


def remove_position_subspace(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    q, r = torch.linalg.qr(pos_weight.cpu(), mode="reduced")
    diag = r.diagonal().abs()
    rank = int((diag > 1e-8).sum().item())
    q = q[:, :rank]
    if rank == 0:
        return train_x, val_x, rank
    return train_x.cpu() - (train_x.cpu() @ q) @ q.T, val_x.cpu() - (val_x.cpu() @ q) @ q.T, rank


def evaluate_variant(
    name: str,
    sem_train_x: torch.Tensor,
    sem_val_x: torch.Tensor,
    sem_train_y: torch.Tensor,
    sem_val_y: torch.Tensor,
    pos_train_x: torch.Tensor,
    pos_val_x: torch.Tensor,
    pos_train_y: torch.Tensor,
    pos_val_y: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict, torch.Tensor]:
    sem_train_z, sem_val_z, _, _ = standardize(sem_train_x, sem_val_x)
    pos_train_z, pos_val_z, _, _ = standardize(pos_train_x, pos_val_x)
    picture_wall = fit_binary_probe(
        sem_train_z,
        sem_train_y,
        sem_val_z,
        sem_val_y,
        steps=args.logreg_steps,
        lr=args.logreg_lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    position, pos_weight = fit_position_probe(
        pos_train_z,
        pos_train_y,
        pos_val_z,
        pos_val_y,
        ridge=args.ridge,
    )
    result = {"variant": name}
    result.update(picture_wall)
    result.update(position)
    return result, pos_weight


def write_outputs(args: argparse.Namespace, rows: list[dict], metadata: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dino_patch_bias_stepA.csv"
    fieldnames = [
        "variant",
        "picture_wall_acc",
        "picture_wall_bal_acc",
        "picture_acc",
        "wall_acc",
        "picture_wall_auc",
        "picture_wall_train_loss",
        "position_r2_row",
        "position_r2_col",
        "position_r2_mean",
        "position_mse_row",
        "position_mse_col",
        "position_basis_dim",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path = args.output_dir / "dino_patch_bias_stepA.md"
    lines = [
        "# DINO Patch Bias Step A'",
        "",
        "## Setup",
        f"- model: `{args.model_name}`",
        f"- data root: `{args.data_root}`",
        f"- train images seen: {metadata['train']['seen_images']}",
        f"- val images seen: {metadata['val']['seen_images']}",
        f"- train semantic samples: {metadata['train']['sem_samples']}",
        f"- val semantic samples: {metadata['val']['sem_samples']}",
        f"- train position samples: {metadata['train']['pos_samples']}",
        f"- val position samples: {metadata['val']['pos_samples']}",
        "",
        "## Results",
        "",
        "| variant | pic/wall bal acc | pic acc | wall acc | AUC | pos R2 mean | pos R2 row | pos R2 col |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {picture_wall_bal_acc:.4f} | {picture_acc:.4f} | "
            "{wall_acc:.4f} | {picture_wall_auc:.4f} | {position_r2_mean:.4f} | "
            "{position_r2_row:.4f} | {position_r2_col:.4f} |".format(**row)
        )
    lines += [
        "",
        "## Interpretation Guide",
        "- High picture/wall accuracy means DINO patch features carry separable 2D semantics for this failure pair.",
        "- High position R2 means DINO patch features linearly expose patch location.",
        "- If RASA-lite lowers position R2 while retaining picture/wall accuracy, teacher-side target debiasing is plausible.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {csv_path}")
    print(f"[write] {md_path}")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    rng = random.Random(args.seed)

    if args.dry_run:
        for split in [args.train_split, args.val_split]:
            records = load_split(args.data_root, split)
            print(f"[dry] split={split} records={len(records)} first={records[0][0]}")
        return 0

    print(f"[info] device={args.device}", flush=True)
    print(f"[info] model={args.model_name}", flush=True)
    print(f"[info] data_root={args.data_root}", flush=True)
    print("[info] loading DINO model", flush=True)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    model.eval()
    print("[info] DINO model loaded", flush=True)
    train = sample_split_features(args, args.train_split, args.max_train_images, rng, model)
    val = sample_split_features(args, args.val_split, args.max_val_images, rng, model)

    sem_train_x = train["sem_features"]
    sem_val_x = val["sem_features"]
    sem_train_y = train["sem_labels"]
    sem_val_y = val["sem_labels"]
    pos_train_x = train["pos_features"]
    pos_val_x = val["pos_features"]
    pos_train_y = train["pos_targets"]
    pos_val_y = val["pos_targets"]

    print(
        "[samples] sem_train={} sem_val={} pos_train={} pos_val={}".format(
            sem_train_x.shape[0],
            sem_val_x.shape[0],
            pos_train_x.shape[0],
            pos_val_x.shape[0],
        ),
        flush=True,
    )
    raw_row, pos_weight = evaluate_variant(
        "raw_dino",
        sem_train_x,
        sem_val_x,
        sem_train_y,
        sem_val_y,
        pos_train_x,
        pos_val_x,
        pos_train_y,
        pos_val_y,
        args,
    )
    sem_train_z, sem_val_z, _, _ = standardize(sem_train_x, sem_val_x)
    pos_train_z, pos_val_z, _, _ = standardize(pos_train_x, pos_val_x)
    sem_train_rasa, sem_val_rasa, rank = remove_position_subspace(
        sem_train_z,
        sem_val_z,
        pos_weight,
    )
    pos_train_rasa, pos_val_rasa, _ = remove_position_subspace(
        pos_train_z,
        pos_val_z,
        pos_weight,
    )
    rasa_row, _ = evaluate_variant(
        "rasa_lite_position_removed",
        sem_train_rasa,
        sem_val_rasa,
        sem_train_y,
        sem_val_y,
        pos_train_rasa,
        pos_val_rasa,
        pos_train_y,
        pos_val_y,
        args,
    )
    raw_row["position_basis_dim"] = rank
    rasa_row["position_basis_dim"] = rank
    rows = [raw_row, rasa_row]
    metadata = {
        "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
        "train": {
            "seen_images": train["seen_images"],
            "class_counts": train["class_counts"],
            "sem_samples": int(sem_train_x.shape[0]),
            "pos_samples": int(pos_train_x.shape[0]),
        },
        "val": {
            "seen_images": val["seen_images"],
            "class_counts": val["class_counts"],
            "sem_samples": int(sem_val_x.shape[0]),
            "pos_samples": int(pos_val_x.shape[0]),
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
