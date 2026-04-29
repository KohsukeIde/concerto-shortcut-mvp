#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_dino_patch_bias_stepA import (  # noqa: E402
    SCANNET20_CLASS_NAMES,
    dino_forward,
    load_image_tensor,
    load_split,
    resize_correspondence_info,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scene-level DINO recognizability calibration under 3D support stress. "
            "DINO sees the actual ScanNet RGB frame; patch labels are derived only "
            "from points retained by each 3D support condition."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/concerto_scannet_imagepoint_absmeta"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/dinov2-with-registers-giant")
    parser.add_argument("--conditions", default="clean,random_keep80,random_keep50,random_keep20,random_keep10,structured_keep80,structured_keep50,structured_keep20,structured_keep10,instance_keep20")
    parser.add_argument("--crop-h", type=int, default=518)
    parser.add_argument("--crop-w", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-images", type=int, default=256)
    parser.add_argument("--max-val-images", type=int, default=128)
    parser.add_argument("--max-per-class", type=int, default=1200)
    parser.add_argument("--patches-per-image-per-class", type=int, default=6)
    parser.add_argument("--min-points-per-patch", type=int, default=4)
    parser.add_argument("--majority-threshold", type=float, default=0.6)
    parser.add_argument("--structured-cell-size", type=float, default=1.28)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_conditions(spec: str) -> list[str]:
    conditions = [item.strip() for item in spec.split(",") if item.strip()]
    if not conditions:
        raise ValueError("empty --conditions")
    return conditions


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def condition_keep_mask(point_root: Path, condition: str, seed: int, cell_size: float) -> np.ndarray:
    segment = np.load(point_root / "segment20.npy").reshape(-1).astype(np.int64)
    n = int(segment.shape[0])
    if condition == "clean":
        return np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    if condition.startswith("random_keep"):
        ratio = int(condition[len("random_keep") :]) / 100.0
        keep = rng.random(n) < ratio
        if not keep.any():
            keep[rng.integers(n)] = True
        return keep
    if condition.startswith("structured_keep"):
        ratio = int(condition[len("structured_keep") :]) / 100.0
        coord = np.load(point_root / "coord.npy").astype(np.float32)
        keys = np.floor(coord / float(cell_size)).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        n_region = int(inv.max()) + 1
        region_keep = rng.random(n_region) < ratio
        if not region_keep.any():
            region_keep[rng.integers(n_region)] = True
        return region_keep[inv]
    if condition.startswith("instance_keep"):
        ratio = int(condition[len("instance_keep") :]) / 100.0
        instance_path = point_root / "instance.npy"
        if not instance_path.is_file():
            return condition_keep_mask(point_root, "random_keep" + str(int(ratio * 100)), seed, cell_size)
        instance = np.load(instance_path).reshape(-1).astype(np.int64)
        keep = np.zeros(n, dtype=bool)
        stuff = instance < 0
        if np.any(stuff):
            keep[stuff] = rng.random(int(stuff.sum())) < ratio
        inst_ids = np.unique(instance[instance >= 0])
        if inst_ids.size:
            inst_keep = rng.random(inst_ids.shape[0]) < ratio
            if not inst_keep.any():
                inst_keep[rng.integers(inst_ids.shape[0])] = True
            kept = set(inst_ids[inst_keep].tolist())
            for inst_id in kept:
                keep[instance == inst_id] = True
        if not keep.any():
            keep[rng.integers(n)] = True
        return keep
    raise ValueError(f"unknown condition: {condition}")


def patch_labels_for_condition(
    record_name: str,
    record: dict,
    image_id: int,
    condition: str,
    size0: tuple[int, int],
    crop: tuple[int, int, int, int],
    patch_h: int,
    patch_w: int,
    patch_size: int,
    min_points: int,
    majority_threshold: float,
    cell_size: float,
) -> dict[int, tuple[int, int, float]]:
    point_root = Path(record["pointclouds"])
    segment_path = point_root / "segment20.npy"
    if not segment_path.is_file():
        segment_path = point_root / "segment.npy"
    if not segment_path.is_file():
        return {}
    segment = np.load(segment_path).reshape(-1).astype(np.int64)
    keep_mask = condition_keep_mask(
        point_root,
        condition,
        seed=stable_seed(record_name, image_id, condition),
        cell_size=cell_size,
    )
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
    counts_by_patch: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(20, dtype=np.int64))
    for row, col, point_idx in corr:
        if row < 0 or row >= patch_h or col < 0 or col >= patch_w:
            continue
        if point_idx < 0 or point_idx >= segment.shape[0]:
            continue
        if not keep_mask[int(point_idx)]:
            continue
        label = int(segment[int(point_idx)])
        if label < 0 or label >= 20:
            continue
        counts_by_patch[int(row) * patch_w + int(col)][label] += 1
    labels: dict[int, tuple[int, int, float]] = {}
    for patch_index, counts in counts_by_patch.items():
        total = int(counts.sum())
        if total < min_points:
            continue
        label = int(counts.argmax())
        conf = float(counts[label] / max(total, 1))
        if conf < majority_threshold:
            continue
        labels[patch_index] = (label, total, conf)
    return labels


def iter_record_images(records: list[tuple[str, dict]], max_images: int, rng: random.Random):
    order = list(range(len(records)))
    rng.shuffle(order)
    yielded = 0
    for idx in order:
        name, record = records[idx]
        image_ids = list(range(len(record["images"])))
        rng.shuffle(image_ids)
        for image_id in image_ids:
            yield name, record, image_id
            yielded += 1
            if yielded >= max_images:
                return


@torch.no_grad()
def collect_condition_features(
    args: argparse.Namespace,
    split: str,
    condition: str,
    max_images: int,
    model: torch.nn.Module,
) -> dict:
    records = load_split(args.data_root, split)
    # Keep the frame set/order fixed across support conditions. Otherwise clean-vs-stress
    # differences can be polluted by different sampled ScanNet RGB frames.
    image_rng = random.Random(stable_seed(args.seed, "image-order", split))
    sample_rng = random.Random(stable_seed(args.seed, "patch-sample", split, condition))
    patch_h = args.crop_h // args.patch_size
    patch_w = args.crop_w // args.patch_size
    patch_count = patch_h * patch_w
    feats: list[torch.Tensor] = []
    labels: list[int] = []
    meta: list[dict] = []
    class_counts = np.zeros(20, dtype=np.int64)
    pending_images: list[torch.Tensor] = []
    pending_infos: list[tuple[str, dict, int, tuple[int, int], tuple[int, int, int, int]]] = []
    seen_images = 0

    def flush() -> None:
        nonlocal pending_images, pending_infos
        if not pending_images:
            return
        print(
            f"[extract] split={split} cond={condition} images={seen_images} samples={len(labels)}",
            flush=True,
        )
        batch = torch.stack(pending_images).to(args.device, non_blocking=True)
        dino = dino_forward(model, batch, patch_count)
        for local_idx, (name, record, image_id, size0, crop) in enumerate(pending_infos):
            patch_labels = patch_labels_for_condition(
                record_name=name,
                record=record,
                image_id=image_id,
                condition=condition,
                size0=size0,
                crop=crop,
                patch_h=patch_h,
                patch_w=patch_w,
                patch_size=args.patch_size,
                min_points=args.min_points_per_patch,
                majority_threshold=args.majority_threshold,
                cell_size=args.structured_cell_size,
            )
            per_img = np.zeros(20, dtype=np.int64)
            items = list(patch_labels.items())
            sample_rng.shuffle(items)
            for patch_index, (label, total, conf) in items:
                if class_counts[label] >= args.max_per_class:
                    continue
                if per_img[label] >= args.patches_per_image_per_class:
                    continue
                feats.append(dino[local_idx, patch_index].clone())
                labels.append(label)
                meta.append(
                    {
                        "scene": name,
                        "image_id": image_id,
                        "patch_index": int(patch_index),
                        "label": int(label),
                        "points": int(total),
                        "confidence": float(conf),
                    }
                )
                class_counts[label] += 1
                per_img[label] += 1
        pending_images = []
        pending_infos = []

    for name, record, image_id in iter_record_images(records, max_images, image_rng):
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
        if np.all(class_counts >= args.max_per_class):
            break
    flush()
    if not feats:
        raise RuntimeError(f"no samples collected split={split} condition={condition}")
    return {
        "features": torch.stack(feats),
        "labels": torch.tensor(labels, dtype=torch.long),
        "meta": meta,
        "seen_images": seen_images,
        "class_counts": {SCANNET20_CLASS_NAMES[i]: int(class_counts[i]) for i in range(20)},
    }


def fit_multiclass_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    train_x, val_x, _, _ = standardize(train_x, val_x)
    train_x = train_x.to(args.device)
    train_y = train_y.to(args.device)
    val_x = val_x.to(args.device)
    val_y = val_y.to(args.device)
    weight = torch.zeros((train_x.shape[1], 20), device=args.device, requires_grad=True)
    bias = torch.zeros((20,), device=args.device, requires_grad=True)
    opt = torch.optim.AdamW([weight, bias], lr=args.lr, weight_decay=args.weight_decay)
    present = torch.bincount(train_y, minlength=20).float().to(args.device)
    cls_weight = (present.sum() / present.clamp_min(1.0)).clamp(max=20.0)
    cls_weight[present == 0] = 0.0
    with torch.enable_grad():
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            logits = train_x @ weight + bias
            loss = F.cross_entropy(logits, train_y, weight=cls_weight)
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = val_x @ weight + bias
        pred = logits.argmax(1)
        acc = float((pred == val_y).float().mean().item())
        per_class = {}
        vals = []
        for cls, name in enumerate(SCANNET20_CLASS_NAMES):
            mask = val_y == cls
            if bool(mask.any()):
                v = float((pred[mask] == val_y[mask]).float().mean().item())
                per_class[name] = v
                vals.append(v)
            else:
                per_class[name] = float("nan")
        return {
            "acc": acc,
            "macro_acc_present": float(np.mean(vals)) if vals else float("nan"),
            "per_class_acc": per_class,
            "train_samples": int(train_y.numel()),
            "val_samples": int(val_y.numel()),
        }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    conditions = parse_conditions(args.conditions)
    if args.dry_run:
        print(json.dumps({"conditions": conditions, "data_root": str(args.data_root)}, indent=2))
        return

    print(f"[info] loading DINO {args.model_name}", flush=True)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    model.eval()
    print("[info] DINO loaded", flush=True)

    rows: list[dict] = []
    metadata: dict[str, dict] = {}
    for condition in conditions:
        print(f"[condition] {condition}", flush=True)
        train = collect_condition_features(args, args.train_split, condition, args.max_train_images, model)
        val = collect_condition_features(args, args.val_split, condition, args.max_val_images, model)
        result = fit_multiclass_probe(train["features"], train["labels"], val["features"], val["labels"], args)
        rows.append(
            {
                "condition": condition,
                "acc": result["acc"],
                "macro_acc_present": result["macro_acc_present"],
                "train_samples": result["train_samples"],
                "val_samples": result["val_samples"],
                "picture_acc": result["per_class_acc"].get("picture", float("nan")),
                "wall_acc": result["per_class_acc"].get("wall", float("nan")),
                "door_acc": result["per_class_acc"].get("door", float("nan")),
                "cabinet_acc": result["per_class_acc"].get("cabinet", float("nan")),
            }
        )
        metadata[condition] = {
            "train_seen_images": train["seen_images"],
            "val_seen_images": val["seen_images"],
            "train_class_counts": train["class_counts"],
            "val_class_counts": val["class_counts"],
        }

    clean = next((row for row in rows if row["condition"] == "clean"), None)
    csv_path = args.output_dir / "scene_dino_support_recognizability.csv"
    fieldnames = ["condition", "acc", "macro_acc_present", "delta_macro_vs_clean", "train_samples", "val_samples", "picture_acc", "wall_acc", "door_acc", "cabinet_acc"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["delta_macro_vs_clean"] = (
                row["macro_acc_present"] - clean["macro_acc_present"] if clean is not None else float("nan")
            )
            writer.writerow({k: out.get(k, "") for k in fieldnames})

    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    md_lines = [
        "# Scene DINO Support Recognizability",
        "",
        "DINO sees actual ScanNet RGB frames. Patch labels are derived only from points retained by each 3D support condition.",
        "This is a scene-level 2D semantic-evidence calibration, not a 3D segmentation result.",
        "",
        f"- DINO model: `{args.model_name}`",
        f"- train images max: `{args.max_train_images}`",
        f"- val images max: `{args.max_val_images}`",
        f"- structured cell size: `{args.structured_cell_size}` m",
        "",
        "| condition | patch acc | macro acc | delta macro vs clean | val samples | picture | wall | door | cabinet |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    clean_macro = clean["macro_acc_present"] if clean is not None else float("nan")
    for row in rows:
        md_lines.append(
            f"| `{row['condition']}` | {row['acc']:.4f} | {row['macro_acc_present']:.4f} | {row['macro_acc_present'] - clean_macro:.4f} | {row['val_samples']} | {row['picture_acc']:.4f} | {row['wall_acc']:.4f} | {row['door_acc']:.4f} | {row['cabinet_acc']:.4f} |"
        )
    md_lines += [
        "",
        "## Paper-safe interpretation",
        "",
        "- If a condition remains close to clean here, the corresponding RGB patches still carry semantic evidence under the stressed 3D support selection.",
        "- If a condition drops strongly here, it should not be used to claim that the 3D model missed obvious 2D evidence.",
        "- Because DINO sees full RGB patches, this calibrates 2D semantic evidence at retained-support locations; it is not an occluded-image VLM study.",
    ]
    (args.output_dir / "scene_dino_support_recognizability.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "md": str(args.output_dir / "scene_dino_support_recognizability.md")}, indent=2))


if __name__ == "__main__":
    main()
