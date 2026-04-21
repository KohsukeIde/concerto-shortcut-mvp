#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (
    SCANNET20_CLASS_NAMES,
    bootstrap_ci,
    fit_probe,
    pair_name,
    parse_pairs,
)


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(x)


class ScanNetRawSceneDataset(Dataset):
    def __init__(self, data_root: Path, split: str, transform) -> None:
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.scene_dirs = sorted(path for path in (data_root / split).iterdir() if path.is_dir())
        if not self.scene_dirs:
            raise RuntimeError(f"no scenes found under {data_root / split}")

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def __getitem__(self, index: int) -> dict:
        scene_dir = self.scene_dirs[index]
        point = {
            "coord": np.load(scene_dir / "coord.npy"),
            "color": np.load(scene_dir / "color.npy"),
            "normal": np.load(scene_dir / "normal.npy"),
            "segment": np.load(scene_dir / "segment20.npy"),
        }
        raw_segment = torch.from_numpy(point["segment"]).long()
        out = self.transform(point)
        out["segment"] = torch.from_numpy(point["segment"]).long()
        out["raw_segment"] = raw_segment
        out["scene_name"] = scene_dir.name
        return out


def collate_one(batch: list[dict]) -> dict:
    assert len(batch) == 1
    return batch[0]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utonia ScanNet point-stagewise trace")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--utonia-weight", type=Path, required=True)
    parser.add_argument("--seg-head-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,door:wall,counter:cabinet",
        help="Comma-separated positive:negative class-name pairs.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=128)
    parser.add_argument("--max-val-batches", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--logreg-steps", type=int, default=600)
    parser.add_argument("--logreg-lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--disable-flash", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(data_root: Path, split: str, batch_size: int, num_worker: int):
    import utonia

    transform = utonia.transform.default(0.5)
    dataset = ScanNetRawSceneDataset(data_root, split, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        collate_fn=collate_one,
    )


def build_model(utonia_weight: Path, seg_head_weight: Path, disable_flash: bool):
    import utonia

    head_ckpt = utonia.load(str(seg_head_weight), ckpt_only=True)
    seg_head = SegHead(**head_ckpt["config"])
    seg_head.load_state_dict(head_ckpt["state_dict"])
    custom_config = None
    if disable_flash:
        custom_config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = utonia.load(str(utonia_weight), custom_config=custom_config)
    return model.cuda().eval(), seg_head.cuda().eval()


def move_to_cuda(input_dict: dict) -> dict:
    skip_keys = {"raw_segment", "scene_name"}
    for key, value in input_dict.items():
        if key in skip_keys:
            continue
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.cuda(non_blocking=True)
    return input_dict


@torch.no_grad()
def extract_batch(model, seg_head, batch: dict, target_classes: set[int], split: str):
    out = model(batch)
    while "pooling_parent" in out.keys():
        parent = out.pop("pooling_parent")
        inverse = out.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
        out = parent
    feat = out.feat.float()
    logits = seg_head(feat).float()
    labels = batch["segment"].long()

    if split == "val":
        inverse = batch["inverse"].long().cpu()
        feat = feat.cpu()[inverse]
        logits = logits.cpu()[inverse]
        labels = batch["raw_segment"].long()
    else:
        feat = feat.cpu()
        logits = logits.cpu()
        labels = labels.cpu()

    if feat.shape[0] != labels.shape[0] or logits.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"feature/logit/label length mismatch: feat={feat.shape} logits={logits.shape} labels={labels.shape}"
        )
    valid = (labels >= 0) & (labels < len(SCANNET20_CLASS_NAMES))
    target_mask = torch.zeros_like(valid)
    for cls in target_classes:
        target_mask |= labels == cls
    keep = valid & target_mask
    if not keep.any():
        return None
    return {
        "point_feature": feat[keep].float(),
        "linear_logits": logits[keep].float(),
        "class_id": labels[keep].long(),
        "pred_id": logits[keep].argmax(dim=1).long(),
    }


def collect_split(args: argparse.Namespace, model, seg_head, loader, split: str, max_batches: int, target_classes: set[int]):
    features = {"point_feature": [], "linear_logits": []}
    labels: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    class_counts = {cls: 0 for cls in sorted(target_classes)}
    seen_batches = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches >= 0 and batch_idx >= max_batches:
                break
            batch = move_to_cuda(batch)
            out = extract_batch(model, seg_head, batch, target_classes=target_classes, split=split)
            seen_batches += 1
            if out is None:
                continue
            for cls in sorted(target_classes):
                mask = out["class_id"] == cls
                if not mask.any():
                    continue
                room = args.max_per_class - class_counts[cls]
                if room <= 0:
                    continue
                idx = mask.nonzero(as_tuple=False).flatten()
                if idx.numel() > room:
                    idx = idx[:room]
                for name in features:
                    features[name].append(out[name][idx])
                labels.append(out["class_id"][idx])
                preds.append(out["pred_id"][idx])
                class_counts[cls] += int(idx.numel())
            if (batch_idx + 1) % 10 == 0:
                counts = " ".join(f"{SCANNET20_CLASS_NAMES[k]}={v}" for k, v in class_counts.items())
                print(f"[collect] split={split} batch={batch_idx + 1} {counts}", flush=True)
    if not labels:
        raise RuntimeError(f"no target point samples collected for split={split}")
    payload = {
        "class_id": torch.cat(labels, dim=0),
        "pred_id": torch.cat(preds, dim=0),
        "seen_batches": seen_batches,
        "class_counts": {SCANNET20_CLASS_NAMES[k]: int(v) for k, v in class_counts.items()},
    }
    for name, tensors in features.items():
        payload[name] = torch.cat(tensors, dim=0)
    print(f"[collect] split={split} done counts={payload['class_counts']}", flush=True)
    return payload


def direct_pair_metrics(scores: torch.Tensor, labels: torch.Tensor, iters: int, seed: int):
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu()
    pred = (scores >= 0).float()
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_acc = (pred[pos_mask] == labels[pos_mask]).float().mean().item() if pos_mask.any() else float("nan")
    neg_acc = (pred[neg_mask] == labels[neg_mask]).float().mean().item() if neg_mask.any() else float("nan")
    row = {
        "acc": (pred == labels).float().mean().item(),
        "balanced_acc": float(np.nanmean([pos_acc, neg_acc])),
        "positive_acc": pos_acc,
        "negative_acc": neg_acc,
        "auc": float("nan"),
        "train_samples_probe": 0,
        "pos_weight": "",
    }
    row.update(bootstrap_ci(scores, labels, iters, seed))
    return row


def evaluate_pair(pair, train, val, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    pos_cls, neg_cls = pair
    train_mask = (train["class_id"] == pos_cls) | (train["class_id"] == neg_cls)
    val_mask = (val["class_id"] == pos_cls) | (val["class_id"] == neg_cls)
    train_y = (train["class_id"][train_mask] == pos_cls).float()
    val_y = (val["class_id"][val_mask] == pos_cls).float()
    if (
        int((train_y == 1).sum().item()) == 0
        or int((train_y == 0).sum().item()) == 0
        or int((val_y == 1).sum().item()) == 0
        or int((val_y == 0).sum().item()) == 0
    ):
        print(
            "[skip] {pair} train_pos={train_pos} train_neg={train_neg} val_pos={val_pos} val_neg={val_neg}".format(
                pair=pair_name(pair),
                train_pos=int((train_y == 1).sum().item()),
                train_neg=int((train_y == 0).sum().item()),
                val_pos=int((val_y == 1).sum().item()),
                val_neg=int((val_y == 0).sum().item()),
            ),
            flush=True,
        )
        return [], []
    rows: list[dict] = []
    for feature in ("point_feature", "linear_logits"):
        for probe in ("unweighted", "balanced", "weighted"):
            result = fit_probe(train[feature][train_mask], train_y, val[feature][val_mask], val_y, args, probe)
            ci = bootstrap_ci(
                result.pop("logits"),
                result.pop("labels"),
                args.bootstrap_iters,
                args.seed + len(rows) + 211,
            )
            row = {
                "pair": pair_name(pair),
                "positive_class": SCANNET20_CLASS_NAMES[pos_cls],
                "negative_class": SCANNET20_CLASS_NAMES[neg_cls],
                "stage": feature,
                "probe": probe,
                "train_positive": int((train_y == 1).sum().item()),
                "train_negative": int((train_y == 0).sum().item()),
                "val_positive": int((val_y == 1).sum().item()),
                "val_negative": int((val_y == 0).sum().item()),
            }
            row.update(result)
            row.update(ci)
            rows.append(row)

    direct_scores = val["linear_logits"][val_mask, pos_cls] - val["linear_logits"][val_mask, neg_cls]
    direct = direct_pair_metrics(direct_scores, val_y, args.bootstrap_iters, args.seed + 409)
    direct_row = {
        "pair": pair_name(pair),
        "positive_class": SCANNET20_CLASS_NAMES[pos_cls],
        "negative_class": SCANNET20_CLASS_NAMES[neg_cls],
        "stage": "linear_logits",
        "probe": "direct_pair_margin",
        "train_positive": int((train_y == 1).sum().item()),
        "train_negative": int((train_y == 0).sum().item()),
        "val_positive": int((val_y == 1).sum().item()),
        "val_negative": int((val_y == 0).sum().item()),
    }
    direct_row.update(direct)
    rows.append(direct_row)

    confusion_rows: list[dict] = []
    val_labels = val["class_id"][val_mask]
    val_preds = val["pred_id"][val_mask]
    for target in (pos_cls, neg_cls):
        target_mask = val_labels == target
        denom = int(target_mask.sum().item())
        if denom == 0:
            continue
        pred_counts = torch.bincount(val_preds[target_mask], minlength=len(SCANNET20_CLASS_NAMES))
        for pred_id, count_tensor in enumerate(pred_counts.tolist()):
            if count_tensor == 0:
                continue
            confusion_rows.append(
                {
                    "pair": pair_name(pair),
                    "target_id": target,
                    "target_name": SCANNET20_CLASS_NAMES[target],
                    "pred_id": pred_id,
                    "pred_name": SCANNET20_CLASS_NAMES[pred_id],
                    "count": int(count_tensor),
                    "fraction_of_target": float(count_tensor / max(denom, 1)),
                }
            )
    return rows, confusion_rows


def write_outputs(args: argparse.Namespace, rows: list[dict], confusion_rows: list[dict], metadata: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "utonia_scannet_point_stagewise_trace.csv"
    fields = [
        "pair",
        "positive_class",
        "negative_class",
        "stage",
        "probe",
        "balanced_acc",
        "bal_acc_std",
        "bal_acc_ci_low",
        "bal_acc_ci_high",
        "auc",
        "auc_std",
        "auc_ci_low",
        "auc_ci_high",
        "positive_acc",
        "negative_acc",
        "acc",
        "train_positive",
        "train_negative",
        "val_positive",
        "val_negative",
        "train_samples_probe",
        "pos_weight",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    confusion_path = args.output_dir / "utonia_scannet_point_stagewise_trace_confusion.csv"
    with confusion_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair", "target_id", "target_name", "pred_id", "pred_name", "count", "fraction_of_target"],
        )
        writer.writeheader()
        writer.writerows(confusion_rows)

    md_path = args.output_dir / "utonia_scannet_point_stagewise_trace.md"
    lines = [
        "# Utonia ScanNet Point-Level Stage-Wise Trace",
        "",
        "## Setup",
        f"- utonia weight: `{args.utonia_weight}`",
        f"- seg head weight: `{args.seg_head_weight}`",
        f"- data root: `{args.data_root}`",
        f"- train batches seen: {metadata['train']['seen_batches']}",
        f"- val batches seen: {metadata['val']['seen_batches']}",
        f"- train class counts: {metadata['train']['class_counts']}",
        f"- val class counts: {metadata['val']['class_counts']}",
        f"- bootstrap iters: {args.bootstrap_iters}",
        "",
        "## Results",
        "",
        "| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {pair} | {stage} | {probe} | {balanced_acc:.4f} | "
            "[{bal_acc_ci_low:.4f}, {bal_acc_ci_high:.4f}] | {auc:.4f} | "
            "[{auc_ci_low:.4f}, {auc_ci_high:.4f}] | {val_positive}/{val_negative} |".format(**row)
        )
    lines += [
        "",
        "## Notes",
        "- `point_feature` is the Utonia point feature after unpooling the released backbone to the segmentation-head resolution.",
        "- `linear_logits` uses the released ScanNet linear probing head bundled with Utonia.",
        "- Validation rows are expanded back to raw points through the transform `inverse` mapping so the trace matches the original scene labels.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[write] {csv_path}", flush=True)
    print(f"[write] {confusion_path}", flush=True)
    print(f"[write] {md_path}", flush=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    seed_everything(args.seed)
    pairs = parse_pairs(args.class_pairs)
    target_classes = {cls for pair in pairs for cls in pair}
    data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    utonia_weight = (repo_root / args.utonia_weight).resolve() if not args.utonia_weight.is_absolute() else args.utonia_weight
    seg_head_weight = (repo_root / args.seg_head_weight).resolve() if not args.seg_head_weight.is_absolute() else args.seg_head_weight
    print(f"[pairs] {[pair_name(pair) for pair in pairs]}", flush=True)

    if args.dry_run:
        loader = build_loader(data_root, args.val_split, 1, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] scene_name={batch['scene_name']}", flush=True)
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Utonia stagewise trace")

    model, seg_head = build_model(utonia_weight, seg_head_weight, args.disable_flash)
    train_loader = build_loader(data_root, args.train_split, args.batch_size, args.num_worker)
    val_loader = build_loader(data_root, args.val_split, args.batch_size, args.num_worker)
    train = collect_split(args, model, seg_head, train_loader, args.train_split, args.max_train_batches, target_classes)
    val = collect_split(args, model, seg_head, val_loader, args.val_split, args.max_val_batches, target_classes)
    rows: list[dict] = []
    confusion_rows: list[dict] = []
    for pair in pairs:
        pair_rows, pair_confusion = evaluate_pair(pair, train, val, args)
        rows.extend(pair_rows)
        confusion_rows.extend(pair_confusion)
    metadata = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pairs": [pair_name(pair) for pair in pairs],
        "train": {"seen_batches": train["seen_batches"], "class_counts": train["class_counts"]},
        "val": {"seen_batches": val["seen_batches"], "class_counts": val["class_counts"]},
    }
    write_outputs(args, rows, confusion_rows, metadata)
    for row in rows:
        if row["probe"] in {"balanced", "direct_pair_margin"}:
            print(
                "[result] {pair}/{stage}/{probe} bal={balanced_acc:.4f} "
                "ci=[{bal_acc_ci_low:.4f},{bal_acc_ci_high:.4f}] auc={auc:.4f}".format(**row),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
