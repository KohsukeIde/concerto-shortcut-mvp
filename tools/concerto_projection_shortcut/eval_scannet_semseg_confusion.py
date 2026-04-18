#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cfg(repo_root: Path, config_name: str):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pointcept.utils.config import Config

    return Config.fromfile(str(repo_root / "configs" / "concerto" / f"{config_name}.py"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(cfg, batch_size: int, num_worker: int):
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn

    dataset = build_dataset(cfg.data.val)
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
    weight = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        weight[key] = value
    load_info = model.load_state_dict(weight, strict=True)
    print(
        "[load_model]",
        f"weight={weight_path}",
        f"epoch={checkpoint.get('epoch', 'NA')}",
        f"missing={len(load_info.missing_keys)}",
        f"unexpected={len(load_info.unexpected_keys)}",
        flush=True,
    )
    return model


def move_to_cuda(input_dict):
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.cuda(non_blocking=True)
    return input_dict


def evaluate_confusion(model, cfg, batch_size: int, num_worker: int, max_batches: int):
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    loader = build_loader(cfg, batch_size=batch_size, num_worker=num_worker)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")
    loss_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, input_dict in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            input_dict = move_to_cuda(input_dict)
            output_dict = model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict.get("loss")
            pred = output.max(1)[1]
            target = input_dict["segment"]
            if "inverse" in input_dict:
                pred = pred[input_dict["inverse"]]
                target = input_dict["origin_segment"]
            valid = target != ignore_index
            pred = pred[valid].long()
            target = target[valid].long()
            in_range = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
            pred = pred[in_range]
            target = target[in_range]
            flat = target * num_classes + pred
            confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
            if loss is not None:
                loss_sum += float(loss.item())
            count += 1
            if count % 25 == 0:
                inter = confusion.diag().double()
                target_sum = confusion.sum(dim=1).double()
                pred_sum = confusion.sum(dim=0).double()
                union = target_sum + pred_sum - inter
                miou = (inter / (union + 1e-10)).mean().item()
                print(f"[eval] batches={count} mIoU_so_far={miou:.4f}", flush=True)
    return confusion.detach().cpu().numpy(), count, loss_sum / max(count, 1)


def write_outputs(confusion: np.ndarray, cfg, label: str, batches: int, loss: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(cfg.data.names)
    inter = np.diag(confusion).astype(np.float64)
    target_sum = confusion.sum(axis=1).astype(np.float64)
    pred_sum = confusion.sum(axis=0).astype(np.float64)
    union = target_sum + pred_sum - inter
    iou = inter / (union + 1e-10)
    acc = inter / (target_sum + 1e-10)
    m_iou = float(np.mean(iou))
    m_acc = float(np.mean(acc))
    all_acc = float(inter.sum() / (target_sum.sum() + 1e-10))

    summary = {
        "label": label,
        "batches": batches,
        "loss": loss,
        "mIoU": m_iou,
        "mAcc": m_acc,
        "allAcc": all_acc,
    }
    (output_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    with (output_dir / f"{label}_class_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "intersection",
                "union",
                "target_count",
                "pred_count",
                "iou",
                "accuracy",
                "target_share",
                "pred_share",
            ],
        )
        writer.writeheader()
        total_target = float(target_sum.sum() + 1e-10)
        total_pred = float(pred_sum.sum() + 1e-10)
        for idx, name in enumerate(names):
            writer.writerow(
                {
                    "class_id": idx,
                    "class_name": name,
                    "intersection": int(inter[idx]),
                    "union": int(union[idx]),
                    "target_count": int(target_sum[idx]),
                    "pred_count": int(pred_sum[idx]),
                    "iou": f"{iou[idx]:.8f}",
                    "accuracy": f"{acc[idx]:.8f}",
                    "target_share": f"{target_sum[idx] / total_target:.8f}",
                    "pred_share": f"{pred_sum[idx] / total_pred:.8f}",
                }
            )

    with (output_dir / f"{label}_confusion_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_id",
                "target_name",
                "pred_id",
                "pred_name",
                "count",
                "fraction_of_target",
            ],
        )
        writer.writeheader()
        for target_id, target_name in enumerate(names):
            denom = target_sum[target_id] + 1e-10
            for pred_id, pred_name in enumerate(names):
                count = int(confusion[target_id, pred_id])
                if count == 0:
                    continue
                writer.writerow(
                    {
                        "target_id": target_id,
                        "target_name": target_name,
                        "pred_id": pred_id,
                        "pred_name": pred_name,
                        "count": count,
                        "fraction_of_target": f"{count / denom:.8f}",
                    }
                )

    with (output_dir / f"{label}_top_confusions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_id",
                "target_name",
                "rank",
                "pred_id",
                "pred_name",
                "count",
                "fraction_of_target",
            ],
        )
        writer.writeheader()
        for target_id, target_name in enumerate(names):
            row = confusion[target_id].astype(np.int64).copy()
            row[target_id] = 0
            order = np.argsort(-row)
            denom = target_sum[target_id] + 1e-10
            rank = 0
            for pred_id in order:
                count = int(row[pred_id])
                if count == 0:
                    continue
                rank += 1
                if rank > 5:
                    break
                writer.writerow(
                    {
                        "target_id": target_id,
                        "target_name": target_name,
                        "rank": rank,
                        "pred_id": int(pred_id),
                        "pred_name": names[int(pred_id)],
                        "count": count,
                        "fraction_of_target": f"{count / denom:.8f}",
                    }
                )
    print(f"[write] {output_dir}", flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ScanNet class metrics and confusion matrix.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = load_cfg(args.repo_root.resolve(), args.config)
    model = build_model(cfg, args.weight.resolve())
    confusion, batches, loss = evaluate_confusion(
        model=model,
        cfg=cfg,
        batch_size=args.batch_size,
        num_worker=args.num_worker,
        max_batches=args.max_batches,
    )
    write_outputs(confusion, cfg, args.label, batches, loss, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
