#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
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


def add_stress_transform(val_cfg, stress: str, voxel_size: float):
    val_cfg = copy.deepcopy(val_cfg)
    if stress == "clean":
        return val_cfg
    transform = list(val_cfg.transform)
    stress_cfg = dict(type="CoordStress", stress=stress, voxel_size=voxel_size)
    insert_at = 1
    if stress.startswith("xy_shift_post_"):
        for idx, item in enumerate(transform):
            if item.get("type") == "CenterShift" and item.get("apply_z") is False:
                insert_at = idx + 1
                break
    else:
        for idx, item in enumerate(transform):
            if item.get("type") == "Copy":
                insert_at = idx + 1
                break
    transform.insert(insert_at, stress_cfg)
    val_cfg.transform = transform
    return val_cfg


def build_loader(cfg, stress: str, voxel_size: float, batch_size: int, num_worker: int):
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn

    val_cfg = add_stress_transform(cfg.data.val, stress=stress, voxel_size=voxel_size)
    dataset = build_dataset(val_cfg)
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


def evaluate_stress(model, cfg, stress: str, voxel_size: float, batch_size: int, num_worker: int, max_batches: int):
    from pointcept.utils.misc import intersection_and_union_gpu

    loader = build_loader(cfg, stress=stress, voxel_size=voxel_size, batch_size=batch_size, num_worker=num_worker)
    intersection_sum = torch.zeros(cfg.data.num_classes, dtype=torch.float64, device="cuda")
    union_sum = torch.zeros_like(intersection_sum)
    target_sum = torch.zeros_like(intersection_sum)
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
            segment = input_dict["segment"]
            if "inverse" in input_dict:
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                cfg.data.num_classes,
                cfg.data.ignore_index,
            )
            intersection_sum += intersection.double()
            union_sum += union.double()
            target_sum += target.double()
            if loss is not None:
                loss_sum += float(loss.item())
            count += 1
            if count % 25 == 0:
                miou_so_far = (intersection_sum / (union_sum + 1e-10)).mean().item()
                print(f"[eval] stress={stress} batches={count} mIoU_so_far={miou_so_far:.4f}", flush=True)

    iou_class = (intersection_sum / (union_sum + 1e-10)).detach().cpu().numpy()
    acc_class = (intersection_sum / (target_sum + 1e-10)).detach().cpu().numpy()
    m_iou = float(np.mean(iou_class))
    m_acc = float(np.mean(acc_class))
    all_acc = float(intersection_sum.sum().item() / (target_sum.sum().item() + 1e-10))
    class_rows = []
    names = list(getattr(cfg.data, "names", [f"class_{i}" for i in range(cfg.data.num_classes)]))
    for idx, name in enumerate(names):
        class_rows.append(
            {
                "stress": stress,
                "class_id": idx,
                "class_name": name,
                "iou": float(iou_class[idx]),
                "accuracy": float(acc_class[idx]),
                "target_count": float(target_sum[idx].item()),
            }
        )
    return {
        "stress": stress,
        "batches": count,
        "mIoU": m_iou,
        "mAcc": m_acc,
        "allAcc": all_acc,
        "loss": loss_sum / max(count, 1),
        "_class_rows": class_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ScanNet semseg checkpoints under coordinate stress.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stress",
        nargs="+",
        default=["clean", "local_surface_destroy", "z_flip", "xy_swap", "roll_90_x"],
    )
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = load_cfg(args.repo_root.resolve(), args.config)
    model = build_model(cfg, args.weight.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for stress in args.stress:
        print(f"[start] stress={stress}", flush=True)
        rows.append(
            evaluate_stress(
                model,
                cfg,
                stress=stress,
                voxel_size=args.voxel_size,
                batch_size=args.batch_size,
                num_worker=args.num_worker,
                max_batches=args.max_batches,
            )
        )
        print(f"[done] {rows[-1]}", flush=True)

    write_rows = []
    class_rows = []
    for row in rows:
        row = dict(row)
        class_rows.extend(row.pop("_class_rows"))
        write_rows.append(row)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stress", "batches", "mIoU", "mAcc", "allAcc", "loss"])
        writer.writeheader()
        writer.writerows(write_rows)
    if args.output.name.endswith(".csv.tmp"):
        class_name = args.output.name[: -len(".csv.tmp")] + "_classwise.csv.tmp"
    else:
        class_name = args.output.stem + "_classwise.csv"
    class_output = args.output.with_name(class_name)
    with class_output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stress", "class_id", "class_name", "iou", "accuracy", "target_count"],
        )
        writer.writeheader()
        writer.writerows(class_rows)
    print(f"[write] {args.output}", flush=True)
    print(f"[write] {class_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
