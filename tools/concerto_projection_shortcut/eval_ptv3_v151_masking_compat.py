#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


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


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    keep_ratio: float = 1.0
    feature_zero_ratio: float = 0.0
    block_size: int = 64


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate released PTv3 v1.5.1 checkpoints with the official v1.5.1 "
            "model/transform code while reading the current npy ScanNet root. "
            "This isolates code/protocol mismatch from retraining."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--official-root", type=Path, default=Path("data/tmp/Pointcept-v1.5.1"))
    parser.add_argument("--config", default="configs/scannet/semseg-pt-v3m1-0-base.py")
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--method-name", default="ptv3_supervised_v151_compat")
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--segment-key", default="segment20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--random-keep-ratios", default="0.2")
    parser.add_argument("--classwise-keep-ratios", default="")
    parser.add_argument("--structured-keep-ratios", default="")
    parser.add_argument("--feature-zero-ratios", default="")
    parser.add_argument("--structured-block-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--full-scene-scoring",
        action="store_true",
        help=(
            "Also score every original voxel by nearest-neighbor propagation "
            "from retained masked-input logits. Retained-subset rows are still written."
        ),
    )
    parser.add_argument(
        "--full-scene-chunk-size",
        type=int,
        default=2048,
        help="Fallback chunk size for torch.cdist nearest-neighbor propagation.",
    )
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument(
        "--class-names",
        default="",
        help="Optional comma-separated class names. Defaults to cfg.data.names, then ScanNet20 names.",
    )
    parser.add_argument("--focus-class", default="picture")
    parser.add_argument("--confusion-class", default="wall")
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_ptv3_v151_masking_compat"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_official_imports(official_root: Path):
    sys.path.insert(0, str(official_root.resolve()))
    from pointcept.datasets.transform import Compose  # noqa: PLC0415
    from pointcept.models.builder import build_model  # noqa: PLC0415
    from pointcept.utils.config import Config  # noqa: PLC0415

    return Config, Compose, build_model


def load_config(config_cls, config_path: Path):
    return config_cls.fromfile(str(config_path))


def build_official_model(build_model_fn, cfg, weight_path: Path):
    model = build_model_fn(cfg.model).cuda().eval()
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


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants = [Variant("clean_voxel", "clean")]
    for keep in parse_float_list(args.random_keep_ratios):
        variants.append(Variant(f"random_keep{str(keep).replace('.', 'p')}", "random_drop", keep_ratio=keep))
    for keep in parse_float_list(args.classwise_keep_ratios):
        variants.append(Variant(f"classwise_keep{str(keep).replace('.', 'p')}", "classwise_random_drop", keep_ratio=keep))
    for keep in parse_float_list(args.structured_keep_ratios):
        variants.append(
            Variant(
                f"structured_b{args.structured_block_size}_keep{str(keep).replace('.', 'p')}",
                "structured_drop",
                keep_ratio=keep,
                block_size=args.structured_block_size,
            )
        )
    for ratio in parse_float_list(args.feature_zero_ratios):
        variants.append(Variant(f"feature_zero{str(ratio).replace('.', 'p')}", "feature_zero", feature_zero_ratio=ratio))
    return variants


def scene_paths(data_root: Path, split: str) -> list[Path]:
    split_path = data_root / split
    return sorted(p for p in split_path.iterdir() if p.is_dir())


def class_names_from_cfg(cfg, args: argparse.Namespace) -> list[str]:
    if args.class_names.strip():
        names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    elif hasattr(cfg, "data") and hasattr(cfg.data, "names"):
        names = list(cfg.data.names)
    else:
        names = SCANNET20_CLASS_NAMES
    if len(names) != args.num_classes:
        raise ValueError(f"class name count {len(names)} does not match num_classes={args.num_classes}")
    return names


def load_scene(path: Path, segment_key: str) -> dict:
    segment_path = path / f"{segment_key}.npy"
    if not segment_path.exists():
        raise FileNotFoundError(f"missing segment file: {segment_path}")
    segment = np.load(segment_path).reshape([-1]).astype(np.int32)
    data = {
        "coord": np.load(path / "coord.npy").astype(np.float32),
        "color": np.load(path / "color.npy").astype(np.float32),
        "normal": np.load(path / "normal.npy").astype(np.float32),
        "segment": segment,
        "instance": np.load(path / "instance.npy").reshape([-1]).astype(np.int32)
        if (path / "instance.npy").exists()
        else np.ones(segment.shape[0], dtype=np.int32) * -1,
        "scene_id": path.name,
    }
    return data


def move_to_cuda(batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def clone_batch(batch: dict) -> dict:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def input_point_count(batch: dict) -> int:
    return int(batch["segment"].shape[0])


def filter_batch(batch: dict, mask: torch.Tensor) -> dict:
    n = input_point_count(batch)
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (n,):
            out[key] = value[mask]
        elif key == "offset" and isinstance(value, torch.Tensor):
            out[key] = torch.tensor([int(mask.sum().item())], dtype=value.dtype, device=value.device)
        else:
            out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def random_drop_mask(n: int, keep_ratio: float, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    keep = torch.rand(n, device=device, generator=generator) < keep_ratio
    if not bool(keep.any()):
        keep[torch.randint(n, (1,), device=device, generator=generator)] = True
    return keep


def classwise_drop_mask(labels: torch.Tensor, keep_ratio: float, generator: torch.Generator) -> torch.Tensor:
    keep = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    for cls in labels.unique(sorted=True):
        cls_value = int(cls.item())
        cls_mask = labels == cls
        idx = torch.where(cls_mask)[0]
        cls_keep = torch.rand(idx.shape[0], device=labels.device, generator=generator) < keep_ratio
        if cls_value >= 0 and not bool(cls_keep.any()):
            cls_keep[torch.randint(idx.shape[0], (1,), device=labels.device, generator=generator)] = True
        keep[idx] = cls_keep
    if not bool(keep.any()):
        keep[torch.randint(labels.shape[0], (1,), device=labels.device, generator=generator)] = True
    return keep


def structured_drop_mask(grid: torch.Tensor, keep_ratio: float, block_size: int, generator: torch.Generator) -> torch.Tensor:
    keys = torch.div(grid.long(), block_size, rounding_mode="floor")
    _, inv = torch.unique(keys, dim=0, return_inverse=True)
    n_region = int(inv.max().item()) + 1
    region_keep = torch.rand(n_region, device=grid.device, generator=generator) < keep_ratio
    if not bool(region_keep.any()):
        region_keep[torch.randint(n_region, (1,), device=grid.device, generator=generator)] = True
    return region_keep[inv]


def make_variant_batch_with_mask(batch: dict, variant: Variant, seed: int) -> tuple[dict, float, torch.Tensor]:
    n = input_point_count(batch)
    device = batch["segment"].device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if variant.kind == "clean":
        keep = torch.ones(n, dtype=torch.bool, device=device)
        return clone_batch(batch), 1.0, keep
    if variant.kind == "random_drop":
        mask = random_drop_mask(n, variant.keep_ratio, device, generator)
        return filter_batch(batch, mask), float(mask.float().mean().item()), mask
    if variant.kind == "classwise_random_drop":
        mask = classwise_drop_mask(batch["segment"].long(), variant.keep_ratio, generator)
        return filter_batch(batch, mask), float(mask.float().mean().item()), mask
    if variant.kind == "structured_drop":
        mask = structured_drop_mask(batch["grid_coord"], variant.keep_ratio, variant.block_size, generator)
        return filter_batch(batch, mask), float(mask.float().mean().item()), mask
    if variant.kind == "feature_zero":
        out = clone_batch(batch)
        zero = random_drop_mask(n, variant.feature_zero_ratio, device, generator)
        if "feat" in out:
            out["feat"][zero] = 0
        keep = torch.ones(n, dtype=torch.bool, device=device)
        return out, 1.0, keep
    raise ValueError(f"unknown variant kind: {variant.kind}")


def make_variant_batch(batch: dict, variant: Variant, seed: int) -> tuple[dict, float]:
    masked_batch, keep_frac, _ = make_variant_batch_with_mask(batch, variant, seed)
    return masked_batch, keep_frac


def inference_batch(batch: dict) -> dict:
    return {key: value for key, value in batch.items() if key not in {"segment"}}


@torch.no_grad()
def forward_logits_labels(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["segment"].long()
    out = model(inference_batch(batch))
    logits = out["seg_logits"].float()
    if logits.shape[0] != labels.shape[0]:
        raise RuntimeError(f"shape mismatch logits={logits.shape} labels={labels.shape}")
    return logits, labels


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    valid = target != ignore_index
    pred = pred[valid].long()
    target = target[valid].long()
    in_range = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[in_range]
    target = target[in_range]
    flat = target * num_classes + pred
    confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def offsets_from_keep_mask(original_offset: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    starts = torch.cat([original_offset.new_zeros(1), original_offset[:-1]])
    kept = []
    for start, end in zip(starts.tolist(), original_offset.tolist()):
        kept.append(int(keep_mask[start:end].sum().item()))
    counts = torch.tensor(kept, dtype=original_offset.dtype, device=original_offset.device)
    return torch.cumsum(counts, dim=0)


def _nearest_index_fallback(
    query_coord: torch.Tensor,
    support_coord: torch.Tensor,
    query_offset: torch.Tensor,
    support_offset: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    starts_q = torch.cat([query_offset.new_zeros(1), query_offset[:-1]])
    starts_s = torch.cat([support_offset.new_zeros(1), support_offset[:-1]])
    out = torch.empty(query_coord.shape[0], dtype=torch.long, device=query_coord.device)
    for q_start, q_end, s_start, s_end in zip(starts_q.tolist(), query_offset.tolist(), starts_s.tolist(), support_offset.tolist()):
        support = support_coord[s_start:s_end].float()
        if support.shape[0] == 0:
            raise RuntimeError("empty retained support for full-scene propagation")
        for begin in range(q_start, q_end, chunk_size):
            end = min(begin + chunk_size, q_end)
            dist = torch.cdist(query_coord[begin:end].float(), support)
            out[begin:end] = dist.argmin(dim=1) + s_start
    return out


def nearest_retained_index(
    query_coord: torch.Tensor,
    support_coord: torch.Tensor,
    query_offset: torch.Tensor,
    support_offset: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    try:
        import pointops  # noqa: PLC0415

        index, _ = pointops.knn_query(
            1,
            support_coord.float(),
            support_offset.int(),
            query_coord.float(),
            query_offset.int(),
        )
        return index.flatten().long()
    except Exception as exc:
        print(f"[warn] pointops knn_query failed; using torch.cdist fallback: {exc}", flush=True)
        return _nearest_index_fallback(query_coord, support_coord, query_offset, support_offset, chunk_size)


def full_scene_logits_from_retained(logits: torch.Tensor, batch: dict, keep_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    n = input_point_count(batch)
    if logits.shape[0] == n and bool(keep_mask.all()):
        return logits
    original_offset = batch.get("offset")
    if original_offset is None:
        original_offset = torch.tensor([n], dtype=torch.int32, device=logits.device)
    kept_offset = offsets_from_keep_mask(original_offset, keep_mask)
    support_coord = batch["coord"][keep_mask]
    nearest = nearest_retained_index(
        batch["coord"],
        support_coord,
        original_offset,
        kept_offset,
        chunk_size,
    )
    return logits[nearest]


def summarize_confusion(conf: np.ndarray):
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
        "conf": conf,
    }


def focus_to_confusion_from_conf(conf: np.ndarray, focus_id: int | None, confusion_id: int | None) -> float:
    if focus_id is None or confusion_id is None:
        return float("nan")
    denom = conf[focus_id].sum()
    return float(conf[focus_id, confusion_id] / denom) if denom else float("nan")


def eval_masking(args: argparse.Namespace, model, transform, scene_list: list[Path], variants: list[Variant]):
    score_spaces = ["retained"]
    if args.full_scene_scoring:
        score_spaces.append("full_nn")
    confs = {
        (space, v.name): torch.zeros((args.num_classes, args.num_classes), dtype=torch.long)
        for space in score_spaces
        for v in variants
    }
    keep_sums = {v.name: 0.0 for v in variants}
    keep_counts = {v.name: 0 for v in variants}
    with torch.inference_mode():
        for scene_idx, scene_path in enumerate(scene_list):
            if args.max_val_batches >= 0 and scene_idx >= args.max_val_batches:
                break
            batch = transform(load_scene(scene_path, args.segment_key))
            batch = move_to_cuda(batch)
            for variant in variants:
                repeat_count = args.repeats if variant.kind in {"random_drop", "classwise_random_drop", "structured_drop", "feature_zero"} else 1
                for repeat_idx in range(repeat_count):
                    seed = args.seed + scene_idx * 1009 + repeat_idx * 9176 + abs(hash(variant.name)) % 1000
                    masked_batch, keep_frac, keep_mask = make_variant_batch_with_mask(batch, variant, seed)
                    logits, labels = forward_logits_labels(model, masked_batch)
                    update_confusion(
                        confs[("retained", variant.name)],
                        logits.argmax(dim=1).cpu(),
                        labels.cpu(),
                        args.num_classes,
                        -1,
                    )
                    if args.full_scene_scoring:
                        full_logits = full_scene_logits_from_retained(logits, batch, keep_mask, args.full_scene_chunk_size)
                        update_confusion(
                            confs[("full_nn", variant.name)],
                            full_logits.argmax(dim=1).cpu(),
                            batch["segment"].long().cpu(),
                            args.num_classes,
                            -1,
                        )
                    keep_sums[variant.name] += keep_frac
                    keep_counts[variant.name] += 1
            if (scene_idx + 1) % 25 == 0:
                clean = summarize_confusion(confs[("retained", "clean_voxel")].numpy())
                print(f"[val] scene={scene_idx+1} clean_voxel_mIoU={clean['mIoU']:.4f}", flush=True)
    return {
        "conf": {f"{space}:{name}": v.numpy() for (space, name), v in confs.items()},
        "score_spaces": score_spaces,
        "keep_sums": keep_sums,
        "keep_counts": keep_counts,
    }


def summary_row(
    method: str,
    variant: Variant,
    repeat_count: int,
    mean_keep: float,
    conf: np.ndarray,
    base: dict,
    focus_id: int | None,
    confusion_id: int | None,
) -> dict:
    s = summarize_confusion(conf)
    focus_iou = float(s["iou"][focus_id]) if focus_id is not None else float("nan")
    base_focus_iou = float(base["iou"][focus_id]) if focus_id is not None else float("nan")
    confusion_iou = float(s["iou"][confusion_id]) if confusion_id is not None else float("nan")
    base_focus_to_conf = focus_to_confusion_from_conf(base["conf"], focus_id, confusion_id)
    focus_to_conf = focus_to_confusion_from_conf(conf, focus_id, confusion_id)
    return {
        "method": method,
        "score_space": "",
        "variant": variant.name,
        "kind": variant.kind,
        "keep_ratio": variant.keep_ratio,
        "feature_zero_ratio": variant.feature_zero_ratio,
        "block_size": variant.block_size,
        "repeats": repeat_count,
        "observed_keep_frac": mean_keep,
        "mIoU": s["mIoU"],
        "delta_mIoU": s["mIoU"] - base["mIoU"],
        "mAcc": s["mAcc"],
        "allAcc": s["allAcc"],
        "focus_iou": focus_iou,
        "delta_focus_iou": focus_iou - base_focus_iou,
        "confusion_iou": confusion_iou,
        "focus_to_confusion": focus_to_conf,
        "delta_focus_to_confusion": focus_to_conf - base_focus_to_conf,
    }


def write_results(args: argparse.Namespace, results: dict, variants: list[Variant], class_names: list[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    focus_id = name_to_id.get(args.focus_class)
    confusion_id = name_to_id.get(args.confusion_class)
    if focus_id is None:
        print(f"[warn] focus class not found: {args.focus_class}", flush=True)
    if confusion_id is None:
        print(f"[warn] confusion class not found: {args.confusion_class}", flush=True)
    rows = []
    for score_space in results.get("score_spaces", ["retained"]):
        base = summarize_confusion(results["conf"][f"{score_space}:clean_voxel"])
        for variant in variants:
            row = summary_row(
                    args.method_name,
                    variant,
                    results["keep_counts"][variant.name],
                    results["keep_sums"][variant.name] / max(results["keep_counts"][variant.name], 1),
                    results["conf"][f"{score_space}:{variant.name}"],
                    base,
                    focus_id,
                    confusion_id,
            )
            row["score_space"] = score_space
            rows.append(row)
    csv_path = args.output_dir / "masking_battery_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    prefix = args.summary_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PTv3 v1.5.1 Compatibility Masking Eval",
        "",
        "Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.",
        "",
        "## Setup",
        "",
        f"- Method: `{args.method_name}`",
        f"- Official root: `{args.official_root}`",
        f"- Config: `{args.config}`",
        f"- Weight: `{args.weight}`",
        f"- Data root: `{args.data_root}`",
        f"- Segment key: `{args.segment_key}`",
        f"- Focus class: `{args.focus_class}`",
        f"- Confusion class: `{args.confusion_class}`",
        f"- Full-scene scoring: `{args.full_scene_scoring}`",
        "",
        "## Results",
        "",
        "| score | variant | keep | mIoU | ΔmIoU | allAcc | focus IoU | Δfocus | confusion IoU | focus->confusion |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['score_space']}` | `{row['variant']}` | {float(row['observed_keep_frac']):.4f} | "
            f"{float(row['mIoU']):.4f} | {float(row['delta_mIoU']):+.4f} | "
            f"{float(row['allAcc']):.4f} | {float(row['focus_iou']):.4f} | "
            f"{float(row['delta_focus_iou']):+.4f} | {float(row['confusion_iou']):.4f} | "
            f"{float(row['focus_to_confusion']):.4f} |"
        )
    lines += ["", "## Files", "", f"- Summary CSV: `{csv_path.resolve()}`"]
    md_path = prefix.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n")
    (args.output_dir / "masking_battery.md").write_text("\n".join(lines) + "\n")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "method_name": args.method_name,
                "official_root": str(args.official_root),
                "config": str(args.config),
                "weight": str(args.weight),
                "segment_key": args.segment_key,
                "class_names": class_names,
                "focus_class": args.focus_class,
                "confusion_class": args.confusion_class,
                "full_scene_scoring": args.full_scene_scoring,
                "variants": [v.__dict__ for v in variants],
                "outputs": {"summary_csv": str(csv_path), "summary_md": str(md_path)},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] wrote {md_path}", flush=True)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.repo_root = args.repo_root.resolve()
    args.official_root = (args.repo_root / args.official_root).resolve() if not args.official_root.is_absolute() else args.official_root
    args.data_root = (args.repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.weight = (args.repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.output_dir = (args.repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (args.repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    if args.dry_run:
        print(f"[dry-run] official_root={args.official_root}")
        print(f"[dry-run] data_root={args.data_root}")
        print(f"[dry-run] variants={[v.name for v in build_variants(args)]}")
        return
    config_cls, compose_cls, build_model_fn = setup_official_imports(args.official_root)
    cfg = load_config(config_cls, args.official_root / args.config)
    class_names = class_names_from_cfg(cfg, args)
    transform = compose_cls(cfg.data.val.transform)
    model = build_official_model(build_model_fn, cfg, args.weight)
    scenes = scene_paths(args.data_root, args.split)
    results = eval_masking(args, model, transform, scenes, build_variants(args))
    write_results(args, results, build_variants(args), class_names)


if __name__ == "__main__":
    main()
