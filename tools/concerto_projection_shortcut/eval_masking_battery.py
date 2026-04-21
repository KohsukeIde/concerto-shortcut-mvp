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
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    build_model,
    load_config,
    move_to_cuda,
    summarize_confusion,
    update_confusion,
    weak_mean,
)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    keep_ratio: float = 1.0
    feature_zero_ratio: float = 0.0
    block_size: int = 64
    fixed_count: int = 0


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shortcut-sensitive masking battery for ScanNet segmentation checkpoints. "
            "The first use is a cheap Concerto decoder smoke: clean vs random point "
            "drop at high mask ratios, evaluated in the same voxel-level space."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--method-name", default="concerto_decoder_origin")
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--random-keep-ratios", default="0.2")
    parser.add_argument("--fixed-point-counts", default="")
    parser.add_argument("--classwise-keep-ratios", default="")
    parser.add_argument("--structured-keep-ratios", default="")
    parser.add_argument("--feature-zero-ratios", default="")
    parser.add_argument(
        "--color-feat-space",
        choices=("current_0_1", "legacy_minus1_1"),
        default="current_0_1",
        help=(
            "Feature color convention after the repo transform. Current repo "
            "NormalizeColor maps RGB to [0, 1]. Pointcept v1.5.1 released "
            "PTv3 checkpoints used RGB / 127.5 - 1; use legacy_minus1_1 for "
            "those downloaded checkpoints."
        ),
    )
    parser.add_argument("--structured-block-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--full-scene-scoring",
        action="store_true",
        help=(
            "Also score every original voxel by nearest-neighbor propagation "
            "from the retained masked-input logits. The default retained-subset "
            "rows are still written."
        ),
    )
    parser.add_argument(
        "--full-scene-chunk-size",
        type=int,
        default=2048,
        help="Fallback chunk size for torch.cdist nearest-neighbor propagation.",
    )
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--save-example-scenes", type=int, default=0)
    parser.add_argument("--example-output-dir", type=Path, default=Path("data/runs/masking_examples"))
    parser.add_argument("--example-max-export-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_masking_battery"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_names(text: str) -> list[int]:
    ids = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {name}")
        ids.append(NAME_TO_ID[name])
    return ids


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


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants = [Variant("clean_voxel", "clean")]
    for keep in parse_float_list(args.random_keep_ratios):
        variants.append(Variant(f"random_keep{str(keep).replace('.', 'p')}", "random_drop", keep_ratio=keep))
    for count in parse_int_list(args.fixed_point_counts):
        variants.append(Variant(f"fixed_points_{count}", "fixed_count_drop", fixed_count=count))
    for keep in parse_float_list(args.classwise_keep_ratios):
        variants.append(
            Variant(
                f"classwise_keep{str(keep).replace('.', 'p')}",
                "classwise_random_drop",
                keep_ratio=keep,
            )
        )
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


def input_point_count(batch: dict) -> int:
    return int(batch["segment"].shape[0])


def clone_batch(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
    return out


def adapt_color_feat_space(batch: dict, color_feat_space: str) -> dict:
    """Adjust concatenated color/normal features to the checkpoint convention.

    The local repo's NormalizeColor produces color in [0, 1]. Older released
    Pointcept v1.5.1 PTv3 checkpoints were trained with color in [-1, 1].
    The first three feature channels are RGB because ScanNet configs collect
    feat_keys=("color", "normal").
    """
    if color_feat_space == "current_0_1":
        return batch
    if color_feat_space != "legacy_minus1_1":
        raise ValueError(f"unknown color feature space: {color_feat_space}")
    out = clone_batch(batch)
    if "feat" in out:
        out["feat"] = out["feat"].clone()
        out["feat"][:, :3] = out["feat"][:, :3] * 2.0 - 1.0
    return out


def filter_batch(batch: dict, mask: torch.Tensor) -> dict:
    n = input_point_count(batch)
    out = {}
    for key, value in batch.items():
        if key in {"inverse", "origin_segment"}:
            continue
        if isinstance(value, torch.Tensor) and value.shape[:1] == (n,):
            out[key] = value[mask]
        elif key == "offset" and isinstance(value, torch.Tensor):
            out[key] = torch.tensor([int(mask.sum().item())], dtype=value.dtype, device=value.device)
        else:
            out[key] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    return out


def random_drop_mask(n: int, keep_ratio: float, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    keep = torch.rand(n, device=device, generator=generator) < keep_ratio
    if not bool(keep.any()):
        keep[torch.randint(n, (1,), device=device, generator=generator)] = True
    return keep


def fixed_count_drop_mask(n: int, fixed_count: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    count = max(1, min(int(fixed_count), int(n)))
    perm = torch.randperm(n, device=device, generator=generator)
    keep = torch.zeros(n, dtype=torch.bool, device=device)
    keep[perm[:count]] = True
    return keep


def classwise_drop_mask(labels: torch.Tensor, keep_ratio: float, generator: torch.Generator) -> torch.Tensor:
    keep = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    for cls in labels.unique(sorted=True):
        cls_value = int(cls.item())
        cls_mask = labels == cls
        if cls_value < 0:
            cls_keep = torch.rand(int(cls_mask.sum().item()), device=labels.device, generator=generator) < keep_ratio
            keep[cls_mask] = cls_keep
            continue
        idx = torch.where(cls_mask)[0]
        cls_keep = torch.rand(idx.shape[0], device=labels.device, generator=generator) < keep_ratio
        if not bool(cls_keep.any()):
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
    if variant.kind == "fixed_count_drop":
        mask = fixed_count_drop_mask(n, variant.fixed_count, device, generator)
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
    """Drop supervision-only tensors so eval forward does not compute losses."""
    skip = {"segment", "origin_segment", "inverse"}
    out = {}
    for key, value in batch.items():
        if key in skip:
            continue
        out[key] = value
    return out


@torch.no_grad()
def forward_logits_labels(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["segment"].long()
    model_input = inference_batch(batch)
    try:
        out = model(model_input, return_point=True)
    except TypeError as exc:
        # PPT-v1m1 does not accept return_point; masking battery only needs logits.
        if "return_point" not in str(exc):
            raise
        out = model(model_input)
    logits = out["seg_logits"].float()
    if logits.shape[0] != labels.shape[0]:
        raise RuntimeError(f"shape mismatch logits={logits.shape} labels={labels.shape}")
    return logits, labels


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


def full_scene_logits_from_support(logits: torch.Tensor, query_batch: dict, support_batch: dict, chunk_size: int) -> torch.Tensor:
    if logits.shape[0] == input_point_count(query_batch) and input_point_count(query_batch) == input_point_count(support_batch):
        return logits
    query_coord = query_batch["coord"]
    support_coord = support_batch["coord"]
    query_offset = query_batch.get("offset")
    support_offset = support_batch.get("offset")
    if query_offset is None:
        query_offset = torch.tensor([int(query_coord.shape[0])], dtype=torch.int32, device=logits.device)
    if support_offset is None:
        support_offset = torch.tensor([int(support_coord.shape[0])], dtype=torch.int32, device=logits.device)
    nearest = nearest_retained_index(
        query_coord,
        support_coord,
        query_offset,
        support_offset,
        chunk_size,
    )
    return logits[nearest]


def infer_rgb_from_batch(batch: dict) -> np.ndarray:
    if "feat" in batch:
        feat = batch["feat"].detach().cpu().numpy()
        if feat.shape[1] >= 3:
            rgb = feat[:, :3]
            if float(rgb.min()) < 0.0:
                rgb = (rgb + 1.0) / 2.0
            return np.clip(rgb, 0.0, 1.0)
    coord = batch["coord"].detach().cpu().numpy()
    return np.ones_like(coord, dtype=np.float32)


def write_ascii_ply(path: Path, coord: np.ndarray, color: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coord = np.asarray(coord, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)
    if color.shape[0] != coord.shape[0]:
        raise ValueError("coord/color size mismatch")
    rgb = np.clip(np.round(color * 255.0), 0, 255).astype(np.uint8)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {coord.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz, c in zip(coord, rgb):
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def maybe_save_example_scene(
    args: argparse.Namespace,
    scene_name: str,
    variant: Variant,
    base_batch: dict,
    masked_batch: dict,
    keep_frac: float,
) -> None:
    if args.save_example_scenes <= 0:
        return
    out_dir = args.example_output_dir / args.dataset_tag / scene_name / variant.name
    meta = {
        "dataset": args.dataset_tag,
        "scene": scene_name,
        "variant": variant.name,
        "kind": variant.kind,
        "keep_ratio": variant.keep_ratio,
        "feature_zero_ratio": variant.feature_zero_ratio,
        "block_size": variant.block_size,
        "fixed_count": variant.fixed_count,
        "base_points": int(input_point_count(base_batch)),
        "masked_points": int(input_point_count(masked_batch)),
        "observed_keep_frac": float(keep_frac),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    base_coord = base_batch["coord"].detach().cpu().numpy()
    masked_coord = masked_batch["coord"].detach().cpu().numpy()
    base_color = infer_rgb_from_batch(base_batch)
    masked_color = infer_rgb_from_batch(masked_batch)
    rng = np.random.default_rng(args.seed)
    for stem, coord, color in [
        ("input_clean", base_coord, base_color),
        ("input_masked", masked_coord, masked_color),
    ]:
        np.savez_compressed(
            out_dir / f"{stem}.npz",
            coord=coord.astype(np.float32),
            color=color.astype(np.float32),
        )
        if coord.shape[0] > args.example_max_export_points:
            idx = rng.choice(coord.shape[0], size=args.example_max_export_points, replace=False)
            idx = np.sort(idx)
            coord_view = coord[idx]
            color_view = color[idx]
        else:
            coord_view = coord
            color_view = color
        write_ascii_ply(out_dir / f"{stem}_preview.ply", coord_view, color_view)


def picture_to_wall_from_conf(conf: np.ndarray) -> float:
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    denom = conf[pic].sum()
    return float(conf[pic, wall] / denom) if denom else float("nan")


def summary_row(method: str, variant: Variant, repeat_count: int, mean_keep: float, conf: np.ndarray, base_summary: dict, weak_classes: list[int]) -> dict[str, float | str | int]:
    s = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    floor = NAME_TO_ID["floor"]
    return {
        "method": method,
        "variant": variant.name,
        "kind": variant.kind,
        "keep_ratio": variant.keep_ratio,
        "fixed_count": variant.fixed_count,
        "feature_zero_ratio": variant.feature_zero_ratio,
        "block_size": variant.block_size,
        "repeats": repeat_count,
        "observed_keep_frac": mean_keep,
        "mIoU": s["mIoU"],
        "delta_mIoU": s["mIoU"] - base_summary["mIoU"],
        "mAcc": s["mAcc"],
        "allAcc": s["allAcc"],
        "weak_mean_iou": weak_mean(s, weak_classes),
        "delta_weak_mean_iou": weak_mean(s, weak_classes) - weak_mean(base_summary, weak_classes),
        "picture_iou": float(s["iou"][pic]),
        "delta_picture_iou": float(s["iou"][pic] - base_summary["iou"][pic]),
        "wall_iou": float(s["iou"][wall]),
        "floor_iou": float(s["iou"][floor]),
        "picture_to_wall": picture_to_wall_from_conf(conf),
        "delta_picture_to_wall": picture_to_wall_from_conf(conf) - picture_to_wall_from_conf(base_summary["conf"]),
    }


def eval_masking(args: argparse.Namespace, model, cfg, variants: list[Variant], weak_classes: list[int], num_classes: int):
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    score_spaces = ["retained"]
    if args.full_scene_scoring:
        score_spaces.append("full_nn")
    confs = {
        (space, v.name): torch.zeros((num_classes, num_classes), dtype=torch.long)
        for space in score_spaces
        for v in variants
    }
    keep_sums = {v.name: 0.0 for v in variants}
    keep_counts = {v.name: 0 for v in variants}

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            batch = adapt_color_feat_space(batch, args.color_feat_space)
            scene_name = loader.dataset.get_data_name(batch_idx)
            for variant in variants:
                repeat_count = (
                    args.repeats
                    if variant.kind in {"random_drop", "fixed_count_drop", "classwise_random_drop", "structured_drop", "feature_zero"}
                    else 1
                )
                for repeat_idx in range(repeat_count):
                    seed = args.seed + batch_idx * 1009 + repeat_idx * 9176 + abs(hash(variant.name)) % 1000
                    masked_batch, keep_frac, keep_mask = make_variant_batch_with_mask(batch, variant, seed)
                    if input_point_count(masked_batch) <= 0:
                        continue
                    if batch_idx < args.save_example_scenes and repeat_idx == 0:
                        maybe_save_example_scene(args, scene_name, variant, batch, masked_batch, keep_frac)
                    logits, labels = forward_logits_labels(model, masked_batch)
                    pred = logits.argmax(dim=1)
                    update_confusion(confs[("retained", variant.name)], pred.cpu(), labels.cpu(), num_classes, -1)
                    if args.full_scene_scoring:
                        full_logits = full_scene_logits_from_support(logits, batch, masked_batch, args.full_scene_chunk_size)
                        update_confusion(
                            confs[("full_nn", variant.name)],
                            full_logits.argmax(dim=1).cpu(),
                            batch["segment"].long().cpu(),
                            num_classes,
                            -1,
                        )
                    keep_sums[variant.name] += keep_frac
                    keep_counts[variant.name] += 1
            if (batch_idx + 1) % 25 == 0:
                clean = summarize_confusion(confs[("retained", "clean_voxel")].numpy(), SCANNET20_CLASS_NAMES)
                print(f"[val] batch={batch_idx+1} clean_voxel_mIoU={clean['mIoU']:.4f}", flush=True)
    return {
        "conf": {f"{space}:{name}": v.numpy() for (space, name), v in confs.items()},
        "score_spaces": score_spaces,
        "keep_sums": keep_sums,
        "keep_counts": keep_counts,
    }


def write_results(args: argparse.Namespace, results: dict, variants: list[Variant], weak_classes: list[int]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for score_space in results.get("score_spaces", ["retained"]):
        base_key = f"{score_space}:clean_voxel"
        base = summarize_confusion(results["conf"][base_key], SCANNET20_CLASS_NAMES)
        base["conf"] = results["conf"][base_key]
        for v in variants:
            row = summary_row(
                    args.method_name,
                    v,
                    results["keep_counts"][v.name],
                    results["keep_sums"][v.name] / max(results["keep_counts"][v.name], 1),
                    results["conf"][f"{score_space}:{v.name}"],
                    base,
                    weak_classes,
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

    sorted_rows = sorted(rows, key=lambda r: (r["score_space"], r["variant"] != "clean_voxel", -float(r["mIoU"])))
    lines = []
    lines.append("# Masking Battery Pilot\n")
    lines.append("Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.\n")
    lines.append("## Setup\n")
    lines.append(f"- Method: `{args.method_name}`")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Weight: `{args.weight}`")
    lines.append(f"- Random keep ratios: `{args.random_keep_ratios}`")
    lines.append(f"- Class-wise keep ratios: `{args.classwise_keep_ratios}`")
    lines.append(f"- Structured keep ratios: `{args.structured_keep_ratios}`")
    lines.append(f"- Feature-zero ratios: `{args.feature_zero_ratios}`")
    lines.append(f"- Color feature space: `{args.color_feat_space}`")
    lines.append(f"- Repeats: `{args.repeats}`")
    lines.append(f"- Full-scene scoring: `{args.full_scene_scoring}`")
    lines.append("")
    lines.append("## Results\n")
    lines.append("| score | variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted_rows:
        lines.append(
            f"| `{r['score_space']}` | `{r['variant']}` | {float(r['observed_keep_frac']):.4f} | "
            f"{float(r['mIoU']):.4f} | {float(r['delta_mIoU']):+.4f} | {float(r['allAcc']):.4f} | "
            f"{float(r['weak_mean_iou']):.4f} | {float(r['picture_iou']):.4f} | {float(r['delta_picture_iou']):+.4f} | "
            f"{float(r['wall_iou']):.4f} | {float(r['floor_iou']):.4f} | {float(r['picture_to_wall']):.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation Gate\n")
    lines.append("- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.")
    lines.append("- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.")
    lines.append("- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.")
    lines.append("")
    lines.append("## Files\n")
    lines.append(f"- Summary CSV: `{csv_path.resolve()}`")
    md = prefix.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    (args.output_dir / "masking_battery.md").write_text("\n".join(lines) + "\n")
    metadata = {
        "method_name": args.method_name,
        "config": str(args.config),
        "weight": str(args.weight),
        "color_feat_space": args.color_feat_space,
        "full_scene_scoring": args.full_scene_scoring,
        "variants": [v.__dict__ for v in variants],
        "outputs": {"summary_csv": str(csv_path), "summary_md": str(md)},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[done] wrote {md}", flush=True)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.repo_root = args.repo_root.resolve()
    args.data_root = (args.repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.weight = (args.repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.output_dir = (args.repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (args.repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    args.example_output_dir = (
        (args.repo_root / args.example_output_dir).resolve()
        if not args.example_output_dir.is_absolute()
        else args.example_output_dir
    )
    if not args.dataset_tag:
        args.dataset_tag = args.data_root.name
    variants = build_variants(args)
    weak_classes = parse_names(args.weak_classes)
    if args.dry_run:
        print(f"[dry-run] data_root={args.data_root}")
        print(f"[dry-run] weight={args.weight}")
        print(f"[dry-run] variants={[v.name for v in variants]}")
        print(f"[dry-run] weak_classes={[SCANNET20_CLASS_NAMES[c] for c in weak_classes]}")
        return
    cfg = load_config(args.repo_root / args.config)
    num_classes = int(cfg.data.num_classes)
    model = build_model(cfg, args.weight)
    results = eval_masking(args, model, cfg, variants, weak_classes, num_classes)
    write_results(args, results, variants, weak_classes)


if __name__ == "__main__":
    main()
