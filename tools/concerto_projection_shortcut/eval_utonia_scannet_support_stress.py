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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    summarize_confusion,
    update_confusion,
)


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(x)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    keep_ratio: float = 1.0
    fixed_count: int = 0
    block_m: float = 1.28


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

    def get_data_name(self, index: int) -> str:
        return self.scene_dirs[index].name

    def raw_scene(self, index: int) -> dict:
        scene_dir = self.scene_dirs[index]
        return {
            "coord": np.load(scene_dir / "coord.npy"),
            "color": np.load(scene_dir / "color.npy"),
            "normal": np.load(scene_dir / "normal.npy"),
            "segment": np.load(scene_dir / "segment20.npy"),
            "instance": np.load(scene_dir / "instance.npy"),
        }

    def transform_scene(self, scene: dict) -> dict:
        point = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in scene.items()}
        raw_segment = torch.from_numpy(point["segment"].copy()).long()
        raw_coord = torch.from_numpy(point["coord"].copy()).float()
        out = self.transform(point)
        # Utonia's default Collect does not keep labels. The transform mutates
        # point in-place before Collect, so point["segment"] is the voxelized
        # segment array at the model input resolution.
        out["segment"] = torch.from_numpy(point["segment"]).long()
        out["raw_segment"] = raw_segment
        out["raw_coord"] = raw_coord
        return out

    def __getitem__(self, index: int) -> dict:
        out = self.transform_scene(self.raw_scene(index))
        out["scene_name"] = self.get_data_name(index)
        return out


def collate_one(batch: list[dict]) -> dict:
    assert len(batch) == 1
    return batch[0]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Utonia ScanNet support-stress battery")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--utonia-weight", type=Path, required=True)
    parser.add_argument("--seg-head-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=2)
    parser.add_argument("--max-val-scenes", type=int, default=-1)
    parser.add_argument("--random-keep-ratios", default="0.2")
    parser.add_argument("--structured-keep-ratios", default="0.2")
    parser.add_argument("--masked-model-keep-ratios", default="0.2")
    parser.add_argument("--fixed-point-counts", default="4000")
    parser.add_argument("--feature-zero", action="store_true")
    parser.add_argument("--structured-block-m", type=float, default=1.28)
    parser.add_argument("--full-scene-chunk-size", type=int, default=2048)
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--disable-flash", action="store_true")
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


def parse_weak_ids(text: str) -> list[int]:
    name_to_id = {name: idx for idx, name in enumerate(SCANNET20_CLASS_NAMES)}
    out = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in name_to_id:
            raise ValueError(f"unknown weak class: {name}")
        out.append(name_to_id[name])
    return out


def build_variants(args: argparse.Namespace) -> list[Variant]:
    out = [Variant("clean", "clean")]
    for keep in parse_float_list(args.random_keep_ratios):
        out.append(Variant(f"random_keep{str(keep).replace('.', 'p')}", "random", keep_ratio=keep))
    for keep in parse_float_list(args.structured_keep_ratios):
        out.append(
            Variant(
                f"structured_b{str(args.structured_block_m).replace('.', 'p')}m_keep{str(keep).replace('.', 'p')}",
                "structured",
                keep_ratio=keep,
                block_m=args.structured_block_m,
            )
        )
    for keep in parse_float_list(args.masked_model_keep_ratios):
        out.append(Variant(f"masked_model_keep{str(keep).replace('.', 'p')}", "masked_model", keep_ratio=keep))
    for count in parse_int_list(args.fixed_point_counts):
        out.append(Variant(f"fixed_points_{count}", "fixed", fixed_count=count))
    if args.feature_zero:
        # Utonia's default Collect builds feat=(coord,color,normal), while the
        # model also receives coord/grid_coord as separate structural keys.
        # Keep the legacy all-feat zero row, but also separate the official
        # --wo_color/--wo_normal style raw ablations from feature-channel
        # ablations so a low all-feat-zero damage cannot be misread as a
        # transform bug.
        out.extend(
            [
                Variant("feature_zero", "feature_zero"),
                Variant("feat_zero_color_normal", "feat_zero_color_normal"),
                Variant("feat_zero_coord", "feat_zero_coord"),
                Variant("raw_wo_color", "raw_wo_color"),
                Variant("raw_wo_normal", "raw_wo_normal"),
                Variant("raw_wo_color_normal", "raw_wo_color_normal"),
            ]
        )
    return out


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
    skip = {"raw_segment", "raw_coord", "scene_name"}
    for key, value in input_dict.items():
        if key in skip:
            continue
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.cuda(non_blocking=True)
    return input_dict


def inference_input(batch: dict) -> dict:
    return {
        key: value
        for key, value in batch.items()
        if key not in {"segment", "raw_segment", "raw_coord", "scene_name"}
    }


@torch.no_grad()
def forward_raw_logits(model, seg_head, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    batch = move_to_cuda(batch)
    out = model(inference_input(batch))
    while "pooling_parent" in out.keys():
        parent = out.pop("pooling_parent")
        inverse = out.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
        out = parent
    logits = seg_head(out.feat.float()).float().cpu()
    inverse = batch["inverse"].long().cpu()
    raw_logits = logits[inverse]
    raw_labels = batch["raw_segment"].long().cpu()
    if raw_logits.shape[0] != raw_labels.shape[0]:
        raise RuntimeError(f"raw logits/labels mismatch: {raw_logits.shape} vs {raw_labels.shape}")
    return raw_logits, raw_labels


def apply_variant(scene: dict, variant: Variant, seed: int) -> tuple[dict, float]:
    if variant.kind in {"clean", "feature_zero", "feat_zero_color_normal", "feat_zero_coord"}:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in scene.items()}, 1.0

    rng = np.random.default_rng(seed)
    n = int(scene["segment"].shape[0])
    keep = np.zeros(n, dtype=bool)

    if variant.kind == "random":
        keep = rng.random(n) < variant.keep_ratio
    elif variant.kind == "fixed":
        count = max(1, min(int(variant.fixed_count), n))
        keep[rng.permutation(n)[:count]] = True
    elif variant.kind == "structured":
        coord = scene["coord"]
        key = np.floor((coord - coord.min(axis=0, keepdims=True)) / variant.block_m).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        n_region = int(inv.max()) + 1
        region_keep = rng.random(n_region) < variant.keep_ratio
        if not region_keep.any():
            region_keep[rng.integers(n_region)] = True
        keep = region_keep[inv]
    elif variant.kind == "masked_model":
        instance = scene["instance"].reshape(-1)
        stuff = instance < 0
        if np.any(stuff):
            keep[stuff] = rng.random(int(stuff.sum())) < variant.keep_ratio
        inst_ids = np.unique(instance[instance >= 0])
        if inst_ids.size > 0:
            inst_keep = rng.random(inst_ids.shape[0]) < variant.keep_ratio
            if not inst_keep.any():
                inst_keep[rng.integers(inst_ids.shape[0])] = True
            kept = set(inst_ids[inst_keep].tolist())
            for inst_id in kept:
                keep[instance == inst_id] = True
    else:
        raise ValueError(f"unknown variant kind: {variant.kind}")

    if not keep.any():
        keep[rng.integers(n)] = True
    out = {}
    for key, value in scene.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n,):
            out[key] = value[keep]
        else:
            out[key] = value.copy() if isinstance(value, np.ndarray) else value
    return out, float(keep.mean())


def apply_feature_variant(batch: dict, kind: str) -> dict:
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    if "feat" not in out:
        return out
    if kind == "feature_zero":
        out["feat"] = torch.zeros_like(out["feat"])
    elif kind == "feat_zero_color_normal":
        # feat layout follows Utonia default Collect: coord(0:3), color(3:6),
        # normal(6:9).
        out["feat"][:, 3:] = 0
    elif kind == "feat_zero_coord":
        out["feat"][:, :3] = 0
    return out


def apply_raw_feature_variant(scene: dict, kind: str) -> dict:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in scene.items()}
    if kind in {"raw_wo_color", "raw_wo_color_normal"} and "color" in out:
        out["color"] = np.zeros_like(out["coord"])
    if kind in {"raw_wo_normal", "raw_wo_color_normal"} and "normal" in out:
        out["normal"] = np.zeros_like(out["coord"])
    return out


def nearest_logits_to_full(
    full_coord: torch.Tensor,
    support_coord: torch.Tensor,
    support_logits: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if full_coord.shape[0] == support_coord.shape[0]:
        return support_logits
    try:
        import pointops  # noqa: PLC0415

        support_offset = torch.tensor([support_coord.shape[0]], dtype=torch.int32, device=support_coord.device)
        full_offset = torch.tensor([full_coord.shape[0]], dtype=torch.int32, device=full_coord.device)
        index, _ = pointops.knn_query(1, support_coord.float(), support_offset, full_coord.float(), full_offset)
        return support_logits[index.flatten().long().cpu()]
    except Exception:
        out_idx = torch.empty(full_coord.shape[0], dtype=torch.long)
        support = support_coord.float().cuda()
        query = full_coord.float().cuda()
        for begin in range(0, full_coord.shape[0], chunk_size):
            end = min(begin + chunk_size, full_coord.shape[0])
            dist = torch.cdist(query[begin:end], support)
            out_idx[begin:end] = dist.argmin(dim=1).cpu()
        return support_logits[out_idx]


def update_from_logits(conf: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
    valid = (labels >= 0) & (labels < len(SCANNET20_CLASS_NAMES))
    update_confusion(conf, logits[valid].argmax(dim=1), labels[valid], len(SCANNET20_CLASS_NAMES), ignore_index=-1)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader = build_loader(args.data_root, args.val_split, args.batch_size, args.num_worker)
    model, seg_head = build_model(args.utonia_weight, args.seg_head_weight, args.disable_flash)
    variants = build_variants(args)
    weak_ids = parse_weak_ids(args.weak_classes)

    confs = {variant.name: torch.zeros((len(SCANNET20_CLASS_NAMES), len(SCANNET20_CLASS_NAMES)), dtype=torch.long) for variant in variants}
    keep_fracs = {variant.name: [] for variant in variants}
    seen = 0

    for scene_idx, clean_batch in enumerate(loader):
        if args.max_val_scenes >= 0 and scene_idx >= args.max_val_scenes:
            break
        scene_name = clean_batch.pop("scene_name")
        raw_scene = loader.dataset.raw_scene(scene_idx)
        full_coord = torch.from_numpy(raw_scene["coord"]).float()
        full_labels = torch.from_numpy(raw_scene["segment"]).long()
        for v_idx, variant in enumerate(variants):
            seed = args.seed + scene_idx * 1009 + v_idx
            if variant.kind == "clean":
                support_batch = loader.dataset.transform_scene(raw_scene)
            elif variant.kind in {"raw_wo_color", "raw_wo_normal", "raw_wo_color_normal"}:
                support_batch = loader.dataset.transform_scene(apply_raw_feature_variant(raw_scene, variant.kind))
                keep_fracs[variant.name].append(1.0)
            else:
                masked_scene, keep_frac = apply_variant(raw_scene, variant, seed)
                keep_fracs[variant.name].append(keep_frac)
                support_batch = loader.dataset.transform_scene(masked_scene)
            if variant.kind in {"feature_zero", "feat_zero_color_normal", "feat_zero_coord"}:
                support_batch = apply_feature_variant(support_batch, variant.kind)
                keep_fracs[variant.name].append(1.0)
            if variant.kind == "clean":
                keep_fracs[variant.name].append(1.0)
            raw_logits_support, _ = forward_raw_logits(model, seg_head, support_batch)
            support_coord = support_batch["raw_coord"].float()
            full_logits = nearest_logits_to_full(
                full_coord,
                support_coord,
                raw_logits_support,
                args.full_scene_chunk_size,
            )
            update_from_logits(confs[variant.name], full_logits, full_labels)
        seen += 1
        if seen % 25 == 0:
            print(f"[eval] scenes={seen}/{len(loader.dataset)} last={scene_name}", flush=True)

    rows = []
    base = summarize_confusion(confs["clean"].numpy(), SCANNET20_CLASS_NAMES)
    for variant in variants:
        summary = summarize_confusion(confs[variant.name].numpy(), SCANNET20_CLASS_NAMES)
        per_class = summary["iou"]
        row = {
            "method": "utonia_scannet_linear_head",
            "condition": variant.name,
            "miou": summary["mIoU"],
            "miou_delta": summary["mIoU"] - base["mIoU"],
            "weak_mean": float(np.mean([per_class[idx] for idx in weak_ids])),
            "mean_keep_frac": float(np.mean(keep_fracs[variant.name])) if keep_fracs[variant.name] else 1.0,
        }
        for name in ["wall", "picture", "counter", "desk", "sink", "cabinet", "shower curtain", "door"]:
            idx = SCANNET20_CLASS_NAMES.index(name)
            row[f"iou_{name.replace(' ', '_')}"] = float(per_class[idx])
        rows.append(row)

    csv_path = args.output_dir / "utonia_scannet_support_stress.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / "utonia_scannet_support_stress.md"
    lines = [
        "# Utonia ScanNet Support Stress",
        "",
        "## Setup",
        f"- utonia weight: `{args.utonia_weight}`",
        f"- seg head weight: `{args.seg_head_weight}`",
        f"- data root: `{args.data_root}`",
        f"- val scenes: `{seen}`",
        "- scoring: full-scene nearest-neighbor propagation from retained support logits",
        "",
        "## Results",
        "",
        "| condition | keep frac | mIoU | delta | weak mean | picture | wall | counter | cabinet | sink | desk | door |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['condition']}` | `{row['mean_keep_frac']:.4f}` | `{row['miou']:.4f}` | "
            f"`{row['miou_delta']:.4f}` | `{row['weak_mean']:.4f}` | `{row['iou_picture']:.4f}` | "
            f"`{row['iou_wall']:.4f}` | `{row['iou_counter']:.4f}` | `{row['iou_cabinet']:.4f}` | "
            f"`{row['iou_sink']:.4f}` | `{row['iou_desk']:.4f}` | `{row['iou_door']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This battery tests whether Utonia's cleaner fixed readout also changes the support-stress profile.",
            "- Random/fixed/structured/object-style rows should be read as full-scene missing-support stress, not retained-subset scoring.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")

    meta = {
        "args": vars(args),
        "seen_scenes": seen,
        "variants": [v.__dict__ for v in variants],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[write] {md_path}")
    print(f"[write] {csv_path}")


if __name__ == "__main__":
    main()
