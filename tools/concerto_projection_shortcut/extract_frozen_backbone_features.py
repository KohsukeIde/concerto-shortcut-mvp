#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(repo_root: Path, config_name: str):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pointcept.utils.config import Config

    return Config.fromfile(str(repo_root / "configs" / "concerto" / f"{config_name}.py"))


def move_batch_to_cuda(batch: dict) -> dict:
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.cuda(non_blocking=True)
    return batch


def build_loader(cfg, split: str, batch_size: int, num_worker: int):
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn

    ds_cfg = cfg.data.train if split == "train" else cfg.data.val
    ds_cfg = ds_cfg.copy()
    ds_cfg.split = split
    if hasattr(ds_cfg, "loop"):
        ds_cfg.loop = 1
    dataset = build_dataset(ds_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        collate_fn=lambda batch: point_collate_fn(batch, mix_prob=0),
    )


def load_backbone(backbone_cfg: dict, weight: Path, keywords: str, replacements: str):
    from pointcept.models.builder import build_model

    backbone = build_model(backbone_cfg).cuda().eval()
    checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if not key.startswith("module."):
            key = "module." + key
        if keywords in key:
            key = key.replace(keywords, replacements)
            key = key[7:]
            if key.startswith("backbone."):
                key = key[9:]
            cleaned[key] = value
    info = backbone.load_state_dict(cleaned, strict=False)
    print(f"[load_backbone] missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return backbone


def unpool_point(point):
    from pointcept.models.utils.structure import Point

    if isinstance(point, Point):
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
    return point


def offset_to_batch(offset: torch.Tensor) -> torch.Tensor:
    counts = torch.diff(torch.cat([offset.new_zeros(1), offset]))
    return torch.repeat_interleave(torch.arange(len(counts), device=offset.device), counts)


def scene_normalize_coord(coord: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    out = coord.clone()
    for scene_id in torch.unique(batch_index):
        mask = batch_index == scene_id
        if not mask.any():
            continue
        c = coord[mask]
        mean = c.mean(dim=0, keepdim=True)
        std = c.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        out[mask] = (c - mean) / std
    return out


def take_scene_sample(mask: torch.Tensor, max_rows: int) -> torch.Tensor:
    idx = mask.nonzero(as_tuple=False).flatten()
    if idx.numel() <= max_rows:
        return idx
    perm = torch.randperm(idx.numel(), device=idx.device)[:max_rows]
    return idx[perm]


def collect_split(backbone, loader, split: str, rows_per_scene: int, max_batches: int) -> Dict[str, torch.Tensor]:
    from pointcept.models.utils.structure import Point

    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    coords: List[torch.Tensor] = []
    batch_scene: List[torch.Tensor] = []

    backbone.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = move_batch_to_cuda(batch)
            point = Point(batch)
            point = backbone(point)
            point = unpool_point(point)
            feat = point.feat.detach().float()
            label = batch["segment"].detach()
            coord = point.coord.detach().float() if hasattr(point, "coord") else batch["coord"].detach().float()
            batch_index = offset_to_batch(point.offset)
            coord_norm = scene_normalize_coord(coord, batch_index)
            chosen = []
            for scene_id in torch.unique(batch_index):
                chosen.append(take_scene_sample(batch_index == scene_id, rows_per_scene))
            chosen = torch.cat(chosen, dim=0) if chosen else torch.empty(0, dtype=torch.long, device=feat.device)
            if chosen.numel() == 0:
                continue
            feats.append(feat[chosen].cpu().half())
            labels.append(label[chosen].cpu().short())
            coords.append(coord_norm[chosen].cpu().float())
            batch_scene.append(batch_index[chosen].cpu().short())
            if (batch_idx + 1) % 10 == 0:
                rows = sum(x.shape[0] for x in feats)
                print(f"[collect] split={split} batch={batch_idx + 1} rows={rows}")

    if not feats:
        raise RuntimeError(f"No rows collected for split={split}")
    payload = {
        "feat": torch.cat(feats, dim=0),
        "label": torch.cat(labels, dim=0),
        "coord_norm": torch.cat(coords, dim=0),
        "scene_batch": torch.cat(batch_scene, dim=0),
    }
    payload["feature_dim"] = int(payload["feat"].shape[1])
    payload["num_rows"] = int(payload["feat"].shape[0])
    print(f"[collect] split={split} done rows={payload['num_rows']} dim={payload['feature_dim']}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Extract frozen Pointcept backbone features for post-hoc nuisance surgery.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="semseg-ptv3-base-v1m1-0a-scannet-lin-proxy")
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--rows-per-scene", type=int, default=16384)
    parser.add_argument("--max-batches-train", type=int, default=-1)
    parser.add_argument("--max-batches-val", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--keywords", default="module.student.backbone")
    parser.add_argument("--replacements", default="module.backbone")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = load_cfg(args.repo_root, args.config)
    backbone = load_backbone(cfg.model.backbone, args.weight, args.keywords, args.replacements)

    args.output_root.mkdir(parents=True, exist_ok=True)
    train_loader = build_loader(cfg, args.train_split, args.batch_size, args.num_worker)
    val_loader = build_loader(cfg, args.val_split, args.batch_size, args.num_worker)

    train_payload = collect_split(backbone, train_loader, args.train_split, args.rows_per_scene, args.max_batches_train)
    val_payload = collect_split(backbone, val_loader, args.val_split, args.rows_per_scene, args.max_batches_val)

    torch.save(train_payload, args.output_root / "train_features.pt")
    torch.save(val_payload, args.output_root / "val_features.pt")
    meta = {
        "config": args.config,
        "weight": str(args.weight),
        "rows_per_scene": args.rows_per_scene,
        "train_rows": train_payload["num_rows"],
        "val_rows": val_payload["num_rows"],
        "feature_dim": train_payload["feature_dim"],
    }
    (args.output_root / "metadata.json").write_text(__import__("json").dumps(meta, indent=2))
    print(f"[done] wrote caches to {args.output_root}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
