#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


GEOM_KEY = "geom_local9"


def repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "configs").is_dir() and (cand / "pointcept").is_dir():
            return cand
    return here.parents[2]


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


def _merge_topk(
    best_dist: torch.Tensor | None,
    best_idx: torch.Tensor | None,
    cand_dist: torch.Tensor,
    cand_idx: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if best_dist is None or best_idx is None:
        if cand_dist.shape[1] > k:
            val, order = torch.topk(cand_dist, k=k, dim=1, largest=False)
            return val, torch.gather(cand_idx, 1, order)
        return cand_dist, cand_idx
    all_dist = torch.cat([best_dist, cand_dist], dim=1)
    all_idx = torch.cat([best_idx, cand_idx], dim=1)
    val, order = torch.topk(all_dist, k=min(k, all_dist.shape[1]), dim=1, largest=False)
    return val, torch.gather(all_idx, 1, order)


@torch.no_grad()
def knn_query_to_scene(
    query: torch.Tensor,
    keys: torch.Tensor,
    k: int,
    query_chunk: int,
    key_chunk: int,
) -> torch.Tensor:
    if query.numel() == 0 or keys.numel() == 0:
        return torch.empty((0, 0), dtype=torch.long, device=query.device)
    k_eff = int(min(max(k, 1), keys.shape[0]))
    all_idx: List[torch.Tensor] = []
    for q_start in range(0, query.shape[0], query_chunk):
        q = query[q_start : q_start + query_chunk]
        best_dist = None
        best_idx = None
        for k_start in range(0, keys.shape[0], key_chunk):
            kb = keys[k_start : k_start + key_chunk]
            dist = torch.cdist(q, kb)
            chunk_k = min(k_eff, kb.shape[0])
            cand_dist, cand_local = torch.topk(dist, k=chunk_k, dim=1, largest=False)
            cand_idx = cand_local + k_start
            best_dist, best_idx = _merge_topk(best_dist, best_idx, cand_dist, cand_idx, k_eff)
        assert best_idx is not None
        all_idx.append(best_idx)
    return torch.cat(all_idx, dim=0)


@torch.no_grad()
def local_geometry_descriptor(
    scene_coord: torch.Tensor,
    local_query_idx: torch.Tensor,
    k: int,
    query_chunk: int,
    key_chunk: int,
    up_axis: str,
) -> torch.Tensor:
    if local_query_idx.numel() == 0:
        return scene_coord.new_zeros((0, 9))
    query = scene_coord[local_query_idx]
    nn_idx = knn_query_to_scene(query, scene_coord, k=k, query_chunk=query_chunk, key_chunk=key_chunk)
    neigh = scene_coord[nn_idx]  # [Q, K, 3]
    neigh_center = neigh - neigh.mean(dim=1, keepdim=True)
    cov = neigh_center.transpose(1, 2).matmul(neigh_center) / max(neigh.shape[1], 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # eigh returns ascending
    l3 = eigvals[:, 0].clamp_min(1e-12)
    l2 = eigvals[:, 1].clamp_min(1e-12)
    l1 = eigvals[:, 2].clamp_min(1e-12)
    denom = (l1 + l2 + l3).clamp_min(1e-12)
    eig_norm = torch.stack([l1 / denom, l2 / denom, l3 / denom], dim=1)
    normal = eigvecs[:, :, 0]
    axis = {"x": 0, "y": 1, "z": 2}[up_axis]
    flip = normal[:, axis] < 0
    normal[flip] = -normal[flip]
    curvature = (l3 / denom).unsqueeze(1)
    linearity = ((l1 - l2) / l1.clamp_min(1e-12)).unsqueeze(1)
    planarity = ((l2 - l3) / l1.clamp_min(1e-12)).unsqueeze(1)
    return torch.cat([eig_norm, normal, curvature, linearity, planarity], dim=1)


def collect_split(
    backbone,
    loader,
    split: str,
    rows_per_scene: int,
    max_batches: int,
    geom_knn: int,
    geom_query_chunk: int,
    geom_key_chunk: int,
    geom_up_axis: str,
) -> Dict[str, torch.Tensor]:
    from pointcept.models.utils.structure import Point

    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    coords_norm: List[torch.Tensor] = []
    coords_raw: List[torch.Tensor] = []
    batch_scene: List[torch.Tensor] = []
    geom_desc: List[torch.Tensor] = []

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
            coord_raw = point.coord.detach().float() if hasattr(point, "coord") else batch["coord"].detach().float()
            batch_index = offset_to_batch(point.offset)
            coord_norm = scene_normalize_coord(coord_raw, batch_index)

            chosen_all: List[torch.Tensor] = []
            geom_all: List[torch.Tensor] = []
            for scene_id in torch.unique(batch_index):
                scene_mask = batch_index == scene_id
                chosen_global = take_scene_sample(scene_mask, rows_per_scene)
                if chosen_global.numel() == 0:
                    continue
                scene_global_idx = scene_mask.nonzero(as_tuple=False).flatten()
                scene_coord_raw = coord_raw[scene_mask]
                local_query_idx = torch.searchsorted(scene_global_idx, chosen_global)
                geom_local = local_geometry_descriptor(
                    scene_coord=scene_coord_raw,
                    local_query_idx=local_query_idx,
                    k=geom_knn,
                    query_chunk=geom_query_chunk,
                    key_chunk=geom_key_chunk,
                    up_axis=geom_up_axis,
                )
                chosen_all.append(chosen_global)
                geom_all.append(geom_local)
            if not chosen_all:
                continue
            chosen = torch.cat(chosen_all, dim=0)
            geom = torch.cat(geom_all, dim=0)
            feats.append(feat[chosen].cpu().half())
            labels.append(label[chosen].cpu().short())
            coords_norm.append(coord_norm[chosen].cpu().float())
            coords_raw.append(coord_raw[chosen].cpu().float())
            batch_scene.append(batch_index[chosen].cpu().short())
            geom_desc.append(geom.cpu().float())
            if (batch_idx + 1) % 10 == 0:
                rows = sum(x.shape[0] for x in feats)
                print(f"[collect] split={split} batch={batch_idx + 1} rows={rows}")

    if not feats:
        raise RuntimeError(f"No rows collected for split={split}")
    payload = {
        "feat": torch.cat(feats, dim=0),
        "label": torch.cat(labels, dim=0),
        "coord_norm": torch.cat(coords_norm, dim=0),
        "coord_raw": torch.cat(coords_raw, dim=0),
        "scene_batch": torch.cat(batch_scene, dim=0),
        GEOM_KEY: torch.cat(geom_desc, dim=0),
    }
    payload["feature_dim"] = int(payload["feat"].shape[1])
    payload["num_rows"] = int(payload["feat"].shape[0])
    payload["geometry_dim"] = int(payload[GEOM_KEY].shape[1])
    print(
        f"[collect] split={split} done rows={payload['num_rows']} dim={payload['feature_dim']} geom_dim={payload['geometry_dim']}"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Extract frozen Pointcept backbone features + local geometry descriptors for Step 1 smoke.")
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
    parser.add_argument("--geometry-knn", type=int, default=32)
    parser.add_argument("--geometry-query-chunk", type=int, default=512)
    parser.add_argument("--geometry-key-chunk", type=int, default=32768)
    parser.add_argument("--geometry-up-axis", choices=["x", "y", "z"], default="z")
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = load_cfg(args.repo_root, args.config)
    backbone = load_backbone(cfg.model.backbone, args.weight, args.keywords, args.replacements)

    args.output_root.mkdir(parents=True, exist_ok=True)
    train_loader = build_loader(cfg, args.train_split, args.batch_size, args.num_worker)
    val_loader = build_loader(cfg, args.val_split, args.batch_size, args.num_worker)

    train_payload = collect_split(
        backbone,
        train_loader,
        args.train_split,
        args.rows_per_scene,
        args.max_batches_train,
        args.geometry_knn,
        args.geometry_query_chunk,
        args.geometry_key_chunk,
        args.geometry_up_axis,
    )
    val_payload = collect_split(
        backbone,
        val_loader,
        args.val_split,
        args.rows_per_scene,
        args.max_batches_val,
        args.geometry_knn,
        args.geometry_query_chunk,
        args.geometry_key_chunk,
        args.geometry_up_axis,
    )

    torch.save(train_payload, args.output_root / "train_features.pt")
    torch.save(val_payload, args.output_root / "val_features.pt")
    meta = {
        "config": args.config,
        "weight": str(args.weight),
        "rows_per_scene": args.rows_per_scene,
        "train_rows": train_payload["num_rows"],
        "val_rows": val_payload["num_rows"],
        "feature_dim": train_payload["feature_dim"],
        "geometry_key": GEOM_KEY,
        "geometry_dim": train_payload["geometry_dim"],
        "geometry_knn": args.geometry_knn,
        "geometry_query_chunk": args.geometry_query_chunk,
        "geometry_key_chunk": args.geometry_key_chunk,
        "geometry_up_axis": args.geometry_up_axis,
    }
    (args.output_root / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote geometry-augmented caches to {args.output_root}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
