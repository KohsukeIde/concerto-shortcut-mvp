#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cfg(repo_root: Path, config_name: str):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pointcept.utils.config import Config

    config_path = repo_root / "configs" / "concerto" / f"{config_name}.py"
    return Config.fromfile(str(config_path))


def clone_batch(batch):
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.clone()
        else:
            output[key] = copy.deepcopy(value)
    return output


def maybe_transform_normals(feat: torch.Tensor, matrix: torch.Tensor | None, zero: bool) -> None:
    if feat.shape[1] < 9:
        return
    if zero:
        feat[:, 6:9] = 0
        return
    if matrix is not None:
        feat[:, 6:9] = feat[:, 6:9] @ matrix.T


def apply_scene_wise(
    batch,
    coord_key: str,
    origin_coord_key: str,
    feat_key: str,
    offset_key: str,
    transform_fn,
    normal_matrix: torch.Tensor | None = None,
    zero_normals: bool = False,
):
    if coord_key not in batch or offset_key not in batch:
        return
    coord = batch[coord_key]
    origin_coord = batch.get(origin_coord_key)
    feat = batch.get(feat_key)
    offsets = batch[offset_key].tolist()
    start = 0
    for end in offsets:
        transformed_coord = transform_fn(coord[start:end])
        coord[start:end] = transformed_coord
        if origin_coord is not None:
            origin_coord[start:end] = transform_fn(origin_coord[start:end])
        if feat is not None and feat.shape[1] >= 3:
            feat[start:end, :3] = transformed_coord
            maybe_transform_normals(
                feat[start:end], matrix=normal_matrix, zero=zero_normals
            )
        start = end


def local_surface_destroy_transform(voxel_size: float):
    def _transform(coord: torch.Tensor) -> torch.Tensor:
        anchor = torch.floor(coord / voxel_size) * voxel_size
        noise = torch.rand_like(coord) * voxel_size
        return anchor + noise

    return _transform


def linear_transform(matrix: torch.Tensor):
    def _transform(coord: torch.Tensor) -> torch.Tensor:
        return coord @ matrix.T

    return _transform


def apply_stress(batch, stress_name: str, voxel_size: float) -> None:
    if stress_name == "clean":
        return
    if stress_name == "local_surface_destroy":
        transform_fn = local_surface_destroy_transform(voxel_size)
        apply_scene_wise(
            batch,
            "global_coord",
            "global_origin_coord",
            "global_feat",
            "global_offset",
            transform_fn,
            zero_normals=True,
        )
        apply_scene_wise(
            batch,
            "local_coord",
            "local_origin_coord",
            "local_feat",
            "local_offset",
            transform_fn,
            zero_normals=True,
        )
        return

    if stress_name == "z_flip":
        matrix = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
        )
    elif stress_name == "xy_swap":
        matrix = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
    elif stress_name == "roll_90_x":
        matrix = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported stress: {stress_name}")

    transform_fn = linear_transform(matrix)
    apply_scene_wise(
        batch,
        "global_coord",
        "global_origin_coord",
        "global_feat",
        "global_offset",
        transform_fn,
        normal_matrix=matrix,
    )
    apply_scene_wise(
        batch,
        "local_coord",
        "local_origin_coord",
        "local_feat",
        "local_offset",
        transform_fn,
        normal_matrix=matrix,
    )


def move_batch_to_cuda(batch):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.cuda(non_blocking=True)
    return batch


def load_weight(model, weight_path: Path) -> None:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    load_info = model.load_state_dict(cleaned, strict=False)
    print(
        "[info] load_weight",
        f"missing={len(load_info.missing_keys)}",
        f"unexpected={len(load_info.unexpected_keys)}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate enc2d loss under stress tests.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="Validation")
    parser.add_argument(
        "--stress",
        nargs="+",
        default=["clean", "local_surface_destroy", "z_flip", "xy_swap", "roll_90_x"],
    )
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_cfg(repo_root, args.config)
    if args.data_root is not None:
        cfg.data.train.data_root = str(args.data_root.resolve())
    cfg.data.train.split = [args.split]
    cfg.data.train.loop = 1

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn
    from pointcept.models.builder import build_model

    dataset = build_dataset(cfg.data.train)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        collate_fn=lambda batch: point_collate_fn(batch, mix_prob=0),
    )
    model = build_model(cfg.model).cuda()
    load_weight(model, args.weight.resolve())
    model.eval()

    metrics = {name: [] for name in args.stress}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            for stress_name in args.stress:
                stressed = clone_batch(batch)
                apply_stress(stressed, stress_name, voxel_size=args.voxel_size)
                stressed = move_batch_to_cuda(stressed)
                output = model(stressed)
                metrics[stress_name].append(float(output["enc2d_loss"].item()))

    print("stress,batches,enc2d_loss_mean")
    for stress_name in args.stress:
        values = metrics[stress_name]
        mean_value = sum(values) / max(len(values), 1)
        print(f"{stress_name},{len(values)},{mean_value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
