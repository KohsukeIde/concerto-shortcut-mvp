#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.ptv3_v151_compat_utils import (  # noqa: E402
    build_official_model,
    class_names_from_cfg,
    load_config,
    load_scene,
    move_to_cuda,
    scene_paths,
    setup_official_imports,
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw-point aligned PTv3 v1.5.1 probabilities for ScanNet20. "
            "This runs in the official Pointcept v1.5.1 import path and writes "
            "per-scene caches that current-repo fusion scripts can consume."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--official-root", type=Path, default=Path("data/tmp/Pointcept-v1.5.1"))
    parser.add_argument("--config", default="configs/scannet/semseg-pt-v3m1-0-base.py")
    parser.add_argument("--weight", type=Path, default=Path("data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--segment-key", default="segment20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--class-names", default="")
    parser.add_argument("--save-logits", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def patch_val_transform_for_raw_inverse(cfg) -> None:
    """Make the official val transform expose raw labels and voxel->raw inverse."""

    transforms = [copy.deepcopy(dict(t)) for t in cfg.data.val.transform]
    out = []
    copied = False
    for t in transforms:
        if t.get("type") == "GridSample":
            if not copied:
                out.append({"type": "Copy", "keys_dict": {"segment": "origin_segment"}})
                copied = True
            t["return_inverse"] = True
            t["return_grid_coord"] = True
        if t.get("type") == "Collect":
            keys = list(t.get("keys", ()))
            for key in ("origin_segment", "inverse"):
                if key not in keys:
                    keys.append(key)
            t["keys"] = tuple(keys)
        out.append(t)
    cfg.data.val.transform = out


def inference_batch(batch: dict) -> dict:
    return {key: value for key, value in batch.items() if key not in {"segment", "origin_segment"}}


@torch.no_grad()
def forward_raw_probs(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = batch["origin_segment"].long()
    out = model(inference_batch(batch))
    logits = out["seg_logits"].float()
    inverse = batch["inverse"].long()
    raw_logits = logits[inverse]
    if raw_logits.shape[0] != labels.shape[0]:
        raise RuntimeError(f"raw logits/labels mismatch: {raw_logits.shape} vs {labels.shape}")
    probs = torch.softmax(raw_logits, dim=1)
    pred = probs.argmax(dim=1)
    return probs.cpu(), pred.cpu(), labels.cpu()


def write_scene_cache(path: Path, probs: torch.Tensor, pred: torch.Tensor, labels: torch.Tensor, save_logits: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "probs": probs.numpy().astype(np.float16),
        "pred": pred.numpy().astype(np.uint8),
        "labels": labels.numpy().astype(np.int16),
    }
    if save_logits:
        arrays["logits"] = torch.log(probs.clamp_min(1e-8)).numpy().astype(np.float16)
    np.savez(path, **arrays)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    official_root = resolve(repo_root, args.official_root)
    data_root = resolve(repo_root, args.data_root)
    weight = resolve(repo_root, args.weight)
    out_dir = resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_cls, compose_cls, build_model_fn, _ = setup_official_imports(official_root)
    cfg = load_config(config_cls, official_root / args.config)
    class_names = class_names_from_cfg(cfg, args.class_names, args.num_classes)
    patch_val_transform_for_raw_inverse(cfg)
    transform = compose_cls(cfg.data.val.transform)
    model = build_official_model(build_model_fn, cfg, weight)
    scenes = scene_paths(data_root, args.split)
    if args.max_val_batches >= 0:
        scenes = scenes[: args.max_val_batches]
    if not scenes:
        raise RuntimeError(f"no scenes found under {data_root / args.split}")

    total = 0
    correct = 0
    for scene_idx, scene_path in enumerate(scenes):
        raw_scene = load_scene(scene_path, args.segment_key)
        batch = move_to_cuda(transform(raw_scene))
        probs, pred, labels = forward_raw_probs(model, batch)
        write_scene_cache(out_dir / f"{scene_path.name}.npz", probs, pred, labels, args.save_logits)
        total += int(labels.numel())
        correct += int((pred == labels).sum().item())
        if (scene_idx + 1) % 25 == 0:
            print(f"[export] scenes={scene_idx + 1}/{len(scenes)} raw_acc={correct / max(total, 1):.4f}", flush=True)

    metadata = {
        "official_root": str(official_root),
        "config": args.config,
        "weight": str(weight),
        "data_root": str(data_root),
        "split": args.split,
        "segment_key": args.segment_key,
        "num_scenes": len(scenes),
        "num_points": total,
        "raw_allAcc": correct / max(total, 1),
        "class_names": class_names,
        "format": "npz per scene with probs float16 [N, C], pred uint8 [N], labels int16 [N]",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {len(scenes)} scene caches to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
