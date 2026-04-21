#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(x)


@dataclass
class SceneBatch:
    coord: np.ndarray
    color: np.ndarray
    normal: np.ndarray
    segment: np.ndarray


def load_scene(scene_dir: str) -> SceneBatch:
    return SceneBatch(
        coord=np.load(os.path.join(scene_dir, "coord.npy")),
        color=np.load(os.path.join(scene_dir, "color.npy")),
        normal=np.load(os.path.join(scene_dir, "normal.npy")),
        segment=np.load(os.path.join(scene_dir, "segment20.npy")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Utonia ScanNet one-scene smoke")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--utonia-weight", required=True)
    parser.add_argument("--seg-head-weight", required=True)
    parser.add_argument("--disable-flash", action="store_true")
    args = parser.parse_args()

    import utonia

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Utonia smoke requires CUDA/spconv on a GPU node")

    scene = load_scene(args.scene_dir)
    print(f"scene_dir={args.scene_dir}")
    print(
        "raw_shapes",
        {
            "coord": tuple(scene.coord.shape),
            "color": tuple(scene.color.shape),
            "normal": tuple(scene.normal.shape),
            "segment": tuple(scene.segment.shape),
        },
    )

    transform = utonia.transform.default(0.5)
    point = transform(
        {
            "coord": scene.coord,
            "color": scene.color,
            "normal": scene.normal,
            "segment": scene.segment,
        }
    )
    print(
        "transformed_shapes",
        {
            "coord": tuple(point["coord"].shape),
            "feat": tuple(point["feat"].shape),
            "inverse": tuple(point["inverse"].shape),
        },
    )

    head_ckpt = utonia.load(args.seg_head_weight, ckpt_only=True)
    seg_head = SegHead(**head_ckpt["config"])
    seg_head.load_state_dict(head_ckpt["state_dict"])

    custom_config = None
    if args.disable_flash:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
    model = utonia.load(args.utonia_weight, custom_config=custom_config)
    model = model.to(device).eval()
    seg_head = seg_head.to(device).eval()

    with torch.inference_mode():
        for key, value in list(point.items()):
            if isinstance(value, torch.Tensor):
                point[key] = value.to(device, non_blocking=True)
        out = model(point)
        unpool_levels = 0
        while "pooling_parent" in out.keys():
            parent = out.pop("pooling_parent")
            inverse = out.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
            out = parent
            unpool_levels += 1
        feat = out.feat
        logits = seg_head(feat)
        pred = logits.argmax(dim=-1)

    pred_cpu = pred.detach().cpu()
    inverse_cpu = point["inverse"].detach().cpu()
    pred_full = pred_cpu[inverse_cpu]
    gt = torch.from_numpy(scene.segment)
    valid = gt >= 0
    acc = (pred_full[valid] == gt[valid]).float().mean().item() if valid.any() else float("nan")

    print(f"device={device}")
    print(f"unpool_levels={unpool_levels}")
    print(f"feat_shape={tuple(feat.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"pred_unique={int(pred_cpu.unique().numel())}")
    print(f"pred_full_shape={tuple(pred_full.shape)}")
    print(f"valid_acc_subset={acc:.6f}")


if __name__ == "__main__":
    main()
