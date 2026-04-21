from __future__ import annotations

import sys
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


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_official_imports(official_root: Path):
    sys.path.insert(0, str(official_root.resolve()))
    from pointcept.datasets.transform import Compose  # noqa: PLC0415
    from pointcept.models.builder import build_model  # noqa: PLC0415
    from pointcept.models.utils.structure import Point  # noqa: PLC0415
    from pointcept.utils.config import Config  # noqa: PLC0415

    return Config, Compose, build_model, Point


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


def scene_paths(data_root: Path, split: str) -> list[Path]:
    split_path = data_root / split
    return sorted(p for p in split_path.iterdir() if p.is_dir())


def class_names_from_cfg(cfg, class_names_text: str, num_classes: int) -> list[str]:
    if class_names_text.strip():
        names = [name.strip() for name in class_names_text.split(",") if name.strip()]
    elif hasattr(cfg, "data") and hasattr(cfg.data, "names"):
        names = list(cfg.data.names)
    else:
        names = SCANNET20_CLASS_NAMES
    if len(names) != num_classes:
        raise ValueError(f"class name count {len(names)} does not match num_classes={num_classes}")
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


def clone_scene(scene: dict) -> dict:
    out = {}
    for key, value in scene.items():
        out[key] = value.copy() if isinstance(value, np.ndarray) else value
    return out


def move_to_cuda(batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def inference_batch(batch: dict) -> dict:
    return {key: value for key, value in batch.items() if key not in {"segment"}}


@torch.no_grad()
def forward_point_features(model, point_cls, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = batch["segment"].long()
    if hasattr(model, "seg_head") and hasattr(model, "backbone"):
        point = point_cls(inference_batch(batch))
        point = model.backbone(point)
        feat = point.feat.float()
        logits = model.seg_head(feat).float()
    else:
        out = model(inference_batch(batch))
        logits = out["seg_logits"].float()
        feat = logits.float()
    if feat.shape[0] != logits.shape[0] or feat.shape[0] != labels.shape[0]:
        raise RuntimeError(f"shape mismatch feat={feat.shape} logits={logits.shape} labels={labels.shape}")
    return feat, logits, labels
