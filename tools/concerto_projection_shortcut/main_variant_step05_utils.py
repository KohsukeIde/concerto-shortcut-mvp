from __future__ import annotations

import copy
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    train_splits: tuple[str, ...]
    val_splits: tuple[str, ...]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_on_path(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_cfg(repo_root: Path, config_name: str):
    ensure_repo_on_path(repo_root)
    from pointcept.utils.config import Config

    return Config.fromfile(str(repo_root / "configs" / "concerto" / f"{config_name}.py"))


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_dataset_specs(repo_root: Path) -> list[DatasetSpec]:
    data = Path(os.environ.get("POINTCEPT_DATA_ROOT", repo_root / "data"))
    return [
        DatasetSpec("arkit", data / "arkitscenes_absmeta", ("Training",), ("Validation",)),
        DatasetSpec(
            "scannet",
            data / "concerto_scannet_imagepoint_absmeta",
            ("train",),
            ("val",),
        ),
        DatasetSpec(
            "scannetpp",
            data / "concerto_scannetpp_imagepoint_absmeta",
            ("train",),
            ("val",),
        ),
        DatasetSpec(
            "s3dis",
            data / "concerto_s3dis_imagepoint_absmeta",
            ("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
            ("Area_5",),
        ),
        DatasetSpec("hm3d", data / "concerto_hm3d_imagepoint_absmeta", ("train",), ("val",)),
        DatasetSpec(
            "structured3d",
            data / "concerto_structured3d_imagepoint_absmeta",
            ("train",),
            ("val",),
        ),
    ]


def select_dataset_specs(
    repo_root: Path, names: Iterable[str], allow_missing: bool = False
) -> list[DatasetSpec]:
    requested = set(names)
    specs = [spec for spec in default_dataset_specs(repo_root) if spec.name in requested]
    known = {spec.name for spec in specs}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown dataset names: {unknown}")
    missing = [
        spec.name
        for spec in specs
        if not any((spec.root / "splits" / f"{split}.json").is_file() for split in spec.train_splits)
        or not any((spec.root / "splits" / f"{split}.json").is_file() for split in spec.val_splits)
    ]
    if missing and not allow_missing:
        raise FileNotFoundError(
            "missing Concerto image-point absmeta roots for datasets: "
            + ", ".join(missing)
            + ". Run tools/concerto_projection_shortcut/setup_concerto_six_imagepoint.sh first."
        )
    return [spec for spec in specs if spec.name not in missing]


def build_loader(
    cfg,
    spec: DatasetSpec,
    split_kind: str,
    batch_size: int,
    num_worker: int,
    shuffle: bool,
):
    ensure_repo_on_path(repo_root_from_here())
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn

    splits = spec.train_splits if split_kind == "train" else spec.val_splits
    dataset_cfg = dict(
        type="DefaultImagePointDataset",
        crop_h=int(cfg.crop_h),
        crop_w=int(cfg.crop_w),
        patch_size=int(cfg.patch_size),
        split=list(splits),
        data_root=str(spec.root.resolve()),
        transform=copy.deepcopy(cfg.transform),
        test_mode=False,
        loop=1,
    )
    dataset = build_dataset(dataset_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        collate_fn=lambda batch: point_collate_fn(batch, mix_prob=0),
        pin_memory=True,
    )


def move_batch_to_cuda(batch):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.cuda(non_blocking=True)
    return batch


def clean_state_dict(weight_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def load_backbone_or_full(model, weight_path: Path) -> dict[str, int]:
    cleaned = clean_state_dict(weight_path)
    bare_backbone = not any(
        key.startswith(("student.", "teacher.", "enc2d_", "patch_proj"))
        for key in cleaned
    )
    if bare_backbone and hasattr(model, "student") and "backbone" in model.student:
        student_info = model.student.backbone.load_state_dict(cleaned, strict=False)
        teacher_info = model.teacher.backbone.load_state_dict(cleaned, strict=False)
        return {
            "bare_backbone": 1,
            "student_missing": len(student_info.missing_keys),
            "student_unexpected": len(student_info.unexpected_keys),
            "teacher_missing": len(teacher_info.missing_keys),
            "teacher_unexpected": len(teacher_info.unexpected_keys),
        }
    info = model.load_state_dict(cleaned, strict=False)
    return {
        "bare_backbone": 0,
        "missing": len(info.missing_keys),
        "unexpected": len(info.unexpected_keys),
    }


def build_main_variant_model(repo_root: Path, config_name: str, weight_path: Path):
    ensure_repo_on_path(repo_root)
    from pointcept.models.builder import build_model

    cfg = load_cfg(repo_root, config_name)
    model = build_model(cfg.model).cuda()
    load_info = load_backbone_or_full(model, weight_path.resolve())
    print(f"[model] loaded {weight_path}: {load_info}")
    return cfg, model, load_info


def freeze_all_but_patch_proj(model) -> None:
    model.requires_grad_(False)
    if hasattr(model, "enc2d_model"):
        model.enc2d_model.requires_grad_(False)
    if not hasattr(model, "patch_proj"):
        raise AttributeError("model does not have patch_proj; enc2d_loss_weight must be > 0")
    model.patch_proj.requires_grad_(True)


def trainable_parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def iter_limited(loader, max_batches: int):
    for batch_idx, batch in enumerate(loader):
        if max_batches >= 0 and batch_idx >= max_batches:
            break
        yield batch_idx, batch


def evaluate_enc2d_modes(
    model,
    loaders: dict[str, DataLoader],
    modes: list[str],
    max_batches: int,
) -> list[dict]:
    old_mode = model.shortcut_probe["mode"]
    rows = []
    model.eval()
    with torch.inference_mode():
        for dataset_name, loader in loaders.items():
            for mode in modes:
                model.shortcut_probe["mode"] = mode
                losses = []
                for _, batch in iter_limited(loader, max_batches):
                    batch = move_batch_to_cuda(batch)
                    output = model(batch)
                    losses.append(float(output["enc2d_loss"].detach().cpu().item()))
                rows.append(
                    {
                        "dataset": dataset_name,
                        "mode": mode,
                        "batches": len(losses),
                        "enc2d_loss_mean": sum(losses) / max(len(losses), 1),
                    }
                )
    model.shortcut_probe["mode"] = old_mode
    return rows


def add_deltas(rows: list[dict]) -> list[dict]:
    baselines = {
        row["dataset"]: float(row["enc2d_loss_mean"])
        for row in rows
        if row["mode"] == "none"
    }
    out = []
    for row in rows:
        item = dict(row)
        base = baselines.get(row["dataset"])
        item["delta_vs_baseline"] = (
            float(row["enc2d_loss_mean"]) - base if base is not None else ""
        )
        out.append(item)
    return out


def write_causal_results(rows: list[dict], csv_path: Path, md_path: Path, title: str) -> None:
    rows = add_deltas(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "mode", "batches", "enc2d_loss_mean", "delta_vs_baseline"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "mode": row["mode"],
                    "batches": row["batches"],
                    "enc2d_loss_mean": f"{float(row['enc2d_loss_mean']):.6f}",
                    "delta_vs_baseline": ""
                    if row["delta_vs_baseline"] == ""
                    else f"{float(row['delta_vs_baseline']):.6f}",
                }
            )
    lines = [
        f"# {title}",
        "",
        "| dataset | mode | batches | enc2d loss mean | delta vs baseline |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = row["delta_vs_baseline"]
        delta_text = "" if delta == "" else f"{float(delta):.6f}"
        lines.append(
            f"| {row['dataset']} | {row['mode']} | {row['batches']} | "
            f"{float(row['enc2d_loss_mean']):.6f} | {delta_text} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cosine_loss_scaled(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(pred, target, dim=1, eps=1e-6)).mean() * 10.0


def shifted_pred(pred: torch.Tensor, target_shifted: bool) -> torch.Tensor:
    if target_shifted:
        return pred - pred.mean(dim=-1, keepdim=True)
    return pred


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
