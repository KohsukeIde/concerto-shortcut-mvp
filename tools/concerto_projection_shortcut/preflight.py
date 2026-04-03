#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_CONFIGS = (
    "pretrain-concerto-v1m1-0-probe-enc2d-baseline",
    "pretrain-concerto-v1m1-0-probe-enc2d-cross-scene-target-swap",
)
EXPECTED_BATCH_KEYS = ("images", "global_correspondence", "img_num")
REQUIRED_SPLITS = ("Training", "Validation")


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def print_check(label: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "fail"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def ensure_repo_layout(repo_root: Path) -> None:
    expected = [
        repo_root / "scripts" / "train.sh",
        repo_root / "pointcept" / "models" / "concerto" / "concerto_v1m1_base.py",
        repo_root / "pointcept" / "models" / "concerto" / "README.md",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        for path in missing:
            print_check("repo path", False, str(path))
        raise SystemExit(1)
    for path in expected:
        print_check("repo path", True, str(path.relative_to(repo_root)))


def resolve_asset_path(data_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path

    candidates = [data_root / path]
    if len(path.parts) >= 1 and path.parts[0] == "data":
        candidates.append(data_root.joinpath(*path.parts[1:]))
    if len(path.parts) >= 2 and path.parts[:2] == ("data", "arkitscenes"):
        candidates.append(data_root.joinpath(*path.parts[2:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_split_file(split_file: Path) -> dict:
    with split_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{split_file} must contain a JSON object.")
    if not payload:
        raise ValueError(f"{split_file} is empty.")
    return payload


def validate_sample_record(data_root: Path, split_name: str, record_name: str, record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{split_name}:{record_name} is not a JSON object.")
    for key in ("pointclouds", "images", "correspondences"):
        if key not in record:
            raise ValueError(f"{split_name}:{record_name} missing '{key}'.")

    pointclouds_path = resolve_asset_path(data_root, record["pointclouds"])
    if not pointclouds_path.exists():
        raise ValueError(f"{split_name}:{record_name} pointclouds path missing: {pointclouds_path}")
    coord_path = pointclouds_path / "coord.npy"
    if not coord_path.exists():
        raise ValueError(f"{split_name}:{record_name} missing coord.npy in {pointclouds_path}")

    image_paths = [resolve_asset_path(data_root, item) for item in record["images"]]
    corr_paths = [resolve_asset_path(data_root, item) for item in record["correspondences"]]
    if not image_paths:
        raise ValueError(f"{split_name}:{record_name} has no images.")
    if len(image_paths) != len(corr_paths):
        raise ValueError(
            f"{split_name}:{record_name} images/correspondences length mismatch: "
            f"{len(image_paths)} vs {len(corr_paths)}"
        )
    for path in image_paths[:2]:
        if not path.exists():
            raise ValueError(f"{split_name}:{record_name} image missing: {path}")
    for path in corr_paths[:2]:
        if not path.exists():
            raise ValueError(f"{split_name}:{record_name} correspondence missing: {path}")


def check_data_layout(data_root: Path) -> None:
    if not data_root.exists():
        print_check("data root", False, str(data_root))
        raise SystemExit(1)
    print_check("data root", True, str(data_root))

    splits_dir = data_root / "splits"
    if not splits_dir.exists():
        print_check("split dir", False, str(splits_dir))
        raise SystemExit(1)

    for split_name in REQUIRED_SPLITS:
        split_file = splits_dir / f"{split_name}.json"
        if not split_file.exists():
            print_check("split file", False, str(split_file))
            raise SystemExit(1)
        payload = load_split_file(split_file)
        record_name, record = next(iter(payload.items()))
        validate_sample_record(data_root, split_name, record_name, record)
        print_check("split file", True, f"{split_name} ({len(payload)} samples)")


def parse_python_file(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def import_config(repo_root: Path, config_name: str):
    config_path = repo_root / "configs" / "concerto" / f"{config_name}.py"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    parse_python_file(config_path)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        config_mod = importlib.import_module("pointcept.utils.config")
    except ModuleNotFoundError as exc:
        print_check("config syntax", True, str(config_path.relative_to(repo_root)))
        print_check("config import", False, f"{config_name}: missing dependency {exc.name}")
        return None

    cfg = config_mod.Config.fromfile(str(config_path))
    print_check("config import", True, config_name)
    return cfg


def move_batch_to_cuda(batch):
    import torch

    if isinstance(batch, dict):
        return {
            key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    raise TypeError("Expected collated batch to be a dict.")


def check_batch_and_forward(repo_root: Path, cfg, do_forward: bool) -> None:
    import torch
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn
    from pointcept.models.builder import build_model

    dataset = build_dataset(cfg.data.train)
    sample = dataset[0]
    batch = point_collate_fn([sample], mix_prob=0)
    for key in EXPECTED_BATCH_KEYS:
        if key not in batch:
            raise KeyError(f"batch missing {key}")
    print_check("batch keys", True, ", ".join(EXPECTED_BATCH_KEYS))

    if not do_forward:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Concerto forward preflight.")

    model = build_model(cfg.model).cuda()
    model.train()
    batch = move_batch_to_cuda(batch)
    with torch.no_grad():
        output = model(batch)
    loss = output["loss"]
    enc2d_loss = output["enc2d_loss"]
    if not torch.isfinite(loss):
        raise RuntimeError("loss is not finite")
    if not torch.isfinite(enc2d_loss):
        raise RuntimeError("enc2d_loss is not finite")
    print_check(
        "forward",
        True,
        f"loss={loss.item():.4f}, enc2d_loss={enc2d_loss.item():.4f}",
    )


def iter_configs(configs: Iterable[str]) -> list[str]:
    return [item for item in configs if item]


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight for the Concerto shortcut MVP.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-batch", action="store_true")
    parser.add_argument("--check-forward", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = (args.data_root or (repo_root / "data" / "arkitscenes")).resolve()
    configs = iter_configs(args.config) or list(DEFAULT_CONFIGS)

    ensure_repo_layout(repo_root)
    if args.check_data:
        check_data_layout(data_root)

    cfg_objects = []
    for config_name in configs:
        cfg = import_config(repo_root, config_name)
        if cfg is not None:
            if args.data_root is not None and hasattr(cfg.data, "train"):
                cfg.data.train.data_root = str(data_root)
            cfg_objects.append((config_name, cfg))

    if args.check_batch or args.check_forward:
        if not cfg_objects:
            raise SystemExit("No importable configs available for batch/forward checks.")
        config_name, cfg = cfg_objects[0]
        print(f"[info] batch/forward check uses {config_name}")
        check_batch_and_forward(repo_root, cfg, do_forward=args.check_forward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
