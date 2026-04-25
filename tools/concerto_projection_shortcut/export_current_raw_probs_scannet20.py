#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_cross_model_fusion_scannet20 import (  # noqa: E402
    forward_current_raw_logits,
    scene_name_from_dataset,
)
from tools.concerto_projection_shortcut.eval_masking_battery import build_loader, inference_batch  # noqa: E402
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    build_model,
    load_config,
    move_to_cuda,
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw-point aligned probabilities for a current-repo Pointcept "
            "ScanNet20 segmentation checkpoint. The output format matches cached "
            "experts consumed by eval_cross_model_fusion_scannet20.py."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--model-name", default="current_model")
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--full-scene-chunk-size", type=int, default=2048)
    parser.add_argument("--save-logits", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


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
    config = resolve(repo_root, Path(args.config))
    weight = resolve(repo_root, args.weight)
    data_root = resolve(repo_root, args.data_root)
    out_dir = resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config)
    loader = build_loader(cfg, args.split, data_root, args.batch_size, args.num_worker)
    if len(loader.dataset) == 0:
        raise RuntimeError(f"empty {args.split} dataset at {data_root}")
    model = build_model(cfg, weight).cuda().eval()

    total = 0
    correct = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            scene_name = scene_name_from_dataset(loader.dataset, batch_idx)
            logits, labels = forward_current_raw_logits(model, batch, args.full_scene_chunk_size)
            probs = torch.softmax(logits.float(), dim=1)
            pred = probs.argmax(dim=1)
            write_scene_cache(out_dir / f"{scene_name}.npz", probs.cpu(), pred.cpu(), labels.cpu(), args.save_logits)
            total += int(labels.numel())
            correct += int((pred.cpu() == labels.cpu()).sum().item())
            if (batch_idx + 1) % 25 == 0:
                print(
                    f"[export] {args.model_name} scenes={batch_idx + 1}/{len(loader.dataset)} "
                    f"raw_acc={correct / max(total, 1):.4f}",
                    flush=True,
                )

    metadata = {
        "model_name": args.model_name,
        "config": str(config),
        "weight": str(weight),
        "data_root": str(data_root),
        "split": args.split,
        "num_scenes": len(list(out_dir.glob("*.npz"))),
        "num_points": total,
        "raw_allAcc": correct / max(total, 1),
        "class_names": SCANNET20_CLASS_NAMES,
        "format": "npz per scene with probs float16 [N, C], pred uint8 [N], labels int16 [N]",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {metadata['num_scenes']} scene caches to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
