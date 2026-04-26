#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_cross_model_fusion_scannet20 import (  # noqa: E402
    build_utonia_scene_transform,
    current_raw_scene_from_dataset,
    forward_current_raw_logits,
    load_cached_expert_logits,
    parse_cached_experts,
    parse_current_specs,
    resolve,
    scene_name_from_dataset,
    softmax_logits,
    transform_utonia_scene,
)
from tools.concerto_projection_shortcut.eval_masking_battery import build_loader
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    build_model,
    load_config,
    move_to_cuda,
    summarize_confusion,
    update_confusion,
    weak_mean,
)
from tools.concerto_projection_shortcut.eval_utonia_scannet_support_stress import (  # noqa: E402
    build_model as build_utonia_model,
    forward_raw_logits as forward_utonia_raw_logits,
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw-point aligned 5-expert fusion predictions and re-score saved predictions."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-prefix", type=Path, required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--full-scene-chunk-size", type=int, default=2048)
    parser.add_argument("--current-model", action="append", default=[])
    parser.add_argument("--include-utonia", action="store_true")
    parser.add_argument("--utonia-weight", type=Path, default=Path("data/weights/utonia/utonia.pth"))
    parser.add_argument("--utonia-head", type=Path, default=Path("data/weights/utonia/utonia_linear_prob_head_sc.pth"))
    parser.add_argument("--disable-utonia-flash", action="store_true")
    parser.add_argument("--cached-expert", action="append", default=[], help="Repeated spec name::cache_dir.")
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    return parser.parse_args()


def parse_weak_ids(text: str) -> list[int]:
    return [NAME_TO_ID[x.strip()] for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def picture_to_wall(conf: np.ndarray) -> float:
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    den = conf[pic].sum()
    return float(conf[pic, wall] / den) if den else float("nan")


def save_pred(root: Path, variant: str, scene_name: str, pred: torch.Tensor, class2id=None) -> None:
    save_dir = root / variant / "pred"
    submit_dir = root / variant / "submit"
    save_dir.mkdir(parents=True, exist_ok=True)
    submit_dir.mkdir(parents=True, exist_ok=True)
    arr = pred.cpu().numpy().astype(np.int16)
    np.save(save_dir / f"{scene_name}.npy", arr)
    if class2id is not None:
        submit = np.asarray(class2id[arr], dtype=np.int32)
    else:
        submit = arr.astype(np.int32)
    np.savetxt(submit_dir / f"{scene_name}.txt", submit.reshape([-1, 1]), fmt="%d")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = resolve(repo_root, args.data_root)
    out_dir = resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_prefix = resolve(repo_root, args.summary_prefix)
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    current_specs = parse_current_specs(args)
    cached_experts = parse_cached_experts(args.cached_expert, repo_root)
    weak_ids = parse_weak_ids(args.weak_classes)

    cfg0 = load_config(resolve(repo_root, current_specs[0].config))
    loader = build_loader(cfg0, args.val_split, data_root, args.batch_size, args.num_worker)
    current_models = []
    for spec in current_specs:
        cfg = load_config(resolve(repo_root, spec.config))
        current_models.append((spec.name, build_model(cfg, resolve(repo_root, spec.weight)).cuda().eval()))

    utonia_model = utonia_head = utonia_transform = None
    if args.include_utonia:
        utonia_model, utonia_head = build_utonia_model(
            resolve(repo_root, args.utonia_weight),
            resolve(repo_root, args.utonia_head),
            args.disable_utonia_flash,
        )
        utonia_transform = build_utonia_scene_transform()

    num_classes = len(SCANNET20_CLASS_NAMES)
    confs = {
        "fullft_single_saved": torch.zeros((num_classes, num_classes), dtype=torch.long),
        "avgprob_all_saved": torch.zeros((num_classes, num_classes), dtype=torch.long),
    }
    class2id = getattr(loader.dataset, "class2id", None)
    if class2id is not None:
        class2id = np.asarray(class2id)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            labels = batch["origin_segment"].long().cpu()
            scene_name = scene_name_from_dataset(loader.dataset, batch_idx)
            logits_by_name = {}
            for name, model in current_models:
                logits, got_labels = forward_current_raw_logits(model, batch, args.full_scene_chunk_size)
                if not torch.equal(got_labels, labels):
                    raise RuntimeError(f"label mismatch for {name} scene={scene_name}")
                logits_by_name[name] = logits
            if utonia_model is not None:
                raw_scene = current_raw_scene_from_dataset(loader.dataset, batch_idx)
                ubatch = transform_utonia_scene(utonia_transform, raw_scene)
                logits, got_labels = forward_utonia_raw_logits(utonia_model, utonia_head, ubatch)
                if not torch.equal(got_labels, labels):
                    raise RuntimeError(f"label mismatch for Utonia scene={scene_name}")
                logits_by_name["Utonia"] = logits
            for name, cache_dir in cached_experts:
                logits_by_name[name] = load_cached_expert_logits(cache_dir, scene_name, labels)

            fullft_name = next((name for name in logits_by_name if "fullFT" in name or "fullft" in name), None)
            if fullft_name is None:
                raise RuntimeError("no cached/fullFT expert name found; expected name containing fullFT")
            fullft_pred = logits_by_name[fullft_name].argmax(dim=1)
            avg_probs = torch.stack([softmax_logits(logits) for logits in logits_by_name.values()], dim=0).mean(dim=0)
            avg_pred = avg_probs.argmax(dim=1)
            save_pred(out_dir, "fullft_single", scene_name, fullft_pred, class2id)
            save_pred(out_dir, "avgprob_all", scene_name, avg_pred, class2id)
            update_confusion(confs["fullft_single_saved"], fullft_pred.cpu(), labels.cpu(), num_classes, -1)
            update_confusion(confs["avgprob_all_saved"], avg_pred.cpu(), labels.cpu(), num_classes, -1)
            if (batch_idx + 1) % 25 == 0:
                print(f"[export] scenes={batch_idx + 1}/{len(loader.dataset)}", flush=True)

    rows = []
    for variant, conf_t in confs.items():
        conf = conf_t.numpy()
        summary = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
        rows.append(
            {
                "variant": variant,
                "mIoU": summary["mIoU"],
                "mAcc": summary["mAcc"],
                "allAcc": summary["allAcc"],
                "weak_mean_iou": weak_mean(summary, weak_ids),
                "picture_iou": float(summary["iou"][NAME_TO_ID["picture"]]),
                "picture_to_wall": picture_to_wall(conf),
                "note": "saved raw-point predictions; same labels/metric as raw-aligned fusion, not Pointcept test-time fragment voting",
            }
        )
    write_csv(summary_prefix.with_suffix(".csv"), rows)
    write_csv(out_dir / "saved_prediction_eval.csv", rows)
    md = [
        "# Cross-Model Fusion Saved-Prediction Evaluation",
        "",
        "Predictions are saved scene-wise in Pointcept-style `pred/*.npy` and `submit/*.txt` folders, then re-scored from the saved raw-point predictions.",
        "",
        "This verifies the save/eval path for fusion predictions. It does not add Pointcept test-time fragment voting; therefore it should match the raw-aligned fusion protocol rather than the `0.8075` Pointcept full-FT precise/test result.",
        "",
        "| variant | mIoU | mAcc | allAcc | weak mIoU | picture | p->wall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| `{row['variant']}` | `{float(row['mIoU']):.4f}` | `{float(row['mAcc']):.4f}` | "
            f"`{float(row['allAcc']):.4f}` | `{float(row['weak_mean_iou']):.4f}` | "
            f"`{float(row['picture_iou']):.4f}` | `{float(row['picture_to_wall']):.4f}` |"
        )
    summary_prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "current_models": [spec.__dict__ | {"config": str(spec.config), "weight": str(spec.weight)} for spec in current_specs],
                "include_utonia": args.include_utonia,
                "cached_experts": [(name, str(path)) for name, path in cached_experts],
                "note": "Saved raw-point predictions for fusion; not a multi-fragment Pointcept tester run.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {summary_prefix.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
