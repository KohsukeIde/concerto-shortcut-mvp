#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (
    SCANNET20_CLASS_NAMES,
    bootstrap_ci,
    fit_probe,
)
from tools.concerto_projection_shortcut.ptv3_v151_compat_utils import (
    build_official_model,
    class_names_from_cfg,
    clone_scene,
    forward_point_features,
    load_config,
    load_scene,
    move_to_cuda,
    repo_root_from_here,
    scene_paths,
    setup_official_imports,
)

ACTIVE_CLASS_NAMES = list(SCANNET20_CLASS_NAMES)


def parse_pairs_with_vocab(text: str, names: list[str]) -> list[tuple[int, int]]:
    name_to_id = {name: idx for idx, name in enumerate(names)}
    pairs = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        pos_name, neg_name = [x.strip() for x in chunk.split(":", 1)]
        if pos_name not in name_to_id or neg_name not in name_to_id:
            raise KeyError(f"unknown class pair: {chunk}")
        pairs.append((name_to_id[pos_name], name_to_id[neg_name]))
    if not pairs:
        raise ValueError("no class pairs parsed")
    return pairs


def pair_name_with_vocab(pair: tuple[int, int]) -> str:
    return f"{ACTIVE_CLASS_NAMES[pair[0]]}_vs_{ACTIVE_CLASS_NAMES[pair[1]]}".replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Point-level stage-wise trace for official PTv3 v1.5.1 checkpoints "
            "using the official v1.5.1 model/transform path on current npy scene roots."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--official-root", type=Path, default=Path("data/tmp/Pointcept-v1.5.1"))
    parser.add_argument("--config", default="configs/scannet/semseg-pt-v3m1-0-base.py")
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--segment-key", default="segment20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_ptv3_v151_point_stagewise_trace"),
    )
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,door:wall",
        help="Comma-separated positive:negative class-name pairs.",
    )
    parser.add_argument("--max-train-scenes", type=int, default=256)
    parser.add_argument("--max-val-scenes", type=int, default=128)
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--logreg-steps", type=int, default=600)
    parser.add_argument("--logreg-lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--class-names", default="")
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_split(
    args: argparse.Namespace,
    model,
    point_cls,
    transform,
    split: str,
    max_scenes: int,
    target_classes: set[int],
):
    features = {"point_feature": [], "linear_logits": []}
    labels: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    class_counts = {cls: 0 for cls in sorted(target_classes)}
    seen_scenes = 0
    paths = scene_paths(args.data_root, split)
    with torch.inference_mode():
        for scene_idx, scene_path in enumerate(paths):
            if max_scenes >= 0 and scene_idx >= max_scenes:
                break
            batch = move_to_cuda(transform(clone_scene(load_scene(scene_path, args.segment_key))))
            feat, logits, scene_labels = forward_point_features(model, point_cls, batch)
            seen_scenes += 1
            valid = (scene_labels >= 0) & (scene_labels < args.num_classes)
            target_mask = torch.zeros_like(valid)
            for cls in target_classes:
                target_mask |= scene_labels == cls
            keep = valid & target_mask
            if not keep.any():
                continue
            feat = feat[keep].float().cpu()
            logits = logits[keep].float().cpu()
            scene_labels = scene_labels[keep].long().cpu()
            scene_preds = logits.argmax(dim=1).long()
            kept_indices = []
            for cls in sorted(target_classes):
                cls_mask = scene_labels == cls
                if not cls_mask.any():
                    continue
                room = args.max_per_class - class_counts[cls]
                if room <= 0:
                    continue
                idx = cls_mask.nonzero(as_tuple=False).flatten()
                if idx.numel() > room:
                    idx = idx[:room]
                kept_indices.append(idx)
                class_counts[cls] += int(idx.numel())
            if kept_indices:
                idx = torch.cat(kept_indices, dim=0)
                for name, value in {"point_feature": feat, "linear_logits": logits}.items():
                    features[name].append(value[idx])
                labels.append(scene_labels[idx])
                preds.append(scene_preds[idx])
            if (scene_idx + 1) % 25 == 0:
                counts = " ".join(f"{ACTIVE_CLASS_NAMES[k]}={v}" for k, v in class_counts.items())
                print(f"[collect] split={split} scene={scene_idx + 1} {counts}", flush=True)
    if not labels:
        raise RuntimeError(f"no target point samples collected for split={split}")
    payload = {
        "class_id": torch.cat(labels, dim=0),
        "pred_id": torch.cat(preds, dim=0),
        "seen_batches": seen_scenes,
        "class_counts": {ACTIVE_CLASS_NAMES[k]: int(v) for k, v in class_counts.items()},
    }
    for name, tensors in features.items():
        payload[name] = torch.cat(tensors, dim=0)
    print(f"[collect] split={split} done counts={payload['class_counts']}", flush=True)
    return payload


def direct_pair_metrics(scores: torch.Tensor, labels: torch.Tensor, iters: int, seed: int):
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu()
    pred = (scores >= 0).float()
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_acc = (pred[pos_mask] == labels[pos_mask]).float().mean().item() if pos_mask.any() else float("nan")
    neg_acc = (pred[neg_mask] == labels[neg_mask]).float().mean().item() if neg_mask.any() else float("nan")
    row = {
        "acc": (pred == labels).float().mean().item(),
        "balanced_acc": float(np.nanmean([pos_acc, neg_acc])),
        "positive_acc": pos_acc,
        "negative_acc": neg_acc,
        "auc": binary_auc(scores, labels),
        "train_samples_probe": 0,
        "pos_weight": "",
    }
    row.update(bootstrap_ci(scores, labels, iters, seed))
    return row


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.float()
    labels = labels.float()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    auc = (diff > 0).float().mean() + 0.5 * (diff == 0).float().mean()
    return float(auc.item())


def evaluate_pair(pair, train, val, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    pos_cls, neg_cls = pair
    train_mask = (train["class_id"] == pos_cls) | (train["class_id"] == neg_cls)
    val_mask = (val["class_id"] == pos_cls) | (val["class_id"] == neg_cls)
    train_y = (train["class_id"][train_mask] == pos_cls).float()
    val_y = (val["class_id"][val_mask] == pos_cls).float()
    if (
        int((train_y == 1).sum().item()) == 0
        or int((train_y == 0).sum().item()) == 0
        or int((val_y == 1).sum().item()) == 0
        or int((val_y == 0).sum().item()) == 0
    ):
        print(
            "[skip] {pair} train_pos={train_pos} train_neg={train_neg} val_pos={val_pos} val_neg={val_neg}".format(
                pair=pair_name_with_vocab(pair),
                train_pos=int((train_y == 1).sum().item()),
                train_neg=int((train_y == 0).sum().item()),
                val_pos=int((val_y == 1).sum().item()),
                val_neg=int((val_y == 0).sum().item()),
            ),
            flush=True,
        )
        return [], []
    rows: list[dict] = []
    for feature in ("point_feature", "linear_logits"):
        for probe in ("unweighted", "balanced", "weighted"):
            result = fit_probe(train[feature][train_mask], train_y, val[feature][val_mask], val_y, args, probe)
            ci = bootstrap_ci(
                result.pop("logits"),
                result.pop("labels"),
                args.bootstrap_iters,
                args.seed + len(rows) + 211,
            )
            row = {
                "pair": pair_name_with_vocab(pair),
                "positive_class": ACTIVE_CLASS_NAMES[pos_cls],
                "negative_class": ACTIVE_CLASS_NAMES[neg_cls],
                "stage": feature,
                "probe": probe,
                "train_positive": int((train_y == 1).sum().item()),
                "train_negative": int((train_y == 0).sum().item()),
                "val_positive": int((val_y == 1).sum().item()),
                "val_negative": int((val_y == 0).sum().item()),
            }
            row.update(result)
            row.update(ci)
            rows.append(row)

    direct_scores = val["linear_logits"][val_mask, pos_cls] - val["linear_logits"][val_mask, neg_cls]
    direct = direct_pair_metrics(direct_scores, val_y, args.bootstrap_iters, args.seed + 409)
    direct_row = {
        "pair": pair_name_with_vocab(pair),
        "positive_class": ACTIVE_CLASS_NAMES[pos_cls],
        "negative_class": ACTIVE_CLASS_NAMES[neg_cls],
        "stage": "linear_logits",
        "probe": "direct_pair_margin",
        "train_positive": int((train_y == 1).sum().item()),
        "train_negative": int((train_y == 0).sum().item()),
        "val_positive": int((val_y == 1).sum().item()),
        "val_negative": int((val_y == 0).sum().item()),
    }
    direct_row.update(direct)
    rows.append(direct_row)

    confusion_rows: list[dict] = []
    val_labels = val["class_id"][val_mask]
    val_preds = val["pred_id"][val_mask]
    for target in (pos_cls, neg_cls):
        target_mask = val_labels == target
        denom = int(target_mask.sum().item())
        if denom == 0:
            continue
        pred_counts = torch.bincount(val_preds[target_mask], minlength=len(ACTIVE_CLASS_NAMES))
        for pred_id, count_tensor in enumerate(pred_counts.tolist()):
            if count_tensor == 0:
                continue
            confusion_rows.append(
                {
                    "pair": pair_name_with_vocab(pair),
                    "target_id": target,
                    "target_name": ACTIVE_CLASS_NAMES[target],
                    "pred_id": pred_id,
                    "pred_name": ACTIVE_CLASS_NAMES[pred_id],
                    "count": int(count_tensor),
                    "fraction_of_target": float(count_tensor / max(denom, 1)),
                }
            )
    return rows, confusion_rows


def write_outputs(args: argparse.Namespace, rows: list[dict], confusion_rows: list[dict], metadata: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ptv3_v151_point_stagewise_trace.csv"
    fields = [
        "pair",
        "positive_class",
        "negative_class",
        "stage",
        "probe",
        "balanced_acc",
        "bal_acc_std",
        "bal_acc_ci_low",
        "bal_acc_ci_high",
        "auc",
        "auc_std",
        "auc_ci_low",
        "auc_ci_high",
        "positive_acc",
        "negative_acc",
        "acc",
        "train_positive",
        "train_negative",
        "val_positive",
        "val_negative",
        "train_samples_probe",
        "pos_weight",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    confusion_path = args.output_dir / "ptv3_v151_point_stagewise_trace_confusion.csv"
    with confusion_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair", "target_id", "target_name", "pred_id", "pred_name", "count", "fraction_of_target"],
        )
        writer.writeheader()
        writer.writerows(confusion_rows)

    md_path = args.output_dir / "ptv3_v151_point_stagewise_trace.md"
    lines = [
        "# PTv3 v1.5.1 Point-Level Stage-Wise Trace",
        "",
        "## Setup",
        f"- official root: `{args.official_root}`",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- segment key: `{args.segment_key}`",
        f"- train scenes seen: {metadata['train']['seen_batches']}",
        f"- val scenes seen: {metadata['val']['seen_batches']}",
        f"- train class counts: {metadata['train']['class_counts']}",
        f"- val class counts: {metadata['val']['class_counts']}",
        f"- bootstrap iters: {args.bootstrap_iters}",
        "",
        "## Results",
        "",
        "| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {pair} | {stage} | {probe} | {balanced_acc:.4f} | "
            "[{bal_acc_ci_low:.4f}, {bal_acc_ci_high:.4f}] | {auc:.4f} | "
            "[{auc_ci_low:.4f}, {auc_ci_high:.4f}] | {val_positive}/{val_negative} |".format(**row)
        )
    lines += [
        "",
        "## Interpretation Guide",
        "- `point_feature` is the official PTv3 decoder feature before the segmentation head.",
        "- `linear_logits` with `unweighted`/`balanced`/`weighted` fits a binary probe on the fixed 20-way logits.",
        "- `linear_logits` with `direct_pair_margin` uses the fixed logit margin `logit(positive_class) - logit(negative_class)` without refitting a pair probe.",
        "- The companion confusion CSV reports full 20-way predictions for the validation rows belonging to each class pair.",
        "- Train and validation both use the official deterministic validation-style transform so the trace is not affected by train-time random augmentations.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    prefix = args.summary_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    prefix.parent.joinpath(f"{prefix.name}_confusion.csv").write_text(confusion_path.read_text(encoding="utf-8"), encoding="utf-8")
    prefix.with_suffix(".md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[write] {csv_path}", flush=True)
    print(f"[write] {confusion_path}", flush=True)
    print(f"[write] {md_path}", flush=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.official_root = (repo_root / args.official_root).resolve() if not args.official_root.is_absolute() else args.official_root
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    seed_everything(args.seed)
    config_cls, compose_cls, build_model_fn, point_cls = setup_official_imports(args.official_root)
    cfg = load_config(config_cls, args.official_root / args.config)
    names = class_names_from_cfg(cfg, args.class_names, args.num_classes)
    global ACTIVE_CLASS_NAMES
    ACTIVE_CLASS_NAMES = list(names)
    pairs = parse_pairs_with_vocab(args.class_pairs, names)
    target_classes = {cls for pair in pairs for cls in pair}
    transform = compose_cls(cfg.data.val.transform)
    print(f"[pairs] {[pair_name_with_vocab(pair) for pair in pairs]}", flush=True)
    if args.dry_run:
        scene = scene_paths(args.data_root, args.val_split)[0]
        batch = transform(clone_scene(load_scene(scene, args.segment_key)))
        print(f"[dry] scene={scene.name} keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] target_classes={[ACTIVE_CLASS_NAMES[i] for i in sorted(target_classes)]}", flush=True)
        return 0

    model = build_official_model(build_model_fn, cfg, args.weight)
    train = collect_split(args, model, point_cls, transform, args.train_split, args.max_train_scenes, target_classes)
    val = collect_split(args, model, point_cls, transform, args.val_split, args.max_val_scenes, target_classes)
    rows: list[dict] = []
    confusion_rows: list[dict] = []
    for pair in pairs:
        pair_rows, pair_confusion = evaluate_pair(pair, train, val, args)
        rows.extend(pair_rows)
        confusion_rows.extend(pair_confusion)
    metadata = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pairs": [pair_name_with_vocab(pair) for pair in pairs],
        "train": {"seen_batches": train["seen_batches"], "class_counts": train["class_counts"]},
        "val": {"seen_batches": val["seen_batches"], "class_counts": val["class_counts"]},
    }
    write_outputs(args, rows, confusion_rows, metadata)
    for row in rows:
        if row["probe"] in {"balanced", "direct_pair_margin"}:
            print(
                "[result] {pair}/{stage}/{probe} bal={balanced_acc:.4f} "
                "ci=[{bal_acc_ci_low:.4f},{bal_acc_ci_high:.4f}] auc={auc:.4f}".format(**row),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
