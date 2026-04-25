#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_cross_model_fusion_scannet20 import (  # noqa: E402
    CurrentModelSpec,
    build_utonia_scene_transform,
    current_raw_scene_from_dataset,
    forward_current_raw_logits,
    load_cached_expert_logits,
    parse_current_specs,
    parse_cached_experts,
    parse_weak_ids,
    picture_to_wall,
    resolve,
    scene_name_from_dataset,
    softmax_logits,
    transform_utonia_scene,
    write_csv,
)
from tools.concerto_projection_shortcut.eval_masking_battery import build_loader  # noqa: E402
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
        description=(
            "Two-fold scene-level CV pilot for trainable cross-model logit/probability "
            "stacking on ScanNet20 val. This is a bounded method-direction test, "
            "not a final train-split baseline."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--full-scene-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--sample-points-per-scene", type=int, default=4096)
    parser.add_argument("--stacker-epochs", type=int, default=60)
    parser.add_argument("--stacker-batch-size", type=int, default=8192)
    parser.add_argument("--stacker-lr", type=float, default=0.03)
    parser.add_argument("--stacker-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--class-weight-powers",
        default="0.0,0.5",
        help="Comma-separated class weight powers. 0.0 is unweighted CE; 0.5 is inverse-sqrt frequency.",
    )
    parser.add_argument("--current-model", action="append", default=[])
    parser.add_argument("--include-utonia", action="store_true")
    parser.add_argument("--utonia-weight", type=Path, default=Path("data/weights/utonia/utonia.pth"))
    parser.add_argument("--utonia-head", type=Path, default=Path("data/weights/utonia/utonia_linear_prob_head_sc.pth"))
    parser.add_argument("--disable-utonia-flash", action="store_true")
    parser.add_argument("--cached-expert", action="append", default=[], help="Repeated spec name::cache_dir.")
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--summary-prefix", type=Path, default=Path("tools/concerto_projection_shortcut/results_cross_model_fusion_cv_stacker_scannet20"))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def default_specs(args: argparse.Namespace) -> list[CurrentModelSpec]:
    specs = parse_current_specs(args)
    return specs


def fusion_features(logits_by_model: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    probs = [softmax_logits(logits_by_model[name]) for name in names]
    # Probability features are more stable across model families than raw logits.
    return torch.cat(probs, dim=1).float()


def stacked_probs(logits_by_model: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.stack([softmax_logits(logits_by_model[name]) for name in names], dim=1).float()


def oracle_expert_target(logits_by_model: dict[str, torch.Tensor], names: list[str], labels: torch.Tensor) -> torch.Tensor:
    probs = stacked_probs(logits_by_model, names)
    n = labels.numel()
    gather_labels = labels.clamp_min(0).clamp_max(probs.shape[2] - 1)
    correct_prob = probs[torch.arange(n), :, gather_labels].transpose(0, 1).transpose(0, 1)
    pred = probs.argmax(dim=2)
    correct = pred == labels[:, None]
    masked = correct_prob.masked_fill(~correct, -1.0)
    any_correct = correct.any(dim=1)
    best_correct = masked.argmax(dim=1)
    best_prob = correct_prob.argmax(dim=1)
    return torch.where(any_correct, best_correct, best_prob).long()


def sample_scene_features(
    feats: torch.Tensor,
    labels: torch.Tensor,
    router_target: torch.Tensor,
    max_points: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (labels >= 0) & (labels < len(SCANNET20_CLASS_NAMES))
    feats = feats[valid]
    labels = labels[valid]
    router_target = router_target[valid]
    n = labels.numel()
    if max_points <= 0 or n <= max_points:
        idx = torch.arange(n)
    else:
        idx = torch.randperm(n, generator=generator)[:max_points]
    return feats[idx].cpu(), labels[idx].cpu(), router_target[idx].cpu()


def class_weights(labels: torch.Tensor, num_classes: int, power: float) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = torch.ones(num_classes)
    nz = counts > 0
    if nz.any():
        weights[nz] = (counts[nz].mean() / counts[nz].clamp_min(1.0)).pow(power)
        weights = weights / weights[nz].mean().clamp_min(1e-6)
    return weights


def train_stacker(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    weight_power: float,
    device: torch.device,
) -> torch.nn.Module:
    model = torch.nn.Linear(x.shape[1], num_classes).to(device)
    weights = class_weights(y, num_classes, weight_power).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x.float(), y.long()), batch_size=batch_size, shuffle=True, num_workers=0)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(yb.numel())
            seen += int(yb.numel())
        if epoch in {0, epochs - 1} or (epoch + 1) % 20 == 0:
            print(f"[stacker] epoch={epoch + 1}/{epochs} loss={total / max(seen, 1):.4f}", flush=True)
    model.eval()
    return model


def train_router(
    x: torch.Tensor,
    target: torch.Tensor,
    num_experts: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.nn.Module:
    model = torch.nn.Linear(x.shape[1], num_experts).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x.float(), target.long()), batch_size=batch_size, shuffle=True, num_workers=0)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(yb.numel())
            seen += int(yb.numel())
        if epoch in {0, epochs - 1} or (epoch + 1) % 20 == 0:
            print(f"[router] epoch={epoch + 1}/{epochs} loss={total / max(seen, 1):.4f}", flush=True)
    model.eval()
    return model


def load_models(args: argparse.Namespace, repo_root: Path):
    specs = default_specs(args)
    cfg0 = load_config(resolve(repo_root, specs[0].config))
    data_root = resolve(repo_root, args.data_root)
    loader = build_loader(cfg0, args.val_split, data_root, args.batch_size, args.num_worker)
    if len(loader.dataset) == 0:
        raise RuntimeError(f"empty dataset at data_root={data_root}")
    current = []
    for spec in specs:
        cfg = load_config(resolve(repo_root, spec.config))
        current.append((spec.name, build_model(cfg, resolve(repo_root, spec.weight)).cuda().eval()))
    utonia = None
    if args.include_utonia:
        model, head = build_utonia_model(
            resolve(repo_root, args.utonia_weight),
            resolve(repo_root, args.utonia_head),
            args.disable_utonia_flash,
        )
        utonia = (model, head, build_utonia_scene_transform())
    cached = parse_cached_experts(args.cached_expert, repo_root)
    names = [name for name, _ in current] + (["Utonia"] if utonia is not None else []) + [name for name, _ in cached]
    return loader, current, utonia, cached, names


@torch.no_grad()
def scene_logits(
    loader,
    current_models,
    utonia,
    cached_experts,
    batch_idx: int,
    batch: dict,
    chunk_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = move_to_cuda(batch)
    labels_ref = batch["origin_segment"].long().cpu()
    logits_by_model: dict[str, torch.Tensor] = {}
    for name, model in current_models:
        logits, labels = forward_current_raw_logits(model, batch, chunk_size)
        if labels.shape != labels_ref.shape or not torch.equal(labels, labels_ref):
            raise RuntimeError(f"label mismatch for {name} at batch {batch_idx}")
        logits_by_model[name] = logits
    if utonia is not None:
        model, head, transform = utonia
        raw_scene = current_raw_scene_from_dataset(loader.dataset, batch_idx)
        utonia_batch = transform_utonia_scene(transform, raw_scene)
        logits, labels = forward_utonia_raw_logits(model, head, utonia_batch)
        if labels.shape != labels_ref.shape or not torch.equal(labels, labels_ref):
            raise RuntimeError(f"Utonia label mismatch at batch {batch_idx}: {labels.shape} vs {labels_ref.shape}")
        logits_by_model["Utonia"] = logits
    scene_name = scene_name_from_dataset(loader.dataset, batch_idx)
    for name, cache_dir in cached_experts:
        logits_by_model[name] = load_cached_expert_logits(cache_dir, scene_name, labels_ref)
    return logits_by_model, labels_ref


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    repo_root = args.repo_root.resolve()
    out_dir = resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_prefix = resolve(repo_root, args.summary_prefix)
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    weak_ids = parse_weak_ids(args.weak_classes)
    num_classes = len(SCANNET20_CLASS_NAMES)
    loader, current_models, utonia, cached_experts, names = load_models(args, repo_root)
    generator = torch.Generator().manual_seed(args.seed)

    fold_x: list[list[torch.Tensor]] = [[], []]
    fold_y: list[list[torch.Tensor]] = [[], []]
    fold_router_y: list[list[torch.Tensor]] = [[], []]
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            logits_by_model, labels = scene_logits(loader, current_models, utonia, cached_experts, batch_idx, batch, args.full_scene_chunk_size)
            feats = fusion_features(logits_by_model, names)
            route_y = oracle_expert_target(logits_by_model, names, labels)
            scene_fold = batch_idx % 2
            train_for_eval_fold = 1 - scene_fold
            xs, ys, rs = sample_scene_features(feats, labels, route_y, args.sample_points_per_scene, generator)
            fold_x[train_for_eval_fold].append(xs)
            fold_y[train_for_eval_fold].append(ys)
            fold_router_y[train_for_eval_fold].append(rs)
            if (batch_idx + 1) % 50 == 0:
                print(f"[collect] scenes={batch_idx + 1}/{len(loader.dataset)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_powers = parse_float_list(args.class_weight_powers)
    stackers: dict[float, list[torch.nn.Module]] = {power: [] for power in weight_powers}
    routers: list[torch.nn.Module] = []
    train_sizes = []
    for fold in range(2):
        x = torch.cat(fold_x[fold], dim=0)
        y = torch.cat(fold_y[fold], dim=0)
        r = torch.cat(fold_router_y[fold], dim=0)
        train_sizes.append(int(y.numel()))
        print(f"[train] eval_fold={fold} train_points={y.numel()} dim={x.shape[1]}", flush=True)
        binc = torch.bincount(r, minlength=len(names)).tolist()
        print(f"[train] eval_fold={fold} oracle_router_targets={dict(zip(names, binc))}", flush=True)
        for power in weight_powers:
            print(f"[train] eval_fold={fold} class_weight_power={power}", flush=True)
            stackers[power].append(
                train_stacker(
                    x,
                    y,
                    num_classes,
                    args.stacker_epochs,
                    args.stacker_batch_size,
                    args.stacker_lr,
                    args.stacker_weight_decay,
                    power,
                    device,
                )
            )
        print(f"[train] eval_fold={fold} oracle-router", flush=True)
        routers.append(
            train_router(
                x,
                r,
                len(names),
                args.stacker_epochs,
                args.stacker_batch_size,
                args.stacker_lr,
                args.stacker_weight_decay,
                device,
            )
        )

    confs = {
        "avgprob_all": torch.zeros((num_classes, num_classes), dtype=torch.long),
        f"single::{names[0]}": torch.zeros((num_classes, num_classes), dtype=torch.long),
    }
    for power in weight_powers:
        confs[f"cv_linear_stacker_w{power:g}"] = torch.zeros((num_classes, num_classes), dtype=torch.long)
    confs["cv_oracle_router_soft"] = torch.zeros((num_classes, num_classes), dtype=torch.long)
    confs["cv_oracle_router_hard"] = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            logits_by_model, labels = scene_logits(loader, current_models, utonia, cached_experts, batch_idx, batch, args.full_scene_chunk_size)
            feats = fusion_features(logits_by_model, names)
            fold = batch_idx % 2
            for power in weight_powers:
                logits = stackers[power][fold](feats.to(device)).cpu()
                update_confusion(confs[f"cv_linear_stacker_w{power:g}"], logits.argmax(dim=1), labels, num_classes, -1)
            route_logits = routers[fold](feats.to(device)).cpu()
            route_w = torch.softmax(route_logits, dim=1)
            probs_stack = stacked_probs(logits_by_model, names)
            soft_probs = (route_w[:, :, None] * probs_stack).sum(dim=1)
            update_confusion(confs["cv_oracle_router_soft"], soft_probs.argmax(dim=1), labels, num_classes, -1)
            hard_choice = route_logits.argmax(dim=1)
            expert_preds = probs_stack.argmax(dim=2)
            hard_pred = expert_preds.gather(1, hard_choice[:, None]).squeeze(1)
            update_confusion(confs["cv_oracle_router_hard"], hard_pred, labels, num_classes, -1)
            avg = torch.stack([softmax_logits(logits_by_model[name]) for name in names], dim=0).mean(dim=0)
            update_confusion(confs["avgprob_all"], avg.argmax(dim=1), labels, num_classes, -1)
            update_confusion(confs[f"single::{names[0]}"], logits_by_model[names[0]].argmax(dim=1), labels, num_classes, -1)
            if (batch_idx + 1) % 50 == 0:
                key = f"cv_linear_stacker_w{weight_powers[0]:g}"
                s = summarize_confusion(confs[key].numpy(), SCANNET20_CLASS_NAMES)
                print(f"[eval] scenes={batch_idx + 1}/{len(loader.dataset)} {key}_mIoU={s['mIoU']:.4f}", flush=True)

    rows = []
    for variant, conf_t in confs.items():
        conf = conf_t.numpy()
        s = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
        rows.append(
            {
                "variant": variant,
                "mIoU": s["mIoU"],
                "mAcc": s["mAcc"],
                "allAcc": s["allAcc"],
                "weak_mean_iou": weak_mean(s, weak_ids),
                "picture_iou": float(s["iou"][NAME_TO_ID["picture"]]),
                "wall_iou": float(s["iou"][NAME_TO_ID["wall"]]),
                "counter_iou": float(s["iou"][NAME_TO_ID["counter"]]),
                "cabinet_iou": float(s["iou"][NAME_TO_ID["cabinet"]]),
                "door_iou": float(s["iou"][NAME_TO_ID["door"]]),
                "picture_to_wall": picture_to_wall(conf),
            }
        )
    write_csv(out_dir / "cross_model_cv_stacker_summary.csv", rows)
    write_csv(summary_prefix.with_suffix(".csv"), rows)

    rows_sorted = sorted(rows, key=lambda r: float(r["mIoU"]), reverse=True)
    md = [
        "# Cross-Model CV Stacker Pilot",
        "",
        "Two-fold scene-level validation pilot. This intentionally uses ScanNet val labels to test whether a learned logit/probability stacker can extract the cross-model oracle headroom; it is not a final train-split baseline.",
        "",
        f"- models: {', '.join(names)}",
        f"- sampled train points per fold: {train_sizes}",
        f"- sample points per scene: `{args.sample_points_per_scene}`",
        f"- epochs: `{args.stacker_epochs}`",
        f"- class weight powers: `{args.class_weight_powers}`",
        "",
        "| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows_sorted, 1):
        md.append(
            f"| {rank} | `{row['variant']}` | `{row['mIoU']:.4f}` | `{row['allAcc']:.4f}` | "
            f"`{row['weak_mean_iou']:.4f}` | `{row['picture_iou']:.4f}` | `{row['picture_to_wall']:.4f}` | "
            f"`{row['counter_iou']:.4f}` | `{row['cabinet_iou']:.4f}` | `{row['door_iou']:.4f}` |"
        )
    md.extend(
        [
            "",
            "## Interpretation Gate",
            "",
            "- If this CV stacker is clearly above simple averaging, learned fusion is worth promoting to a train-split method.",
            "- If it is near or below simple averaging, the high oracle complementarity is not easily actionable even with a learned logit stacker.",
        ]
    )
    summary_prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "cross_model_cv_stacker_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "models": names,
                "train_sizes": train_sizes,
                "sample_points_per_scene": args.sample_points_per_scene,
                "epochs": args.stacker_epochs,
                "max_val_batches": args.max_val_batches,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {summary_prefix.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
