#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_cross_model_deferral_scannet20 import (  # noqa: E402
    sample_valid_indices,
)
from tools.concerto_projection_shortcut.eval_cross_model_feature_deferral_scannet20 import (  # noqa: E402
    parse_model_spec,
)
from tools.concerto_projection_shortcut.eval_cross_model_fusion_scannet20 import (  # noqa: E402
    CurrentModelSpec,
    build_utonia_scene_transform,
    current_raw_scene_from_dataset,
    load_cached_expert_logits,
    parse_cached_experts,
    picture_to_wall,
    resolve,
    scene_name_from_dataset,
    transform_utonia_scene,
)
from tools.concerto_projection_shortcut.eval_cross_model_residual_fusion_scannet20 import (  # noqa: E402
    ResidualFusion,
    build_input_features,
    default_aux_specs,
    forward_current_raw_logits_features,
    forward_utonia_raw_logits_features,
    parse_float_list,
    parse_weak_ids,
    probs_from_logits,
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
)


NUM_CLASSES = len(SCANNET20_CLASS_NAMES)


@dataclass
class SampleTensors:
    x: torch.Tensor
    z0: torch.Tensor
    y: torch.Tensor


@dataclass
class EvalBundle:
    labels: torch.Tensor
    default_logits: torch.Tensor
    probs_by_name: dict[str, torch.Tensor]
    input_features: torch.Tensor


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-split FullFT-centered residual fusion for ScanNet20. "
            "This trains on ScanNet train scenes, selects hyperparameters on a "
            "held-out subset of train scenes, and evaluates ScanNet val once."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-prefix", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=384)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--heldout-every", type=int, default=5)
    parser.add_argument("--full-scene-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--sample-points-per-train-scene", type=int, default=2048)
    parser.add_argument("--sample-points-per-heldout-scene", type=int, default=4096)
    parser.add_argument("--feature-proj-dim", type=int, default=64)
    parser.add_argument("--no-feature-pairs", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size-train", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--kl-weights", default="0.0,0.03")
    parser.add_argument("--safe-ce-weights", default="2.0,4.0")
    parser.add_argument("--delta-l2", type=float, default=1e-4)
    parser.add_argument(
        "--default-current-model",
        default=(
            "Concerto fullFT::"
            "data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/config.py::"
            "data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_best.pth"
        ),
    )
    parser.add_argument("--aux-current-model", action="append", default=[])
    parser.add_argument("--include-utonia", action="store_true")
    parser.add_argument("--utonia-weight", type=Path, default=Path("data/weights/utonia/utonia.pth"))
    parser.add_argument("--utonia-head", type=Path, default=Path("data/weights/utonia/utonia_linear_prob_head_sc.pth"))
    parser.add_argument("--disable-utonia-flash", action="store_true")
    parser.add_argument("--cached-expert", action="append", default=[], help="Repeated probability-only auxiliary spec name::cache_dir.")
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_loaders(args: argparse.Namespace, repo_root: Path, default_spec: CurrentModelSpec):
    cfg = load_config(resolve(repo_root, default_spec.config))
    data_root = resolve(repo_root, args.data_root)
    train_loader = build_loader(cfg, args.train_split, data_root, args.batch_size, args.num_worker)
    val_loader = build_loader(cfg, args.val_split, data_root, args.batch_size, args.num_worker)
    return train_loader, val_loader


def load_experts(args: argparse.Namespace, repo_root: Path):
    default_spec = parse_model_spec(args.default_current_model)
    default_model = build_model(load_config(resolve(repo_root, default_spec.config)), resolve(repo_root, default_spec.weight)).cuda().eval()
    aux_specs = default_aux_specs(args)
    current_aux = []
    for spec in aux_specs:
        model = build_model(load_config(resolve(repo_root, spec.config)), resolve(repo_root, spec.weight)).cuda().eval()
        current_aux.append((spec.name, model))
    utonia_pack = (None, None, None)
    if args.include_utonia:
        utonia_model, utonia_head = build_utonia_model(
            resolve(repo_root, args.utonia_weight),
            resolve(repo_root, args.utonia_head),
            args.disable_utonia_flash,
        )
        utonia_pack = (utonia_model, utonia_head, build_utonia_scene_transform())
    cached_aux = parse_cached_experts(args.cached_expert, repo_root)
    aux_names = [name for name, _ in current_aux] + (["Utonia"] if utonia_pack[0] is not None else []) + [name for name, _ in cached_aux]
    return default_spec, default_model, current_aux, utonia_pack, cached_aux, aux_names


@torch.no_grad()
def collect_scene(
    args: argparse.Namespace,
    loader,
    batch_idx: int,
    batch: dict,
    default_model,
    current_aux,
    utonia_pack,
    cached_aux,
    aux_names: list[str],
    projections: dict[str, torch.Tensor],
) -> EvalBundle:
    batch = move_to_cuda(batch)
    scene_name = scene_name_from_dataset(loader.dataset, batch_idx)
    default_logits, default_feat, labels = forward_current_raw_logits_features(default_model, batch, args.full_scene_chunk_size)
    logits_by_aux: dict[str, torch.Tensor] = {}
    feat_by_aux: dict[str, torch.Tensor | None] = {}
    for name, model in current_aux:
        logits, feat, got_labels = forward_current_raw_logits_features(model, batch, args.full_scene_chunk_size)
        if not torch.equal(got_labels, labels):
            raise RuntimeError(f"label mismatch for {name} scene={scene_name}")
        logits_by_aux[name] = logits
        feat_by_aux[name] = feat
    utonia_model, utonia_head, utonia_transform = utonia_pack
    if utonia_model is not None:
        raw_scene = current_raw_scene_from_dataset(loader.dataset, batch_idx)
        ubatch = transform_utonia_scene(utonia_transform, raw_scene)
        logits, feat, got_labels = forward_utonia_raw_logits_features(utonia_model, utonia_head, ubatch)
        if not torch.equal(got_labels, labels):
            raise RuntimeError(f"label mismatch for Utonia scene={scene_name}")
        logits_by_aux["Utonia"] = logits
        feat_by_aux["Utonia"] = feat
    for name, cache_dir in cached_aux:
        logits_by_aux[name] = load_cached_expert_logits(cache_dir, scene_name, labels)
        feat_by_aux[name] = None
    if args.no_feature_pairs:
        feat_by_aux = {name: None for name in feat_by_aux}
    input_features, probs_by_name = build_input_features(
        default_logits,
        default_feat,
        logits_by_aux,
        feat_by_aux,
        aux_names,
        projections,
        args.feature_proj_dim,
        args.seed,
    )
    return EvalBundle(
        labels=labels.cpu(),
        default_logits=default_logits.cpu(),
        probs_by_name=probs_by_name,
        input_features=input_features.cpu(),
    )


def sample_from_bundle(bundle: EvalBundle, max_points: int, generator: torch.Generator) -> SampleTensors:
    idx = sample_valid_indices(bundle.labels, max_points, generator)
    return SampleTensors(bundle.input_features[idx], bundle.default_logits[idx], bundle.labels[idx])


def concat_samples(samples: list[SampleTensors]) -> SampleTensors:
    return SampleTensors(
        torch.cat([s.x for s in samples], dim=0),
        torch.cat([s.z0 for s in samples], dim=0),
        torch.cat([s.y for s in samples], dim=0),
    )


def train_residual(
    x: torch.Tensor,
    z0: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
    kl_weight: float,
    safe_ce_weight: float,
    device: torch.device,
) -> torch.nn.Module:
    model = ResidualFusion(x.shape[1], args.hidden_dim, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(x.float(), z0.float(), y.long()), batch_size=args.batch_size_train, shuffle=True, num_workers=0)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        seen = 0
        for xb, z0b, yb in loader:
            xb = xb.to(device, non_blocking=True)
            z0b = z0b.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            delta = model(xb)
            z = z0b + args.delta_scale * torch.tanh(delta)
            ce = F.cross_entropy(z, yb, reduction="none")
            default_correct = z0b.argmax(dim=1) == yb
            weights = torch.where(default_correct, torch.full_like(ce, safe_ce_weight), torch.ones_like(ce))
            loss = (ce * weights).mean()
            if kl_weight > 0:
                loss = loss + kl_weight * F.kl_div(F.log_softmax(z, dim=1), F.softmax(z0b, dim=1), reduction="batchmean")
            if args.delta_l2 > 0:
                loss = loss + args.delta_l2 * delta.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(yb.numel())
            seen += int(yb.numel())
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 10 == 0:
            print(f"[train] epoch={epoch + 1}/{args.epochs} kl={kl_weight:g} safe={safe_ce_weight:g} loss={total / max(seen, 1):.4f}", flush=True)
    model.eval()
    return model


def confusion_for_samples(model: torch.nn.Module, sample: SampleTensors, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    with torch.inference_mode():
        loader = DataLoader(TensorDataset(sample.x.float(), sample.z0.float(), sample.y.long()), batch_size=args.batch_size_train, shuffle=False, num_workers=0)
        for xb, z0b, yb in loader:
            xb = xb.to(device, non_blocking=True)
            z0b = z0b.to(device, non_blocking=True)
            delta = model(xb).cpu()
            pred = (z0b.cpu() + args.delta_scale * torch.tanh(delta)).argmax(dim=1)
            update_confusion(conf, pred, yb, NUM_CLASSES, -1)
    return conf


def update_val_conf(model: torch.nn.Module, bundles: list[EvalBundle], conf: torch.Tensor, args: argparse.Namespace, device: torch.device) -> None:
    with torch.inference_mode():
        for bundle in bundles:
            valid = (bundle.labels >= 0) & (bundle.labels < NUM_CLASSES)
            x = bundle.input_features[valid].to(device)
            z0 = bundle.default_logits[valid].to(device)
            labels = bundle.labels[valid]
            delta = model(x).cpu()
            pred = (z0.cpu() + args.delta_scale * torch.tanh(delta)).argmax(dim=1)
            update_confusion(conf, pred, labels, NUM_CLASSES, -1)


def metrics_row(variant: str, conf_t: torch.Tensor, weak_ids: list[int], phase: str) -> dict:
    conf = conf_t.numpy()
    s = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
    return {
        "phase": phase,
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    repo_root = args.repo_root.resolve()
    out_dir = resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_prefix = resolve(repo_root, args.summary_prefix)
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    weak_ids = parse_weak_ids(args.weak_classes)

    default_spec, default_model, current_aux, utonia_pack, cached_aux, aux_names = load_experts(args, repo_root)
    train_loader, val_loader = build_loaders(args, repo_root, default_spec)
    projections: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(args.seed)
    train_samples: list[SampleTensors] = []
    heldout_samples: list[SampleTensors] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
                break
            bundle = collect_scene(args, train_loader, batch_idx, batch, default_model, current_aux, utonia_pack, cached_aux, aux_names, projections)
            if args.heldout_every > 0 and batch_idx % args.heldout_every == 0:
                heldout_samples.append(sample_from_bundle(bundle, args.sample_points_per_heldout_scene, generator))
            else:
                train_samples.append(sample_from_bundle(bundle, args.sample_points_per_train_scene, generator))
            if (batch_idx + 1) % 25 == 0:
                print(f"[collect-train] scenes={batch_idx + 1} train_samples={len(train_samples)} heldout={len(heldout_samples)}", flush=True)

    train = concat_samples(train_samples)
    heldout = concat_samples(heldout_samples)
    print(f"[data] train_points={train.y.numel()} heldout_points={heldout.y.numel()} dim={train.x.shape[1]}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kl_weights = parse_float_list(args.kl_weights)
    safe_weights = parse_float_list(args.safe_ce_weights)
    heldout_rows = []
    models: dict[str, torch.nn.Module] = {}
    for kl in kl_weights:
        for safe in safe_weights:
            key = f"train_residual_kl{kl:g}_safe{safe:g}"
            model = train_residual(train.x, train.z0, train.y, args, kl, safe, device)
            models[key] = model
            conf = confusion_for_samples(model, heldout, args, device)
            heldout_rows.append(metrics_row(key, conf, weak_ids, "train_heldout"))

    heldout_rows = sorted(heldout_rows, key=lambda r: float(r["mIoU"]), reverse=True)
    best_key = heldout_rows[0]["variant"]
    print(f"[select] best={best_key} heldout_mIoU={heldout_rows[0]['mIoU']:.4f}", flush=True)

    val_bundles: list[EvalBundle] = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            val_bundles.append(collect_scene(args, val_loader, batch_idx, batch, default_model, current_aux, utonia_pack, cached_aux, aux_names, projections))
            if (batch_idx + 1) % 25 == 0:
                print(f"[collect-val] scenes={batch_idx + 1}/{len(val_loader.dataset)}", flush=True)

    val_confs: dict[str, torch.Tensor] = {
        "single::Concerto fullFT": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long),
        "avgprob_all": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long),
        best_key: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long),
    }
    for bundle in val_bundles:
        valid = (bundle.labels >= 0) & (bundle.labels < NUM_CLASSES)
        update_confusion(val_confs["single::Concerto fullFT"], bundle.default_logits[valid].argmax(dim=1), bundle.labels[valid], NUM_CLASSES, -1)
        probs = torch.stack([bundle.probs_by_name["Concerto fullFT"][valid]] + [bundle.probs_by_name[n][valid] for n in aux_names], dim=0)
        update_confusion(val_confs["avgprob_all"], probs.mean(dim=0).argmax(dim=1), bundle.labels[valid], NUM_CLASSES, -1)
    update_val_conf(models[best_key], val_bundles, val_confs[best_key], args, device)

    val_rows = [metrics_row(k, c, weak_ids, "val_final") for k, c in val_confs.items()]
    all_rows = heldout_rows + sorted(val_rows, key=lambda r: float(r["mIoU"]), reverse=True)
    write_csv(out_dir / "train_split_residual_fusion_summary.csv", all_rows)
    write_csv(summary_prefix.with_suffix(".csv"), all_rows)

    md = [
        "# Train-Split FullFT-Centered Residual Fusion",
        "",
        "This is the publishable-protocol pilot: residual fusion is trained on ScanNet train scenes, selected on held-out train scenes, and evaluated once on ScanNet val. It should not be mixed with the earlier val-CV diagnostic.",
        "",
        f"- experts: Concerto fullFT default + {', '.join(aux_names)}",
        f"- max train scenes: `{args.max_train_batches}`",
        f"- train/heldout points: `{train.y.numel()}` / `{heldout.y.numel()}`",
        f"- val scenes: `{len(val_bundles)}`",
        f"- feature pairs: `{not args.no_feature_pairs}`; projection dim `{args.feature_proj_dim}`",
        f"- selected by heldout mIoU: `{best_key}`",
        "",
        "## Heldout Selection",
        "",
        "| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(heldout_rows, 1):
        md.append(
            f"| {rank} | `{row['variant']}` | `{row['mIoU']:.4f}` | `{row['allAcc']:.4f}` | "
            f"`{row['weak_mean_iou']:.4f}` | `{row['picture_iou']:.4f}` | `{row['picture_to_wall']:.4f}` | "
            f"`{row['counter_iou']:.4f}` | `{row['cabinet_iou']:.4f}` | `{row['door_iou']:.4f}` |"
        )
    md.extend(
        [
            "",
            "## Final Val Evaluation",
            "",
            "| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(sorted(val_rows, key=lambda r: float(r["mIoU"]), reverse=True), 1):
        md.append(
            f"| {rank} | `{row['variant']}` | `{row['mIoU']:.4f}` | `{row['allAcc']:.4f}` | "
            f"`{row['weak_mean_iou']:.4f}` | `{row['picture_iou']:.4f}` | `{row['picture_to_wall']:.4f}` | "
            f"`{row['counter_iou']:.4f}` | `{row['cabinet_iou']:.4f}` | `{row['door_iou']:.4f}` |"
        )
    summary_prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "aux_names": aux_names,
                "best_key": best_key,
                "train_points": int(train.y.numel()),
                "heldout_points": int(heldout.y.numel()),
                "val_scenes": len(val_bundles),
                "feature_pairs": not args.no_feature_pairs,
                "kl_weights": kl_weights,
                "safe_weights": safe_weights,
                "note": "train/heldout/val protocol; not val-CV",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {summary_prefix.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
