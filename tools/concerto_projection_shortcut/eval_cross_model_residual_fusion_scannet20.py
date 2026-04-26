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
from tools.concerto_projection_shortcut.eval_cross_model_deferral_scannet20 import pair_features, sample_valid_indices  # noqa: E402
from tools.concerto_projection_shortcut.eval_cross_model_feature_deferral_scannet20 import (  # noqa: E402
    build_feature_pair,
    forward_current_raw_logits_features,
    forward_utonia_raw_logits_features,
    parse_model_spec,
    probs_from_logits,
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
class SceneBundle:
    fold: int
    labels: torch.Tensor
    default_logits: torch.Tensor
    probs_by_name: dict[str, torch.Tensor]
    input_features: torch.Tensor


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FullFT-centered residual fusion diagnostic for ScanNet20. "
            "This is a two-fold scene-level validation pilot, not a final train-split method result."
        )
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
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--sample-points-per-scene", type=int, default=4096)
    parser.add_argument("--feature-proj-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size-train", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--kl-weights", default="0.0,0.03,0.1")
    parser.add_argument("--safe-ce-weights", default="1.0,2.0,4.0")
    parser.add_argument("--delta-l2", type=float, default=1e-4)
    parser.add_argument(
        "--default-current-model",
        default=(
            "Concerto fullFT::"
            "data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/config.py::"
            "data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_best.pth"
        ),
        help="name::config::weight for the default full-FT expert.",
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


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_weak_ids(text: str) -> list[int]:
    name_to_id = {name: idx for idx, name in enumerate(SCANNET20_CLASS_NAMES)}
    return [name_to_id[x.strip()] for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_aux_specs(args: argparse.Namespace) -> list[CurrentModelSpec]:
    if args.aux_current_model:
        return [parse_model_spec(x) for x in args.aux_current_model]
    return [
        CurrentModelSpec(
            "Concerto decoder",
            Path("configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py"),
            Path("data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth"),
        ),
        CurrentModelSpec(
            "Sonata linear",
            Path("configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py"),
            Path("data/weights/sonata/sonata_scannet_linear_merged.pth"),
        ),
    ]


class ResidualFusion(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, num_classes),
        )
        last = self.net[-1]
        assert isinstance(last, torch.nn.Linear)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_input_features(
    default_logits: torch.Tensor,
    default_feat: torch.Tensor,
    logits_by_aux: dict[str, torch.Tensor],
    feat_by_aux: dict[str, torch.Tensor | None],
    aux_names: list[str],
    projections: dict[str, torch.Tensor],
    proj_dim: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    default_probs = probs_from_logits(default_logits)
    features = [default_probs]
    probs_by_name = {"Concerto fullFT": default_probs}
    for aux in aux_names:
        aux_probs = probs_from_logits(logits_by_aux[aux])
        probs_by_name[aux] = aux_probs
        features.extend([aux_probs, aux_probs - default_probs, pair_features(default_probs, aux_probs)])
        fpair = build_feature_pair(
            default_feat,
            feat_by_aux[aux],
            projections,
            "Concerto fullFT::feat",
            f"{aux}::feat",
            proj_dim,
            seed,
        )
        if fpair is not None:
            features.append(fpair)
    return torch.cat(features, dim=1).float(), probs_by_name


def load_models(args: argparse.Namespace, repo_root: Path):
    default_spec = parse_model_spec(args.default_current_model)
    aux_specs = default_aux_specs(args)
    cfg0 = load_config(resolve(repo_root, default_spec.config))
    loader = build_loader(cfg0, args.val_split, resolve(repo_root, args.data_root), args.batch_size, args.num_worker)
    default_model = build_model(cfg0, resolve(repo_root, default_spec.weight)).cuda().eval()
    current_aux = []
    for spec in aux_specs:
        cfg = load_config(resolve(repo_root, spec.config))
        current_aux.append((spec.name, build_model(cfg, resolve(repo_root, spec.weight)).cuda().eval()))
    utonia_model = utonia_head = utonia_transform = None
    if args.include_utonia:
        utonia_model, utonia_head = build_utonia_model(
            resolve(repo_root, args.utonia_weight),
            resolve(repo_root, args.utonia_head),
            args.disable_utonia_flash,
        )
        utonia_transform = build_utonia_scene_transform()
    cached_aux = parse_cached_experts(args.cached_expert, repo_root)
    aux_names = [name for name, _ in current_aux] + (["Utonia"] if utonia_model is not None else []) + [name for name, _ in cached_aux]
    return loader, default_model, current_aux, (utonia_model, utonia_head, utonia_transform), cached_aux, aux_names


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
) -> SceneBundle:
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
    return SceneBundle(
        fold=batch_idx % 2,
        labels=labels.cpu(),
        default_logits=default_logits.cpu(),
        probs_by_name=probs_by_name,
        input_features=input_features.cpu(),
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
    loader = DataLoader(
        TensorDataset(x.float(), z0.float(), y.long()),
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=0,
    )
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        seen = 0
        for xb, z0b, yb in loader:
            xb = xb.to(device, non_blocking=True)
            z0b = z0b.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            delta = model(xb)
            z = z0b + args.delta_scale * delta
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
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 20 == 0:
            print(
                f"[residual] epoch={epoch + 1}/{args.epochs} kl={kl_weight:g} "
                f"safe={safe_ce_weight:g} loss={total / max(seen, 1):.4f}",
                flush=True,
            )
    model.eval()
    return model


def sample_training_tensors(bundles: list[SceneBundle], max_points: int, generator: torch.Generator):
    xs, zs, ys = [], [], []
    for bundle in bundles:
        idx = sample_valid_indices(bundle.labels, max_points, generator)
        xs.append(bundle.input_features[idx])
        zs.append(bundle.default_logits[idx])
        ys.append(bundle.labels[idx])
    return torch.cat(xs, dim=0), torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def update_eval_conf(model, bundles: list[SceneBundle], conf: torch.Tensor, args: argparse.Namespace, device: torch.device) -> None:
    with torch.inference_mode():
        for bundle in bundles:
            valid = (bundle.labels >= 0) & (bundle.labels < NUM_CLASSES)
            x = bundle.input_features[valid].to(device)
            z0 = bundle.default_logits[valid].to(device)
            labels = bundle.labels[valid]
            delta = model(x).cpu()
            pred = (z0.cpu() + args.delta_scale * delta).argmax(dim=1)
            update_confusion(conf, pred, labels, NUM_CLASSES, -1)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    repo_root = args.repo_root.resolve()
    out_dir = resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_prefix = resolve(repo_root, args.summary_prefix)
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    weak_ids = parse_weak_ids(args.weak_classes)
    loader, default_model, current_aux, utonia_pack, cached_aux, aux_names = load_models(args, repo_root)
    projections: dict[str, torch.Tensor] = {}
    bundles: list[SceneBundle] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            bundles.append(
                collect_scene(
                    args,
                    loader,
                    batch_idx,
                    batch,
                    default_model,
                    current_aux,
                    utonia_pack,
                    cached_aux,
                    aux_names,
                    projections,
                )
            )
            if (batch_idx + 1) % 25 == 0:
                print(f"[collect] scenes={batch_idx + 1}/{len(loader.dataset)}", flush=True)

    confs: dict[str, torch.Tensor] = {
        "single::Concerto fullFT": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long),
        "avgprob_all": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long),
    }
    for bundle in bundles:
        valid = (bundle.labels >= 0) & (bundle.labels < NUM_CLASSES)
        update_confusion(confs["single::Concerto fullFT"], bundle.default_logits[valid].argmax(dim=1), bundle.labels[valid], NUM_CLASSES, -1)
        probs = torch.stack([bundle.probs_by_name["Concerto fullFT"][valid]] + [bundle.probs_by_name[n][valid] for n in aux_names], dim=0)
        update_confusion(confs["avgprob_all"], probs.mean(dim=0).argmax(dim=1), bundle.labels[valid], NUM_CLASSES, -1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    kl_weights = parse_float_list(args.kl_weights)
    safe_weights = parse_float_list(args.safe_ce_weights)
    train_sizes = []
    for eval_fold in (0, 1):
        train_b = [b for b in bundles if b.fold != eval_fold]
        eval_b = [b for b in bundles if b.fold == eval_fold]
        x_train, z_train, y_train = sample_training_tensors(train_b, args.sample_points_per_scene, generator)
        train_sizes.append(int(y_train.numel()))
        print(f"[train] eval_fold={eval_fold} train_points={y_train.numel()} dim={x_train.shape[1]}", flush=True)
        for kl in kl_weights:
            for safe in safe_weights:
                key = f"cv_residual_kl{kl:g}_safe{safe:g}"
                conf = confs.setdefault(key, torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long))
                model = train_residual(x_train, z_train, y_train, args, kl, safe, device)
                update_eval_conf(model, eval_b, conf, args, device)

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
    rows_sorted = sorted(rows, key=lambda r: float(r["mIoU"]), reverse=True)
    write_csv(out_dir / "cross_model_residual_fusion_summary.csv", rows_sorted)
    write_csv(summary_prefix.with_suffix(".csv"), rows_sorted)

    md = [
        "# Cross-Model FullFT-Centered Residual Fusion Diagnostic",
        "",
        "Two-fold scene-level validation pilot. This uses ScanNet val labels only to test whether a fullFT-centered residual fusion decoder is a plausible method direction; it is not a final train-split result.",
        "",
        f"- experts: Concerto fullFT default + {', '.join(aux_names)}",
        f"- sampled train points per fold: {train_sizes}",
        f"- feature projection dim: `{args.feature_proj_dim}`",
        f"- epochs: `{args.epochs}`",
        f"- KL weights: `{args.kl_weights}`",
        f"- safe CE weights: `{args.safe_ce_weights}`",
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
    summary_prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "aux_names": aux_names,
                "train_sizes": train_sizes,
                "sample_points_per_scene": args.sample_points_per_scene,
                "feature_proj_dim": args.feature_proj_dim,
                "epochs": args.epochs,
                "kl_weights": kl_weights,
                "safe_ce_weights": safe_weights,
                "note": "Diagnostic two-fold validation result; not a final train-split method baseline.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {summary_prefix.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
