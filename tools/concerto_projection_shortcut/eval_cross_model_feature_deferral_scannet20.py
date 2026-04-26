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
from tools.concerto_projection_shortcut.eval_cross_model_fusion_scannet20 import (  # noqa: E402
    CurrentModelSpec,
    build_utonia_scene_transform,
    current_raw_scene_from_dataset,
    load_cached_expert_logits,
    parse_cached_experts,
    resolve,
    scene_name_from_dataset,
    transform_utonia_scene,
    write_csv,
)
from tools.concerto_projection_shortcut.eval_cross_model_deferral_scannet20 import (  # noqa: E402
    build_rows_from_conf,
    confusion_from_sample,
    pair_features,
    pr_metrics,
    sample_valid_indices,
    train_binary_predictor,
    update_abcd,
)
from tools.concerto_projection_shortcut.eval_masking_battery import build_loader, inference_batch
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import build_model, load_config, move_to_cuda
from tools.concerto_projection_shortcut.eval_utonia_scannet_support_stress import (  # noqa: E402
    build_model as build_utonia_model,
)


NUM_CLASSES = len(SCANNET20_CLASS_NAMES)


@dataclass
class SampleBundle:
    fold: int
    labels: torch.Tensor
    default_pred: torch.Tensor
    aux_pred: dict[str, torch.Tensor]
    logit_features: dict[str, torch.Tensor]
    feature_features: dict[str, torch.Tensor]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Feature-level selective-deferral diagnostic for ScanNet20. "
            "Concerto full-FT is the default expert; auxiliary experts are tested "
            "for whether raw features improve prediction of default-wrong / aux-correct points."
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
    parser.add_argument("--sample-points-per-scene", type=int, default=2048)
    parser.add_argument("--feature-proj-dim", type=int, default=64)
    parser.add_argument("--predictor-epochs", type=int, default=30)
    parser.add_argument("--predictor-batch-size", type=int, default=8192)
    parser.add_argument("--predictor-lr", type=float, default=0.03)
    parser.add_argument("--predictor-weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision-targets", default="0.8,0.9,0.95")
    parser.add_argument(
        "--default-current-model",
        default=(
            "Concerto fullFT::"
            "data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/config.py::"
            "data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_best.pth"
        ),
        help="name::config::weight for the default current-repo model.",
    )
    parser.add_argument(
        "--aux-current-model",
        action="append",
        default=[],
        help="Repeated spec name::config::weight for current-repo auxiliary models.",
    )
    parser.add_argument("--include-utonia", action="store_true")
    parser.add_argument("--utonia-weight", type=Path, default=Path("data/weights/utonia/utonia.pth"))
    parser.add_argument("--utonia-head", type=Path, default=Path("data/weights/utonia/utonia_linear_prob_head_sc.pth"))
    parser.add_argument("--disable-utonia-flash", action="store_true")
    parser.add_argument("--cached-expert", action="append", default=[], help="Repeated probability-only auxiliary spec name::cache_dir.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_model_spec(raw: str) -> CurrentModelSpec:
    parts = raw.split("::")
    if len(parts) != 3:
        raise ValueError(f"invalid model spec: {raw}")
    return CurrentModelSpec(parts[0], Path(parts[1]), Path(parts[2]))


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


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=1)


@torch.no_grad()
def forward_current_raw_logits_features(model, batch: dict, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del chunk_size  # current configs expose exact inverse; kept for CLI compatibility.
    model_input = inference_batch(batch)
    out = model(model_input, return_point=True)
    logits = out["seg_logits"].float()
    point = out.get("point")
    if point is None or not hasattr(point, "feat"):
        feat = logits.float()
    else:
        feat = point.feat.float()
    labels = batch["origin_segment"].long()
    if "inverse" in batch:
        inverse = batch["inverse"].long()
        raw_logits = logits[inverse]
        raw_feat = feat[inverse]
    else:
        raise RuntimeError("current model batch lacks inverse; feature-level deferral requires raw alignment")
    if raw_logits.shape[0] != labels.shape[0] or raw_feat.shape[0] != labels.shape[0]:
        raise RuntimeError(f"raw shape mismatch logits={raw_logits.shape} feat={raw_feat.shape} labels={labels.shape}")
    return raw_logits.cpu(), raw_feat.cpu(), labels.cpu()


@torch.no_grad()
def forward_utonia_raw_logits_features(model, seg_head, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    skip = {"segment", "raw_segment", "raw_coord", "scene_name"}
    for key, value in batch.items():
        if key not in skip and isinstance(value, torch.Tensor):
            batch[key] = value.cuda(non_blocking=True)
    model_input = {key: value for key, value in batch.items() if key not in skip}
    out = model(model_input)
    while "pooling_parent" in out.keys():
        parent = out.pop("pooling_parent")
        inverse = out.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
        out = parent
    feat = out.feat.float().cpu()
    logits = seg_head(out.feat.float()).float().cpu()
    inverse = batch["inverse"].long().cpu()
    raw_feat = feat[inverse]
    raw_logits = logits[inverse]
    raw_labels = batch["raw_segment"].long().cpu()
    if raw_logits.shape[0] != raw_labels.shape[0] or raw_feat.shape[0] != raw_labels.shape[0]:
        raise RuntimeError(f"Utonia raw mismatch logits={raw_logits.shape} feat={raw_feat.shape} labels={raw_labels.shape}")
    return raw_logits, raw_feat, raw_labels


def make_projection(dim: int, out_dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + dim * 1009 + out_dim * 917)
    mat = torch.randn(dim, out_dim, generator=gen) / max(dim, 1) ** 0.5
    return mat.float()


def project_feature(feat: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    feat = feat.float()
    feat = (feat - feat.mean(dim=0, keepdim=True)) / feat.std(dim=0, keepdim=True).clamp_min(1e-6)
    return feat @ proj


def build_feature_pair(
    default_feat: torch.Tensor,
    aux_feat: torch.Tensor | None,
    projections: dict[str, torch.Tensor],
    default_key: str,
    aux_key: str,
    proj_dim: int,
    seed: int,
) -> torch.Tensor | None:
    if aux_feat is None:
        return None
    if default_key not in projections:
        projections[default_key] = make_projection(default_feat.shape[1], proj_dim, seed)
    if aux_key not in projections:
        projections[aux_key] = make_projection(aux_feat.shape[1], proj_dim, seed + 17)
    d = project_feature(default_feat, projections[default_key])
    a = project_feature(aux_feat, projections[aux_key])
    return torch.cat([d, a, a - d, (a - d).abs()], dim=1).cpu()


def train_eval_feature_predictor(
    train_b: list[SampleBundle],
    eval_b: list[SampleBundle],
    aux: str,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    precision_targets: list[float],
) -> tuple[dict, torch.nn.Module, dict[float, float]]:
    feat_key = "logit_features" if mode == "logit" else "feature_features"
    x_train_parts = [getattr(b, feat_key)[aux] for b in train_b if aux in getattr(b, feat_key)]
    x_eval_parts = [getattr(b, feat_key)[aux] for b in eval_b if aux in getattr(b, feat_key)]
    if not x_train_parts or not x_eval_parts:
        raise KeyError(f"missing {mode} features for {aux}")
    x_train = torch.cat(x_train_parts, dim=0)
    x_eval = torch.cat(x_eval_parts, dim=0)
    y_train = torch.cat([((b.default_pred != b.labels) & (b.aux_pred[aux] == b.labels)).long() for b in train_b if aux in getattr(b, feat_key)], dim=0)
    y_eval = torch.cat([((b.default_pred != b.labels) & (b.aux_pred[aux] == b.labels)).long() for b in eval_b if aux in getattr(b, feat_key)], dim=0)
    model = train_binary_predictor(x_train, y_train, args, device)
    with torch.inference_mode():
        train_scores = torch.sigmoid(model(x_train.to(device)).squeeze(1)).cpu()
        eval_scores = torch.sigmoid(model(x_eval.to(device)).squeeze(1)).cpu()
    metrics = pr_metrics(eval_scores, y_eval, precision_targets)
    order = torch.argsort(train_scores, descending=True)
    y_sorted = y_train.bool()[order].float()
    tp = torch.cumsum(y_sorted, dim=0)
    precision = tp / torch.arange(1, y_sorted.numel() + 1).float()
    thresholds = {}
    for target in precision_targets:
        good = torch.where(precision >= target)[0]
        thresholds[target] = float(train_scores[order[good[-1]]].item()) if good.numel() else 1.1
    return metrics, model, thresholds


def write_markdown(path: Path, rows: list[dict], router_rows: list[dict]) -> None:
    lines = [
        "# Feature-Level Selective Deferral Diagnostic",
        "",
        "Default expert is Concerto full-FT. `logit` uses probability/confidence features only; `feature` adds fixed random projections of raw point features from the default and auxiliary experts.",
        "",
        "This is a two-fold scene-level validation diagnostic, not a publishable train-split method result.",
        "",
        "## Deferral Predictability",
        "",
        "| mode | expert | fold | pos rate | PR-AUC | R@P80 | R@P90 | R@P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['mode']}` | `{row['expert']}` | `{row['fold']}` | `{row['pos_rate']:.4f}` | "
            f"`{row['pr_auc']:.4f}` | `{row.get('recall_at_p80', 0.0):.4f}` | "
            f"`{row.get('recall_at_p90', 0.0):.4f}` | `{row.get('recall_at_p95', 0.0):.4f}` |"
        )
    lines.extend(["", "## Sample Conservative Router", "", "| variant | sample mIoU | allAcc | picture | p->wall |", "|---|---:|---:|---:|---:|"])
    for row in router_rows:
        lines.append(
            f"| `{row['variant']}` | `{row['sample_mIoU']:.4f}` | `{row['sample_allAcc']:.4f}` | "
            f"`{row['picture_iou']:.4f}` | `{row['picture_to_wall']:.4f}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    repo_root = args.repo_root.resolve()
    out_dir = resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_prefix = resolve(repo_root, args.summary_prefix)
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    precision_targets = parse_float_list(args.precision_targets)
    generator = torch.Generator().manual_seed(args.seed)

    default_spec = parse_model_spec(args.default_current_model)
    aux_specs = default_aux_specs(args)
    cached_aux = parse_cached_experts(args.cached_expert, repo_root)

    cfg0 = load_config(resolve(repo_root, default_spec.config))
    loader = build_loader(cfg0, args.val_split, resolve(repo_root, args.data_root), args.batch_size, args.num_worker)

    default_cfg = load_config(resolve(repo_root, default_spec.config))
    default_model = build_model(default_cfg, resolve(repo_root, default_spec.weight)).cuda().eval()
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

    aux_names = [name for name, _ in current_aux] + (["Utonia"] if utonia_model is not None else []) + [name for name, _ in cached_aux]
    counts: dict[str, torch.Tensor] = {}
    bundles: list[SampleBundle] = []
    projections: dict[str, torch.Tensor] = {}

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            scene_name = scene_name_from_dataset(loader.dataset, batch_idx)
            default_logits, default_feat, labels = forward_current_raw_logits_features(default_model, batch, args.full_scene_chunk_size)
            default_probs = probs_from_logits(default_logits)
            default_pred = default_probs.argmax(dim=1)
            default_correct = default_pred == labels
            logits_by_aux: dict[str, torch.Tensor] = {}
            feat_by_aux: dict[str, torch.Tensor | None] = {}

            for name, model in current_aux:
                logits, feat, got_labels = forward_current_raw_logits_features(model, batch, args.full_scene_chunk_size)
                if not torch.equal(got_labels, labels):
                    raise RuntimeError(f"label mismatch for {name} scene={scene_name}")
                logits_by_aux[name] = logits
                feat_by_aux[name] = feat

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

            idx = sample_valid_indices(labels, args.sample_points_per_scene, generator)
            bundle = SampleBundle(
                fold=batch_idx % 2,
                labels=labels[idx].cpu(),
                default_pred=default_pred[idx].cpu(),
                aux_pred={},
                logit_features={},
                feature_features={},
            )
            for aux in aux_names:
                aux_probs = probs_from_logits(logits_by_aux[aux])
                aux_pred = aux_probs.argmax(dim=1)
                aux_correct = aux_pred == labels
                update_abcd(counts, aux, labels, default_correct, aux_correct)
                bundle.aux_pred[aux] = aux_pred[idx].cpu()
                bundle.logit_features[aux] = pair_features(default_probs[idx], aux_probs[idx])
                fpair = build_feature_pair(
                    default_feat[idx],
                    feat_by_aux[aux][idx] if feat_by_aux[aux] is not None else None,
                    projections,
                    f"{default_spec.name}::feat",
                    f"{aux}::feat",
                    args.feature_proj_dim,
                    args.seed,
                )
                if fpair is not None:
                    bundle.feature_features[aux] = torch.cat([bundle.logit_features[aux], fpair], dim=1)
            bundles.append(bundle)
            if (batch_idx + 1) % 25 == 0:
                print(f"[collect] scenes={batch_idx + 1}/{len(loader.dataset)}", flush=True)

    recover_rows = []
    for aux, arr in counts.items():
        for class_id, class_name in enumerate(SCANNET20_CLASS_NAMES):
            vals = arr[class_id].float()
            total = float(vals.sum().item())
            if total <= 0:
                continue
            a, b, c, d = [float(v.item()) for v in vals]
            recover_rows.append(
                {
                    "expert": aux,
                    "class": class_name,
                    "A": int(a),
                    "B": int(b),
                    "C": int(c),
                    "D": int(d),
                    "A_frac": a / total,
                    "B_frac": b / total,
                    "C_frac": c / total,
                    "D_frac": d / total,
                    "B_over_A": b / max(a, 1.0),
                }
            )

    pred_rows = []
    router_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for eval_fold in (0, 1):
        train_b = [b for b in bundles if b.fold != eval_fold]
        eval_b = [b for b in bundles if b.fold == eval_fold]
        mode_models: dict[tuple[str, str], torch.nn.Module] = {}
        mode_thresholds: dict[tuple[str, str, float], float] = {}
        modes_by_aux = {aux: ["logit"] + (["feature"] if all(aux in b.feature_features for b in bundles) else []) for aux in aux_names}

        for aux in aux_names:
            for mode in modes_by_aux[aux]:
                metrics, model, thresholds = train_eval_feature_predictor(
                    train_b, eval_b, aux, mode, args, device, precision_targets
                )
                pred_rows.append({"mode": mode, "expert": aux, "fold": eval_fold, **metrics})
                mode_models[(mode, aux)] = model
                for target, threshold in thresholds.items():
                    mode_thresholds[(mode, aux, target)] = threshold

        labels_eval = torch.cat([b.labels for b in eval_b], dim=0)
        default_pred_eval = torch.cat([b.default_pred for b in eval_b], dim=0)
        pred_by_aux = {aux: torch.cat([b.aux_pred[aux] for b in eval_b], dim=0) for aux in aux_names}
        router_rows.append(build_rows_from_conf(f"fold{eval_fold}_default::{default_spec.name}", confusion_from_sample(default_pred_eval, labels_eval)))

        for mode in ("logit", "feature"):
            eligible_aux = [aux for aux in aux_names if (mode, aux) in mode_models]
            if not eligible_aux:
                continue
            score_by_aux = {}
            for aux in eligible_aux:
                feat_key = "logit_features" if mode == "logit" else "feature_features"
                x_eval = torch.cat([getattr(b, feat_key)[aux] for b in eval_b], dim=0)
                with torch.inference_mode():
                    score_by_aux[aux] = torch.sigmoid(mode_models[(mode, aux)](x_eval.to(device)).squeeze(1)).cpu()
            for target in precision_targets:
                pred = default_pred_eval.clone()
                best_score = torch.zeros_like(labels_eval, dtype=torch.float32)
                best_aux = torch.full_like(labels_eval, -1)
                for aux_idx, aux in enumerate(eligible_aux):
                    eligible = score_by_aux[aux] >= mode_thresholds[(mode, aux, target)]
                    better = eligible & (score_by_aux[aux] > best_score)
                    best_score[better] = score_by_aux[aux][better]
                    best_aux[better] = aux_idx
                for aux_idx, aux in enumerate(eligible_aux):
                    mask = best_aux == aux_idx
                    pred[mask] = pred_by_aux[aux][mask]
                router_rows.append(
                    build_rows_from_conf(
                        f"fold{eval_fold}_{mode}_defer_p{int(target * 100)}",
                        confusion_from_sample(pred, labels_eval),
                    )
                )

    write_csv(out_dir / "feature_deferral_recoverability.csv", recover_rows)
    write_csv(out_dir / "feature_deferral_predictability.csv", pred_rows)
    write_csv(out_dir / "feature_conservative_router.csv", router_rows)
    write_csv(summary_prefix.with_name(summary_prefix.name + "_recoverability").with_suffix(".csv"), recover_rows)
    write_csv(summary_prefix.with_name(summary_prefix.name + "_predictability").with_suffix(".csv"), pred_rows)
    write_csv(summary_prefix.with_name(summary_prefix.name + "_router").with_suffix(".csv"), router_rows)
    write_markdown(summary_prefix.with_suffix(".md"), pred_rows, router_rows)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "default_current_model": default_spec.__dict__ | {"config": str(default_spec.config), "weight": str(default_spec.weight)},
                "aux_current_models": [s.__dict__ | {"config": str(s.config), "weight": str(s.weight)} for s in aux_specs],
                "include_utonia": args.include_utonia,
                "cached_experts": [(name, str(path)) for name, path in cached_aux],
                "sample_points_per_scene": args.sample_points_per_scene,
                "feature_proj_dim": args.feature_proj_dim,
                "predictor_epochs": args.predictor_epochs,
                "note": "Two-fold scene-level validation diagnostic; not a train-split publishable method baseline.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {summary_prefix.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
