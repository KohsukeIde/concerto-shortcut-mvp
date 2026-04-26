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
    forward_current_raw_logits,
    load_cached_expert_logits,
    parse_cached_experts,
    parse_current_specs,
    resolve,
    scene_name_from_dataset,
    transform_utonia_scene,
    write_csv,
)
from tools.concerto_projection_shortcut.eval_cross_model_fusion_cv_stacker_scannet20 import stacked_probs  # noqa: E402
from tools.concerto_projection_shortcut.eval_masking_battery import build_loader  # noqa: E402
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    build_model,
    load_config,
    move_to_cuda,
    summarize_confusion,
    update_confusion,
)
from tools.concerto_projection_shortcut.eval_utonia_scannet_support_stress import (  # noqa: E402
    build_model as build_utonia_model,
    forward_raw_logits as forward_utonia_raw_logits,
)


NUM_CLASSES = len(SCANNET20_CLASS_NAMES)


@dataclass
class SampleBundle:
    fold: int
    labels: torch.Tensor
    default_pred: torch.Tensor
    aux_pred: dict[str, torch.Tensor]
    features: dict[str, torch.Tensor]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Selective deferral recoverability/predictability audit. The default "
            "expert is normally Concerto full-FT; auxiliary experts are tested for "
            "whether they recover default mistakes without damaging default-correct points."
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
    parser.add_argument("--predictor-epochs", type=int, default=30)
    parser.add_argument("--predictor-batch-size", type=int, default=8192)
    parser.add_argument("--predictor-lr", type=float, default=0.03)
    parser.add_argument("--predictor-weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision-targets", default="0.8,0.9,0.95")
    parser.add_argument("--default-expert", required=True, help="name::cache_dir")
    parser.add_argument("--current-model", action="append", default=[])
    parser.add_argument("--include-utonia", action="store_true")
    parser.add_argument("--utonia-weight", type=Path, default=Path("data/weights/utonia/utonia.pth"))
    parser.add_argument("--utonia-head", type=Path, default=Path("data/weights/utonia/utonia_linear_prob_head_sc.pth"))
    parser.add_argument("--disable-utonia-flash", action="store_true")
    parser.add_argument("--cached-expert", action="append", default=[], help="Repeated auxiliary spec name::cache_dir.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def default_current_specs(args: argparse.Namespace) -> list[CurrentModelSpec]:
    return parse_current_specs(args)


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=1)


def entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=1)


def margin(probs: torch.Tensor) -> torch.Tensor:
    top2 = probs.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def pair_features(default_probs: torch.Tensor, aux_probs: torch.Tensor) -> torch.Tensor:
    d_conf, d_cls = default_probs.max(dim=1)
    a_conf, a_cls = aux_probs.max(dim=1)
    d_ent = entropy(default_probs)
    a_ent = entropy(aux_probs)
    d_margin = margin(default_probs)
    a_margin = margin(aux_probs)
    agree = (d_cls == a_cls).float()
    d_oh = F.one_hot(d_cls.cpu(), NUM_CLASSES).float()
    a_oh = F.one_hot(a_cls.cpu(), NUM_CLASSES).float()
    scalars = torch.stack(
        [
            d_conf.cpu(),
            a_conf.cpu(),
            d_ent.cpu(),
            a_ent.cpu(),
            d_margin.cpu(),
            a_margin.cpu(),
            (a_conf - d_conf).cpu(),
            (d_ent - a_ent).cpu(),
            (a_margin - d_margin).cpu(),
            agree.cpu(),
        ],
        dim=1,
    )
    return torch.cat([scalars, d_oh, a_oh], dim=1)


def sample_indices(n: int, max_points: int, generator: torch.Generator) -> torch.Tensor:
    if max_points <= 0 or n <= max_points:
        return torch.arange(n)
    return torch.randperm(n, generator=generator)[:max_points]


def sample_valid_indices(labels: torch.Tensor, max_points: int, generator: torch.Generator) -> torch.Tensor:
    valid = torch.where((labels >= 0) & (labels < NUM_CLASSES))[0]
    if max_points <= 0 or valid.numel() <= max_points:
        return valid
    chosen = torch.randperm(valid.numel(), generator=generator)[:max_points]
    return valid[chosen]


def update_abcd(counts: dict[str, torch.Tensor], aux_name: str, labels: torch.Tensor, default_correct: torch.Tensor, aux_correct: torch.Tensor) -> None:
    arr = counts.setdefault(aux_name, torch.zeros((NUM_CLASSES, 4), dtype=torch.long))
    valid = (labels >= 0) & (labels < NUM_CLASSES)
    masks = [
        valid & default_correct & ~aux_correct,  # A: false defer danger
        valid & ~default_correct & aux_correct,  # B: desired defer opportunity
        valid & default_correct & aux_correct,   # C
        valid & ~default_correct & ~aux_correct, # D
    ]
    for idx, mask in enumerate(masks):
        valid_labels = labels[mask].cpu()
        if valid_labels.numel():
            arr[:, idx] += torch.bincount(valid_labels, minlength=NUM_CLASSES)


def train_binary_predictor(x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = torch.nn.Linear(x.shape[1], 1).to(device)
    pos = float(y.sum().item())
    neg = float(y.numel() - y.sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.predictor_lr, weight_decay=args.predictor_weight_decay)
    loader = DataLoader(TensorDataset(x.float(), y.float()), batch_size=args.predictor_batch_size, shuffle=True, num_workers=0)
    model.train()
    for _ in range(args.predictor_epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = F.binary_cross_entropy_with_logits(model(xb).squeeze(1), yb, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def pr_metrics(scores: torch.Tensor, target: torch.Tensor, precision_targets: list[float]) -> dict[str, float]:
    scores = scores.float().cpu()
    target = target.bool().cpu()
    order = torch.argsort(scores, descending=True)
    y = target[order].float()
    tp = torch.cumsum(y, dim=0)
    rank = torch.arange(1, y.numel() + 1).float()
    total_pos = float(y.sum().item())
    if total_pos <= 0:
        out = {"pos_rate": 0.0, "pr_auc": 0.0}
        out.update({f"recall_at_p{int(p * 100)}": 0.0 for p in precision_targets})
        return out
    precision = tp / rank
    recall = tp / total_pos
    # Step-wise area under precision-recall curve.
    recall0 = torch.cat([torch.zeros(1), recall])
    precision0 = torch.cat([precision[:1], precision])
    auc = float(((recall0[1:] - recall0[:-1]) * precision0[1:]).sum().item())
    out = {"pos_rate": total_pos / max(float(y.numel()), 1.0), "pr_auc": auc}
    for p in precision_targets:
        good = precision >= p
        out[f"recall_at_p{int(p * 100)}"] = float(recall[good].max().item()) if bool(good.any()) else 0.0
    return out


def confusion_from_sample(pred: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    update_confusion(conf, pred.cpu(), labels.cpu(), NUM_CLASSES, -1)
    return conf.numpy()


def build_rows_from_conf(prefix: str, conf: np.ndarray) -> dict:
    summary = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    pic_denom = conf[pic].sum()
    return {
        "variant": prefix,
        "sample_mIoU": summary["mIoU"],
        "sample_allAcc": summary["allAcc"],
        "picture_iou": float(summary["iou"][pic]),
        "picture_to_wall": float(conf[pic, wall] / pic_denom) if pic_denom else float("nan"),
    }


def write_markdown(summary_prefix: Path, recover_rows: list[dict], pred_rows: list[dict], router_rows: list[dict]) -> None:
    lines = [
        "# Selective Deferral Recoverability / Predictability",
        "",
        "Default expert is Concerto full-FT. `A` means default correct / auxiliary wrong; `B` means default wrong / auxiliary correct and is the desired deferral opportunity.",
        "",
        "## Recoverability",
        "",
        "| expert | class | A frac | B frac | C frac | D frac | B/A |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in recover_rows:
        lines.append(
            f"| `{row['expert']}` | `{row['class']}` | `{row['A_frac']:.4f}` | `{row['B_frac']:.4f}` | "
            f"`{row['C_frac']:.4f}` | `{row['D_frac']:.4f}` | `{row['B_over_A']:.4f}` |"
        )
    lines.extend(["", "## Deferral Predictability", "", "| expert | fold | pos rate | PR-AUC | R@P80 | R@P90 | R@P95 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in pred_rows:
        lines.append(
            f"| `{row['expert']}` | `{row['fold']}` | `{row['pos_rate']:.4f}` | `{row['pr_auc']:.4f}` | "
            f"`{row['recall_at_p80']:.4f}` | `{row['recall_at_p90']:.4f}` | `{row['recall_at_p95']:.4f}` |"
        )
    lines.extend(["", "## Sample-Level Conservative Router Pilot", "", "| variant | sample mIoU | allAcc | picture | p->wall |", "|---|---:|---:|---:|---:|"])
    for row in router_rows:
        lines.append(
            f"| `{row['variant']}` | `{row['sample_mIoU']:.4f}` | `{row['sample_allAcc']:.4f}` | "
            f"`{row['picture_iou']:.4f}` | `{row['picture_to_wall']:.4f}` |"
        )
    summary_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    default_name, default_dir = parse_cached_experts([args.default_expert], repo_root)[0]
    cached_aux = parse_cached_experts(args.cached_expert, repo_root)
    specs = default_current_specs(args)
    cfg0 = load_config(resolve(repo_root, specs[0].config))
    loader = build_loader(cfg0, args.val_split, resolve(repo_root, args.data_root), args.batch_size, args.num_worker)

    current_models = []
    for spec in specs:
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

    aux_names = [name for name, _ in current_models] + (["Utonia"] if utonia_model is not None else []) + [name for name, _ in cached_aux]
    counts: dict[str, torch.Tensor] = {}
    bundles: list[SampleBundle] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            scene_name = scene_name_from_dataset(loader.dataset, batch_idx)
            labels = batch["origin_segment"].long().cpu()
            logits_by_name: dict[str, torch.Tensor] = {
                default_name: load_cached_expert_logits(default_dir, scene_name, labels)
            }
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
            for name, cache_dir in cached_aux:
                logits_by_name[name] = load_cached_expert_logits(cache_dir, scene_name, labels)

            default_probs = probs_from_logits(logits_by_name[default_name])
            default_pred = default_probs.argmax(dim=1)
            default_correct = default_pred == labels
            idx = sample_valid_indices(labels, args.sample_points_per_scene, generator)
            bundle = SampleBundle(
                fold=batch_idx % 2,
                labels=labels[idx].cpu(),
                default_pred=default_pred[idx].cpu(),
                aux_pred={},
                features={},
            )
            for aux in aux_names:
                aux_probs = probs_from_logits(logits_by_name[aux])
                aux_pred = aux_probs.argmax(dim=1)
                aux_correct = aux_pred == labels
                update_abcd(counts, aux, labels, default_correct, aux_correct)
                bundle.aux_pred[aux] = aux_pred[idx].cpu()
                bundle.features[aux] = pair_features(default_probs[idx], aux_probs[idx])
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
    write_csv(out_dir / "deferral_recoverability.csv", recover_rows)
    write_csv(summary_prefix.with_name(summary_prefix.name + "_recoverability").with_suffix(".csv"), recover_rows)

    pred_rows = []
    router_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for eval_fold in (0, 1):
        train_b = [b for b in bundles if b.fold != eval_fold]
        eval_b = [b for b in bundles if b.fold == eval_fold]
        models = {}
        thresholds: dict[tuple[str, float], float] = {}
        for aux in aux_names:
            x_train = torch.cat([b.features[aux] for b in train_b], dim=0)
            y_train = torch.cat([((b.default_pred != b.labels) & (b.aux_pred[aux] == b.labels)).long() for b in train_b], dim=0)
            x_eval = torch.cat([b.features[aux] for b in eval_b], dim=0)
            y_eval = torch.cat([((b.default_pred != b.labels) & (b.aux_pred[aux] == b.labels)).long() for b in eval_b], dim=0)
            model = train_binary_predictor(x_train, y_train, args, device)
            models[aux] = model
            with torch.inference_mode():
                train_scores = torch.sigmoid(model(x_train.to(device)).squeeze(1)).cpu()
                eval_scores = torch.sigmoid(model(x_eval.to(device)).squeeze(1)).cpu()
            metrics = pr_metrics(eval_scores, y_eval, precision_targets)
            pred_rows.append({"expert": aux, "fold": eval_fold, **metrics})
            # Fold-local thresholds chosen on the train half for diagnostic conservative deferral.
            order = torch.argsort(train_scores, descending=True)
            y_sorted = y_train.bool()[order].float()
            tp = torch.cumsum(y_sorted, dim=0)
            precision = tp / torch.arange(1, y_sorted.numel() + 1).float()
            for target in precision_targets:
                good = torch.where(precision >= target)[0]
                thresholds[(aux, target)] = float(train_scores[order[good[-1]]].item()) if good.numel() else 1.1

        # Sample-level conservative router on eval fold.
        labels_eval = torch.cat([b.labels for b in eval_b], dim=0)
        default_pred_eval = torch.cat([b.default_pred for b in eval_b], dim=0)
        score_by_aux = {}
        pred_by_aux = {}
        for aux in aux_names:
            x_eval = torch.cat([b.features[aux] for b in eval_b], dim=0)
            with torch.inference_mode():
                score_by_aux[aux] = torch.sigmoid(models[aux](x_eval.to(device)).squeeze(1)).cpu()
            pred_by_aux[aux] = torch.cat([b.aux_pred[aux] for b in eval_b], dim=0)
        base_conf = confusion_from_sample(default_pred_eval, labels_eval)
        router_rows.append(build_rows_from_conf(f"fold{eval_fold}_default::{default_name}", base_conf))
        for target in precision_targets:
            pred = default_pred_eval.clone()
            best_score = torch.zeros_like(labels_eval, dtype=torch.float32)
            best_aux = torch.full_like(labels_eval, -1)
            for aux_idx, aux in enumerate(aux_names):
                eligible = score_by_aux[aux] >= thresholds[(aux, target)]
                better = eligible & (score_by_aux[aux] > best_score)
                best_score[better] = score_by_aux[aux][better]
                best_aux[better] = aux_idx
            for aux_idx, aux in enumerate(aux_names):
                mask = best_aux == aux_idx
                pred[mask] = pred_by_aux[aux][mask]
            router_rows.append(build_rows_from_conf(f"fold{eval_fold}_defer_p{int(target * 100)}", confusion_from_sample(pred, labels_eval)))

    write_csv(out_dir / "deferral_predictability.csv", pred_rows)
    write_csv(out_dir / "sample_conservative_router.csv", router_rows)
    write_csv(summary_prefix.with_name(summary_prefix.name + "_predictability").with_suffix(".csv"), pred_rows)
    write_csv(summary_prefix.with_name(summary_prefix.name + "_router").with_suffix(".csv"), router_rows)
    write_markdown(summary_prefix, recover_rows, pred_rows, router_rows)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "default_expert": default_name,
                "aux_experts": aux_names,
                "sample_points_per_scene": args.sample_points_per_scene,
                "predictor_epochs": args.predictor_epochs,
                "precision_targets": precision_targets,
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
