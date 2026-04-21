#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.concerto_projection_shortcut.eval_oracle_actionability_analysis as oracle_utils

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_oracle_actionability_analysis import (
    TrainCache,
    binomial_ci,
    build_neighbor_mask,
    candidate_mask,
    cosine_distance_matrix,
    eval_pair_probe_predictions,
    fit_bias,
    fit_pair_probe,
    parse_float_list,
    parse_int_list,
    summarize_confusion,
    update_confusion,
    write_csv,
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


def parse_names_with_vocab(text: str, names: list[str]) -> list[int]:
    name_to_id = {name: idx for idx, name in enumerate(names)}
    out = []
    for chunk in text.split(","):
        name = chunk.strip()
        if not name:
            continue
        if name not in name_to_id:
            raise KeyError(f"unknown class name: {name}")
        out.append(name_to_id[name])
    if not out:
        raise ValueError("no class names parsed")
    return out


def parse_pairs_with_vocab(text: str, names: list[str]) -> list[tuple[int, int]]:
    name_to_id = {name: idx for idx, name in enumerate(names)}
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        pos_name, neg_name = [x.strip() for x in chunk.split(":", 1)]
        if pos_name not in name_to_id or neg_name not in name_to_id:
            raise KeyError(f"unknown class pair: {chunk}")
        out.append((name_to_id[pos_name], name_to_id[neg_name]))
    if not out:
        raise ValueError("no class pairs parsed")
    return out


def pair_name_with_vocab(pair: tuple[int, int]) -> str:
    return f"{ACTIVE_CLASS_NAMES[pair[0]]}_vs_{ACTIVE_CLASS_NAMES[pair[1]]}".replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle/actionability analysis for official PTv3 v1.5.1 checkpoints "
            "using the official model/transform path on current npy scene roots."
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
        default=Path("tools/concerto_projection_shortcut/results_ptv3_v151_oracle_actionability_analysis"),
    )
    parser.add_argument(
        "--weak-classes",
        default="picture,counter,desk,sink,cabinet,shower curtain,door",
    )
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,door:wall",
    )
    parser.add_argument("--top-ks", default="1,2,3,5,10,20")
    parser.add_argument("--oracle-top-ks", default="2,3,5,10")
    parser.add_argument("--graph-top-ks", default="1,2,3,5")
    parser.add_argument("--prior-alphas", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--max-train-scenes", type=int, default=256)
    parser.add_argument("--max-val-scenes", type=int, default=128)
    parser.add_argument("--max-train-points", type=int, default=600000)
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--max-geometry-per-class", type=int, default=60000)
    parser.add_argument("--pair-probe-steps", type=int, default=800)
    parser.add_argument("--pair-probe-lr", type=float, default=0.05)
    parser.add_argument("--pair-probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--bias-steps", type=int, default=1000)
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-weight-decay", type=float, default=1e-3)
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


def collect_train_cache(args: argparse.Namespace, model, point_cls, transform, num_classes: int) -> TrainCache:
    feats: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    class_counts = {idx: 0 for idx in range(num_classes)}
    total_points = 0
    seen_scenes = 0
    with torch.inference_mode():
        for scene_idx, scene_path in enumerate(scene_paths(args.data_root, args.train_split)):
            if args.max_train_scenes >= 0 and scene_idx >= args.max_train_scenes:
                break
            if total_points >= args.max_train_points:
                break
            batch = move_to_cuda(transform(clone_scene(load_scene(scene_path, args.segment_key))))
            feat, logits, labels = forward_point_features(model, point_cls, batch)
            seen_scenes += 1
            valid = (labels >= 0) & (labels < num_classes)
            keep_indices = []
            for cls in range(num_classes):
                room = args.max_per_class - class_counts[cls]
                if room <= 0:
                    continue
                idx = ((labels == cls) & valid).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                remaining_total = args.max_train_points - total_points - sum(i.numel() for i in keep_indices)
                if remaining_total <= 0:
                    break
                cap = min(room, remaining_total)
                if idx.numel() > cap:
                    idx = idx[:cap]
                keep_indices.append(idx)
                class_counts[cls] += int(idx.numel())
            if keep_indices:
                keep = torch.cat(keep_indices, dim=0)
                feats.append(feat[keep].detach().cpu())
                logits_list.append(logits[keep].detach().cpu())
                labels_list.append(labels[keep].detach().cpu())
                total_points += int(keep.numel())
            if (scene_idx + 1) % 25 == 0:
                counts = " ".join(f"{ACTIVE_CLASS_NAMES[k]}={v}" for k, v in class_counts.items() if v)
                print(f"[collect-train] scene={scene_idx + 1} total={total_points} {counts}", flush=True)
    if not labels_list:
        raise RuntimeError("no train cache collected")
    cache = TrainCache(
        feat=torch.cat(feats, dim=0),
        logits=torch.cat(logits_list, dim=0),
        labels=torch.cat(labels_list, dim=0).long(),
        class_counts={ACTIVE_CLASS_NAMES[k]: int(v) for k, v in class_counts.items()},
        seen_batches=seen_scenes,
    )
    print(f"[collect-train] done points={cache.labels.numel()} seen_scenes={seen_scenes}", flush=True)
    return cache


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.official_root = (repo_root / args.official_root).resolve() if not args.official_root.is_absolute() else args.official_root
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    seed_everything(args.seed)
    config_cls, compose_cls, build_model_fn, point_cls = setup_official_imports(args.official_root)
    cfg = load_config(config_cls, args.official_root / args.config)
    names = class_names_from_cfg(cfg, args.class_names, args.num_classes)
    global ACTIVE_CLASS_NAMES
    ACTIVE_CLASS_NAMES = list(names)
    oracle_utils.pair_name = pair_name_with_vocab
    weak_classes = parse_names_with_vocab(args.weak_classes, names)
    pairs = parse_pairs_with_vocab(args.class_pairs, names)
    top_ks = parse_int_list(args.top_ks)
    oracle_top_ks = parse_int_list(args.oracle_top_ks)
    graph_top_ks = parse_int_list(args.graph_top_ks)
    prior_alphas = parse_float_list(args.prior_alphas)
    num_classes = int(args.num_classes)
    transform = compose_cls(cfg.data.val.transform)
    if args.dry_run:
        scene = scene_paths(args.data_root, args.val_split)[0]
        batch = transform(clone_scene(load_scene(scene, args.segment_key)))
        print(f"[dry] scene={scene.name} keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] weak={[ACTIVE_CLASS_NAMES[i] for i in weak_classes]}", flush=True)
        print(f"[dry] pairs={[pair_name_with_vocab(pair) for pair in pairs]}", flush=True)
        return 0

    model = build_official_model(build_model_fn, cfg, args.weight)
    train_cache = collect_train_cache(args, model, point_cls, transform, num_classes)
    pair_probes = [fit_pair_probe(pair, train_cache, args) for pair in pairs]
    bias_unweighted, bias_hist_unweighted = fit_bias(train_cache.logits, train_cache.labels, num_classes, args, balanced=False)
    bias_balanced, bias_hist_balanced = fit_bias(train_cache.logits, train_cache.labels, num_classes, args, balanced=True)
    train_counts = torch.bincount(train_cache.labels, minlength=num_classes).float().clamp_min(1.0)
    log_prior = (train_counts / train_counts.sum()).log()
    neighbor_mask = build_neighbor_mask(pairs, num_classes).cuda()

    variants: dict[str, torch.Tensor] = {
        "base": torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda"),
        "pair_probe_top2": torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda"),
        "bias_unweighted": torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda"),
        "bias_balanced": torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda"),
    }
    for k in oracle_top_ks:
        variants[f"oracle_top{k}"] = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")
    for k in graph_top_ks:
        variants[f"oracle_graph_top{k}"] = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")
    for alpha in prior_alphas:
        variants[f"prior_alpha{str(alpha).replace('.', 'p')}"] = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")

    hit_counts = {(cls, k): 0 for cls in weak_classes for k in top_ks}
    graph_hit_counts = {(cls, k): 0 for cls in weak_classes for k in graph_top_ks}
    target_counts = {cls: 0 for cls in weak_classes}
    top3_counts = {(cls, pred_cls): 0 for cls in weak_classes for pred_cls in range(num_classes)}
    geom_feats = {cls: [] for cls in range(num_classes)}
    geom_counts = {cls: 0 for cls in range(num_classes)}

    seen_val_scenes = 0
    with torch.inference_mode():
        for scene_idx, scene_path in enumerate(scene_paths(args.data_root, args.val_split)):
            if args.max_val_scenes >= 0 and scene_idx >= args.max_val_scenes:
                break
            batch = move_to_cuda(transform(clone_scene(load_scene(scene_path, args.segment_key))))
            feat_e, logits_e, labels_e = forward_point_features(model, point_cls, batch)
            valid = (labels_e >= 0) & (labels_e < num_classes)
            if not valid.any():
                continue
            feat_e = feat_e[valid]
            logits_e = logits_e[valid]
            labels_e = labels_e[valid]
            base_pred = logits_e.argmax(dim=1)
            update_confusion(variants["base"], base_pred, labels_e, num_classes, -1)
            update_confusion(
                variants["pair_probe_top2"],
                eval_pair_probe_predictions(feat_e, logits_e, pair_probes),
                labels_e,
                num_classes,
                -1,
            )
            update_confusion(variants["bias_unweighted"], (logits_e + bias_unweighted.to(logits_e.device)).argmax(dim=1), labels_e, num_classes, -1)
            update_confusion(variants["bias_balanced"], (logits_e + bias_balanced.to(logits_e.device)).argmax(dim=1), labels_e, num_classes, -1)
            for alpha in prior_alphas:
                pred = (logits_e - alpha * log_prior.to(logits_e.device)).argmax(dim=1)
                update_confusion(variants[f"prior_alpha{str(alpha).replace('.', 'p')}"], pred, labels_e, num_classes, -1)
            for k in oracle_top_ks:
                cand = candidate_mask(logits_e, k)
                pred = torch.where(cand.gather(1, labels_e[:, None]).squeeze(1), labels_e, base_pred)
                update_confusion(variants[f"oracle_top{k}"], pred, labels_e, num_classes, -1)
            for k in graph_top_ks:
                cand = candidate_mask(logits_e, k, neighbor_mask)
                pred = torch.where(cand.gather(1, labels_e[:, None]).squeeze(1), labels_e, base_pred)
                update_confusion(variants[f"oracle_graph_top{k}"], pred, labels_e, num_classes, -1)

            max_k = max(top_ks)
            top_idx = logits_e.topk(k=min(max_k, num_classes), dim=1).indices
            for cls in weak_classes:
                mask = labels_e == cls
                n = int(mask.sum().item())
                if n == 0:
                    continue
                target_counts[cls] += n
                top_cls = top_idx[mask]
                for k in top_ks:
                    hit_counts[(cls, k)] += int((top_cls[:, : min(k, num_classes)] == cls).any(dim=1).sum().item())
                top3 = top_cls[:, : min(3, num_classes)]
                for pred_cls in range(num_classes):
                    top3_counts[(cls, pred_cls)] += int((top3 == pred_cls).any(dim=1).sum().item())
                for k in graph_top_ks:
                    cand = candidate_mask(logits_e[mask], k, neighbor_mask)
                    graph_hit_counts[(cls, k)] += int(cand[:, cls].sum().item())
            for cls in range(num_classes):
                room = args.max_geometry_per_class - geom_counts[cls]
                if room <= 0:
                    continue
                idx = (labels_e == cls).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                if idx.numel() > room:
                    idx = idx[:room]
                geom_feats[cls].append(feat_e[idx].detach().cpu())
                geom_counts[cls] += int(idx.numel())
            seen_val_scenes += 1
            if (scene_idx + 1) % 25 == 0:
                base_summary = summarize_confusion(variants["base"].detach().cpu().numpy(), names)
                print(f"[eval] scene={scene_idx + 1} base_mIoU={base_summary['mIoU']:.4f}", flush=True)

    summaries = {name: summarize_confusion(conf.detach().cpu().numpy(), names) for name, conf in variants.items()}
    base = summaries["base"]

    topk_rows = []
    for cls in weak_classes:
        denom = max(target_counts[cls], 1)
        for k in top_ks:
            count = hit_counts[(cls, k)]
            ci_low, ci_high = binomial_ci(count, denom)
            topk_rows.append(
                {
                    "class_id": cls,
                    "class_name": ACTIVE_CLASS_NAMES[cls],
                    "kind": "topk",
                    "k": k,
                    "hit_count": count,
                    "hit_rate": count / denom,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "target_count": target_counts[cls],
                }
            )
        for k in graph_top_ks:
            count = graph_hit_counts[(cls, k)]
            ci_low, ci_high = binomial_ci(count, denom)
            topk_rows.append(
                {
                    "class_id": cls,
                    "class_name": ACTIVE_CLASS_NAMES[cls],
                    "kind": "topk_plus_confusion_graph",
                    "k": k,
                    "hit_count": count,
                    "hit_rate": count / denom,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "target_count": target_counts[cls],
                }
            )

    variant_rows = []
    for name, summary in summaries.items():
        row = {
            "variant": name,
            "mIoU": summary["mIoU"],
            "mAcc": summary["mAcc"],
            "allAcc": summary["allAcc"],
            "delta_mIoU": summary["mIoU"] - base["mIoU"],
        }
        for cls in weak_classes:
            cname = ACTIVE_CLASS_NAMES[cls].replace(" ", "_")
            row[f"{cname}_iou"] = summary["iou"][cls]
            row[f"{cname}_delta_iou"] = summary["iou"][cls] - base["iou"][cls]
        variant_rows.append(row)

    confusion_rows = []
    base_conf = variants["base"].detach().cpu().numpy()
    for cls in weak_classes:
        denom = max(base_conf[cls].sum(), 1)
        for pred_id in np.argsort(base_conf[cls])[::-1]:
            count = int(base_conf[cls, pred_id])
            if count == 0:
                continue
            confusion_rows.append(
                {
                    "target_id": cls,
                    "target_name": ACTIVE_CLASS_NAMES[cls],
                    "pred_id": int(pred_id),
                    "pred_name": ACTIVE_CLASS_NAMES[pred_id],
                    "count": count,
                    "fraction_of_target": count / denom,
                }
            )

    top3_rows = []
    for cls in weak_classes:
        denom = max(target_counts[cls], 1)
        for pred_id in range(num_classes):
            count = top3_counts[(cls, pred_id)]
            if count == 0:
                continue
            top3_rows.append(
                {
                    "target_id": cls,
                    "target_name": ACTIVE_CLASS_NAMES[cls],
                    "top3_class_id": pred_id,
                    "top3_class_name": ACTIVE_CLASS_NAMES[pred_id],
                    "count": count,
                    "fraction_of_target": count / denom,
                }
            )

    centroid = []
    within_var = []
    template = None
    for cls in range(num_classes):
        if geom_feats[cls]:
            x = torch.cat(geom_feats[cls], dim=0).float()
            template = x[0]
            c = x.mean(dim=0)
            cn = F.normalize(c, dim=0)
            sims = F.normalize(x, dim=1) @ cn
            centroid.append(c)
            within_var.append(float((1.0 - sims).mean().item()))
        else:
            if template is None:
                template = torch.zeros(train_cache.feat.shape[1])
            centroid.append(torch.zeros_like(template))
            within_var.append(float("nan"))
    centroids = torch.stack(centroid, dim=0)
    cos = cosine_distance_matrix(centroids).numpy()
    geometry_rows = []
    for cls in weak_classes:
        order = np.argsort(cos[cls])[::-1]
        rank = 0
        for other in order:
            if other == cls:
                continue
            rank += 1
            geometry_rows.append(
                {
                    "class_id": cls,
                    "class_name": ACTIVE_CLASS_NAMES[cls],
                    "other_id": int(other),
                    "other_name": ACTIVE_CLASS_NAMES[int(other)],
                    "centroid_cosine": float(cos[cls, other]),
                    "rank": rank,
                    "class_within_cos_distance": within_var[cls],
                    "other_within_cos_distance": within_var[int(other)],
                    "class_count": geom_counts[cls],
                    "other_count": geom_counts[int(other)],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "oracle_topk_hit_rates.csv",
        topk_rows,
        ["class_id", "class_name", "kind", "k", "hit_count", "hit_rate", "ci_low", "ci_high", "target_count"],
    )
    variant_fields = ["variant", "mIoU", "mAcc", "allAcc", "delta_mIoU"]
    for cls in weak_classes:
        cname = ACTIVE_CLASS_NAMES[cls].replace(" ", "_")
        variant_fields += [f"{cname}_iou", f"{cname}_delta_iou"]
    write_csv(args.output_dir / "oracle_variants.csv", variant_rows, variant_fields)
    write_csv(
        args.output_dir / "oracle_confusion_distribution.csv",
        confusion_rows,
        ["target_id", "target_name", "pred_id", "pred_name", "count", "fraction_of_target"],
    )
    write_csv(
        args.output_dir / "oracle_top3_distribution.csv",
        top3_rows,
        ["target_id", "target_name", "top3_class_id", "top3_class_name", "count", "fraction_of_target"],
    )
    write_csv(
        args.output_dir / "oracle_feature_geometry.csv",
        geometry_rows,
        [
            "class_id",
            "class_name",
            "other_id",
            "other_name",
            "centroid_cosine",
            "rank",
            "class_within_cos_distance",
            "other_within_cos_distance",
            "class_count",
            "other_count",
        ],
    )
    probe_rows = [
        {
            "pair": pair_name_with_vocab(probe.pair),
            "positive_class": ACTIVE_CLASS_NAMES[probe.pair[0]],
            "negative_class": ACTIVE_CLASS_NAMES[probe.pair[1]],
            "train_bal_acc": probe.train_bal_acc,
        }
        for probe in pair_probes
    ]
    write_csv(args.output_dir / "oracle_pair_probe_train.csv", probe_rows, ["pair", "positive_class", "negative_class", "train_bal_acc"])

    metadata = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "weak_classes": [ACTIVE_CLASS_NAMES[i] for i in weak_classes],
        "pairs": [pair_name_with_vocab(pair) for pair in pairs],
        "train": {
            "seen_batches": train_cache.seen_batches,
            "class_counts": train_cache.class_counts,
            "num_points": int(train_cache.labels.numel()),
        },
        "val": {"seen_batches": seen_val_scenes, "target_counts": {ACTIVE_CLASS_NAMES[k]: int(v) for k, v in target_counts.items()}},
        "bias_history_unweighted": bias_hist_unweighted,
        "bias_history_balanced": bias_hist_balanced,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    name_to_id = {name: idx for idx, name in enumerate(ACTIVE_CLASS_NAMES)}
    picture = name_to_id.get("picture")
    wall = name_to_id.get("wall")
    best_variant = max((r for r in variant_rows if r["variant"] != "base"), key=lambda r: r["mIoU"])
    best_picture = max((r for r in variant_rows if r["variant"] != "base"), key=lambda r: r.get("picture_iou", -1.0))
    lines = [
        "# PTv3 v1.5.1 Oracle Actionability Analysis",
        "",
        "## Setup",
        f"- official root: `{args.official_root}`",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- segment key: `{args.segment_key}`",
        f"- weak classes: `{args.weak_classes}`",
        f"- class pairs: `{args.class_pairs}`",
        f"- train scenes seen: {train_cache.seen_batches}",
        f"- val scenes seen: {seen_val_scenes}",
        "",
        "## Aggregate Variants",
        "",
        "| variant | mIoU | delta mIoU | picture IoU | picture delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(variant_rows, key=lambda r: r["mIoU"], reverse=True)[:12]:
        lines.append(
            "| {variant} | {mIoU:.4f} | {delta_mIoU:+.4f} | {picture_iou:.4f} | {picture_delta_iou:+.4f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Top-K Hit Rates",
        "",
        "| class | kind | K | hit rate | 95% CI | target count |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in topk_rows:
        if row["class_name"] in {"picture", "counter", "desk", "sink", "door"} or row["k"] in {1, 2, 5}:
            lines.append(
                f"| {row['class_name']} | {row['kind']} | {row['k']} | {row['hit_rate']:.4f} | "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['target_count']} |"
            )
    if picture is not None and wall is not None:
        picture_wall_frac = float(base_conf[picture, wall] / max(base_conf[picture].sum(), 1))
    else:
        picture_wall_frac = float("nan")
    lines += [
        "",
        "## Key Readout Headroom",
        "",
        f"- base mIoU: `{base['mIoU']:.4f}`",
        f"- base picture IoU: `{base['iou'][picture]:.4f}`",
        f"- base picture -> wall fraction: `{picture_wall_frac:.4f}`",
        f"- best non-base mIoU variant: `{best_variant['variant']}` "
        f"(`{best_variant['mIoU']:.4f}`, delta `{best_variant['delta_mIoU']:+.4f}`)",
        f"- best non-base picture variant: `{best_picture['variant']}` "
        f"(`{best_picture['picture_iou']:.4f}`, delta `{best_picture['picture_delta_iou']:+.4f}`)",
        "",
        "## Output Files",
        "",
        "- `oracle_topk_hit_rates.csv`",
        "- `oracle_variants.csv`",
        "- `oracle_confusion_distribution.csv`",
        "- `oracle_top3_distribution.csv`",
        "- `oracle_feature_geometry.csv`",
        "- `oracle_pair_probe_train.csv`",
        "- `metadata.json`",
    ]
    md_path = args.output_dir / "oracle_actionability_analysis.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    prefix = args.summary_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    for src_name, dst_name in [
        ("oracle_variants.csv", prefix.with_suffix(".csv").name),
        ("oracle_topk_hit_rates.csv", f"{prefix.name}_topk.csv"),
        ("oracle_confusion_distribution.csv", f"{prefix.name}_confusion.csv"),
        ("oracle_feature_geometry.csv", f"{prefix.name}_geometry.csv"),
        ("oracle_pair_probe_train.csv", f"{prefix.name}_pairprobe.csv"),
    ]:
        prefix.parent.joinpath(dst_name).write_text((args.output_dir / src_name).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[write] {md_path}", flush=True)
    print(f"[best-variant] {best_variant}", flush=True)
    print(f"[best-picture] {best_picture}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
