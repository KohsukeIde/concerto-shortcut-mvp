#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
    parse_pairs,
)
from tools.concerto_projection_shortcut.fit_coda_decoder_adapter import (  # noqa: E402
    build_adapter_from_state,
    build_loader,
    build_model,
    corrected_logits,
    eval_tensors,
    forward_features,
    load_config,
    move_to_cuda,
    summarize_confusion,
    update_confusion,
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal CoDA transfer-failure analysis: compare train/heldout/val "
            "feature statistics, heldout-vs-val overcorrection, and picture "
            "candidate-rank drift for the saved CoDA adapter."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--heldout-mod", type=int, default=5)
    parser.add_argument("--heldout-remainder", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=-1)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument(
        "--classes",
        default="picture,wall,counter,desk,table,sink,cabinet,door,shower curtain",
    )
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall",
    )
    parser.add_argument(
        "--variants",
        default=(
            "heldout_selected=1.0:1.0,"
            "best_miou=0.1:1.0,"
            "best_picture=0.2:1.0,"
            "mid=0.5:0.5"
        ),
        help="Comma-separated NAME=LAMBDA:TAU variants. Base is always included.",
    )
    parser.add_argument("--center-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_names(text: str) -> list[int]:
    ids = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {name}")
        ids.append(NAME_TO_ID[name])
    return ids


def parse_variants(text: str) -> dict[str, tuple[float, float] | None]:
    variants: dict[str, tuple[float, float] | None] = {"base": None}
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, spec = raw.split("=", 1)
        lam_s, tau_s = spec.split(":", 1)
        variants[name.strip()] = (float(lam_s), float(tau_s))
    return variants


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    repo_root = args.repo_root.resolve()
    args.config = str((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.adapter_path = (repo_root / args.adapter_path).resolve() if not args.adapter_path.is_absolute() else args.adapter_path
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def empty_feature_stats(class_ids: list[int]) -> dict[int, dict[str, torch.Tensor | int]]:
    return {
        cls: {
            "count": 0,
            "sum": None,
            "sumsq": None,
        }
        for cls in class_ids
    }


def update_feature_stats(stats: dict[int, dict], feat: torch.Tensor, labels: torch.Tensor, class_ids: list[int]) -> None:
    feat = feat.float()
    for cls in class_ids:
        mask = labels == cls
        count = int(mask.sum().item())
        if count == 0:
            continue
        x = feat[mask].double()
        x_sum = x.sum(dim=0).detach().cpu()
        x_sumsq = (x * x).sum(dim=0).detach().cpu()
        if stats[cls]["sum"] is None:
            stats[cls]["sum"] = torch.zeros_like(x_sum)
            stats[cls]["sumsq"] = torch.zeros_like(x_sumsq)
        stats[cls]["count"] += count
        stats[cls]["sum"] += x_sum
        stats[cls]["sumsq"] += x_sumsq


def feature_summary(stats: dict[int, dict]) -> dict[int, dict[str, torch.Tensor | float | int]]:
    out = {}
    for cls, item in stats.items():
        count = int(item["count"])
        if count <= 0:
            out[cls] = {"count": 0, "mean": None, "var_trace": float("nan")}
            continue
        mean = item["sum"] / count
        second = item["sumsq"] / count
        var = (second - mean * mean).clamp_min(0)
        out[cls] = {
            "count": count,
            "mean": mean.float(),
            "var_trace": float(var.sum().item()),
        }
    return out


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(F.cosine_similarity(a.float()[None], b.float()[None]).item())


def l2(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(torch.linalg.vector_norm(a.float() - b.float()).item())


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in row.items()})


def init_confusions(domains: list[str], variants: dict[str, tuple[float, float] | None], num_classes: int):
    return {
        domain: {
            name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")
            for name in variants
        }
        for domain in domains
    }


def update_variant_confusions(
    confusions: dict[str, dict[str, torch.Tensor]],
    domain: str,
    variants: dict[str, tuple[float, float] | None],
    adapter,
    feat: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    center_delta: bool,
) -> dict[str, torch.Tensor]:
    variant_logits = {}
    for name, spec in variants.items():
        if spec is None:
            current = logits
        else:
            current = corrected_logits(adapter, feat, logits, spec[0], spec[1], center_delta)
        variant_logits[name] = current
        update_confusion(confusions[domain][name], current.argmax(dim=1), labels, num_classes, ignore_index)
    return variant_logits


def empty_rank_stats(variants: dict[str, tuple[float, float] | None], num_classes: int) -> dict[str, dict]:
    return {
        name: {
            "count": 0,
            "gt_rank_sum": 0.0,
            "wall_rank_sum": 0.0,
            "top1": 0,
            "top2": 0,
            "top5": 0,
            "wall_top1": 0,
            "margin_sum": 0.0,
            "margin_sumsq": 0.0,
            "rank_hist": torch.zeros(num_classes, dtype=torch.int64),
            "margins": [],
        }
        for name in variants
    }


def update_picture_rank_stats(rank_stats: dict[str, dict], variant_logits: dict[str, torch.Tensor], labels: torch.Tensor):
    picture = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    mask = labels == picture
    count = int(mask.sum().item())
    if count == 0:
        return
    for name, logits in variant_logits.items():
        z = logits[mask]
        pic_score = z[:, picture]
        wall_score = z[:, wall]
        gt_rank = (z > pic_score[:, None]).sum(dim=1) + 1
        wall_rank = (z > wall_score[:, None]).sum(dim=1) + 1
        pred = z.argmax(dim=1)
        margin = (pic_score - wall_score).detach().float().cpu()
        stats = rank_stats[name]
        stats["count"] += count
        stats["gt_rank_sum"] += float(gt_rank.float().sum().item())
        stats["wall_rank_sum"] += float(wall_rank.float().sum().item())
        stats["top1"] += int((gt_rank <= 1).sum().item())
        stats["top2"] += int((gt_rank <= 2).sum().item())
        stats["top5"] += int((gt_rank <= 5).sum().item())
        stats["wall_top1"] += int((pred == wall).sum().item())
        stats["margin_sum"] += float(margin.sum().item())
        stats["margin_sumsq"] += float((margin * margin).sum().item())
        stats["rank_hist"] += torch.bincount((gt_rank.detach().cpu() - 1).long(), minlength=20)
        stats["margins"].append(margin)


def rank_rows(domain: str, rank_stats: dict[str, dict]) -> list[dict]:
    rows = []
    for name, stats in rank_stats.items():
        count = int(stats["count"])
        if count == 0:
            continue
        margins = torch.cat(stats["margins"]) if stats["margins"] else torch.empty(0)
        margin_mean = stats["margin_sum"] / count
        margin_var = max(stats["margin_sumsq"] / count - margin_mean * margin_mean, 0.0)
        qs = torch.quantile(margins, torch.tensor([0.1, 0.5, 0.9])) if margins.numel() else torch.full((3,), float("nan"))
        rows.append(
            {
                "domain": domain,
                "variant": name,
                "picture_count": count,
                "gt_top1": stats["top1"] / count,
                "gt_top2": stats["top2"] / count,
                "gt_top5": stats["top5"] / count,
                "wall_top1": stats["wall_top1"] / count,
                "mean_gt_rank": stats["gt_rank_sum"] / count,
                "mean_wall_rank": stats["wall_rank_sum"] / count,
                "margin_mean": margin_mean,
                "margin_std": margin_var**0.5,
                "margin_p10": float(qs[0].item()),
                "margin_p50": float(qs[1].item()),
                "margin_p90": float(qs[2].item()),
            }
        )
    return rows


def scan_split(
    args: argparse.Namespace,
    model,
    cfg,
    adapter,
    split: str,
    domains_for_batch,
    class_ids: list[int],
    variants: dict[str, tuple[float, float] | None],
    confusions,
    feature_stats,
    rank_stats_by_domain,
    num_classes: int,
    ignore_index: int,
    max_batches: int,
) -> int:
    loader = build_loader(cfg, split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches >= 0 and batch_idx >= max_batches:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, batch = forward_features(model, batch)
            feat_e, logits_e, labels_e = eval_tensors(feat, logits, labels, batch)
            domains = domains_for_batch(batch_idx)
            for domain in domains:
                update_feature_stats(feature_stats[domain], feat_e, labels_e, class_ids)
                if domain in confusions:
                    variant_logits = update_variant_confusions(
                        confusions,
                        domain,
                        variants,
                        adapter,
                        feat_e,
                        logits_e,
                        labels_e,
                        num_classes,
                        ignore_index,
                        args.center_delta,
                    )
                    update_picture_rank_stats(rank_stats_by_domain[domain], variant_logits, labels_e)
            seen += 1
            if (batch_idx + 1) % 100 == 0:
                print(f"[scan] split={split} batch={batch_idx + 1}", flush=True)
    return seen


def confusion_rows(confusions: dict[str, dict[str, torch.Tensor]], class_ids: list[int], num_classes: int) -> tuple[list[dict], dict]:
    summaries = {}
    rows = []
    for domain, by_variant in confusions.items():
        summaries[domain] = {}
        base_summary = None
        for name, conf_t in by_variant.items():
            summary = summarize_confusion(conf_t.detach().cpu().numpy(), SCANNET20_CLASS_NAMES)
            summaries[domain][name] = summary
            if name == "base":
                base_summary = summary
        assert base_summary is not None
        for name, summary in summaries[domain].items():
            for cls in class_ids:
                rows.append(
                    {
                        "domain": domain,
                        "variant": name,
                        "class": SCANNET20_CLASS_NAMES[cls],
                        "iou": summary["iou"][cls],
                        "delta_iou": summary["iou"][cls] - base_summary["iou"][cls],
                        "acc": summary["acc"][cls],
                        "target_count": summary["target_sum"][cls],
                        "pred_count": summary["pred_sum"][cls],
                    }
                )
    return rows, summaries


def pair_rows(confusions: dict[str, dict[str, torch.Tensor]], pairs: list[tuple[int, int]]) -> list[dict]:
    rows = []
    for domain, by_variant in confusions.items():
        for name, conf_t in by_variant.items():
            conf = conf_t.detach().cpu().numpy()
            target_sum = conf.sum(axis=1)
            for a, b in pairs:
                a_denom = max(float(target_sum[a]), 1.0)
                b_denom = max(float(target_sum[b]), 1.0)
                rows.append(
                    {
                        "domain": domain,
                        "variant": name,
                        "pair": f"{SCANNET20_CLASS_NAMES[a]}:{SCANNET20_CLASS_NAMES[b]}",
                        "a_to_b_frac": float(conf[a, b] / a_denom),
                        "b_to_a_frac": float(conf[b, a] / b_denom),
                        "a_correct_frac": float(conf[a, a] / a_denom),
                        "b_correct_frac": float(conf[b, b] / b_denom),
                    }
                )
    return rows


def feature_shift_rows(feature_stats: dict[str, dict[int, dict]], class_ids: list[int], pairs: list[tuple[int, int]]) -> tuple[list[dict], list[dict]]:
    summaries = {domain: feature_summary(stats) for domain, stats in feature_stats.items()}
    class_rows = []
    for cls in class_ids:
        train = summaries["train"][cls]
        held = summaries["heldout"][cls]
        val = summaries["val"][cls]
        class_rows.append(
            {
                "class": SCANNET20_CLASS_NAMES[cls],
                "train_count": train["count"],
                "heldout_count": held["count"],
                "val_count": val["count"],
                "cos_train_heldout": cosine(train["mean"], held["mean"]),
                "cos_train_val": cosine(train["mean"], val["mean"]),
                "cos_heldout_val": cosine(held["mean"], val["mean"]),
                "l2_train_heldout": l2(train["mean"], held["mean"]),
                "l2_train_val": l2(train["mean"], val["mean"]),
                "l2_heldout_val": l2(held["mean"], val["mean"]),
                "var_trace_train": train["var_trace"],
                "var_trace_heldout": held["var_trace"],
                "var_trace_val": val["var_trace"],
            }
        )
    pair_centroid_rows = []
    for domain, summary in summaries.items():
        for a, b in pairs:
            pair_centroid_rows.append(
                {
                    "domain": domain,
                    "pair": f"{SCANNET20_CLASS_NAMES[a]}:{SCANNET20_CLASS_NAMES[b]}",
                    "centroid_cos": cosine(summary[a]["mean"], summary[b]["mean"]),
                    "centroid_l2": l2(summary[a]["mean"], summary[b]["mean"]),
                    "a_var_trace": summary[a]["var_trace"],
                    "b_var_trace": summary[b]["var_trace"],
                    "a_count": summary[a]["count"],
                    "b_count": summary[b]["count"],
                }
            )
    return class_rows, pair_centroid_rows


def main() -> int:
    args = resolve_paths(parse_args())
    cfg = load_config(Path(args.config))
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    class_ids = parse_names(args.classes)
    pairs = parse_pairs(args.class_pairs)
    variants = parse_variants(args.variants)
    if args.dry_run:
        print(f"[dry] classes={[SCANNET20_CLASS_NAMES[i] for i in class_ids]}", flush=True)
        print(f"[dry] pairs={args.class_pairs}", flush=True)
        print(f"[dry] variants={variants}", flush=True)
        return 0

    model = build_model(cfg, args.weight)
    payload = torch.load(args.adapter_path, map_location="cpu", weights_only=False)
    adapter = build_adapter_from_state(payload.get("state_dict", payload), dropout=0.0).cuda().eval()

    feature_stats = {
        "train": empty_feature_stats(class_ids),
        "heldout": empty_feature_stats(class_ids),
        "val": empty_feature_stats(class_ids),
    }
    confusions = init_confusions(["heldout", "val"], variants, num_classes)
    rank_stats = {domain: empty_rank_stats(variants, num_classes) for domain in ["heldout", "val"]}

    train_seen = scan_split(
        args,
        model,
        cfg,
        adapter,
        args.train_split,
        lambda batch_idx: ["heldout"] if (batch_idx % args.heldout_mod) == args.heldout_remainder else ["train"],
        class_ids,
        variants,
        confusions,
        feature_stats,
        rank_stats,
        num_classes,
        ignore_index,
        args.max_train_batches,
    )
    val_seen = scan_split(
        args,
        model,
        cfg,
        adapter,
        args.val_split,
        lambda batch_idx: ["val"],
        class_ids,
        variants,
        confusions,
        feature_stats,
        rank_stats,
        num_classes,
        ignore_index,
        args.max_val_batches,
    )

    class_rows, summaries = confusion_rows(confusions, class_ids, num_classes)
    pairs_csv = pair_rows(confusions, pairs)
    class_shift_rows, pair_centroid_rows = feature_shift_rows(feature_stats, class_ids, pairs)
    rank_csv = []
    for domain in ["heldout", "val"]:
        rank_csv.extend(rank_rows(domain, rank_stats[domain]))

    write_csv(
        args.output_dir / "coda_transfer_class_iou.csv",
        class_rows,
        ["domain", "variant", "class", "iou", "delta_iou", "acc", "target_count", "pred_count"],
    )
    write_csv(
        args.output_dir / "coda_transfer_pair_confusion.csv",
        pairs_csv,
        ["domain", "variant", "pair", "a_to_b_frac", "b_to_a_frac", "a_correct_frac", "b_correct_frac"],
    )
    write_csv(
        args.output_dir / "coda_transfer_feature_shift.csv",
        class_shift_rows,
        [
            "class",
            "train_count",
            "heldout_count",
            "val_count",
            "cos_train_heldout",
            "cos_train_val",
            "cos_heldout_val",
            "l2_train_heldout",
            "l2_train_val",
            "l2_heldout_val",
            "var_trace_train",
            "var_trace_heldout",
            "var_trace_val",
        ],
    )
    write_csv(
        args.output_dir / "coda_transfer_pair_centroids.csv",
        pair_centroid_rows,
        ["domain", "pair", "centroid_cos", "centroid_l2", "a_var_trace", "b_var_trace", "a_count", "b_count"],
    )
    write_csv(
        args.output_dir / "coda_transfer_picture_rank_drift.csv",
        rank_csv,
        [
            "domain",
            "variant",
            "picture_count",
            "gt_top1",
            "gt_top2",
            "gt_top5",
            "wall_top1",
            "mean_gt_rank",
            "mean_wall_rank",
            "margin_mean",
            "margin_std",
            "margin_p10",
            "margin_p50",
            "margin_p90",
        ],
    )

    metadata = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_seen_batches": train_seen,
        "val_seen_batches": val_seen,
        "variants": {k: list(v) if v is not None else None for k, v in variants.items()},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# CoDA Transfer-Failure Analysis",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- adapter: `{args.adapter_path}`",
        f"- train batches seen: `{train_seen}`",
        f"- val batches seen: `{val_seen}`",
        "",
        "## Why This Exists",
        "",
        "CoDA improved the heldout train split but failed on ScanNet val. This analysis",
        "checks whether the failure is visible as train/heldout/val feature drift,",
        "class/pair overcorrection, and picture candidate-ordering drift.",
        "",
        "## Feature Shift",
        "",
        "| class | train n | heldout n | val n | cos train-heldout | cos train-val | cos heldout-val | var train | var heldout | var val |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in class_shift_rows:
        if row["class"] in {"picture", "wall", "counter", "desk", "table", "sink", "cabinet"}:
            lines.append(
                f"| {row['class']} | {int(row['train_count'])} | {int(row['heldout_count'])} | {int(row['val_count'])} | "
                f"{row['cos_train_heldout']:.4f} | {row['cos_train_val']:.4f} | {row['cos_heldout_val']:.4f} | "
                f"{row['var_trace_train']:.2f} | {row['var_trace_heldout']:.2f} | {row['var_trace_val']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Picture Candidate-Ordering Drift",
            "",
            "| domain | variant | picture n | GT top1 | GT top2 | GT top5 | wall top1 | mean p-wall margin | p10 | p50 | p90 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    wanted_variants = {"base", "heldout_selected", "best_miou", "best_picture", "mid"}
    for row in rank_csv:
        if row["variant"] not in wanted_variants:
            continue
        lines.append(
            f"| {row['domain']} | {row['variant']} | {int(row['picture_count'])} | "
            f"{row['gt_top1']:.4f} | {row['gt_top2']:.4f} | {row['gt_top5']:.4f} | "
            f"{row['wall_top1']:.4f} | {row['margin_mean']:.4f} | {row['margin_p10']:.4f} | "
            f"{row['margin_p50']:.4f} | {row['margin_p90']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Pair Confusion: Heldout vs Val",
            "",
            "| domain | variant | pair | a->b | b->a | a correct | b correct |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pairs_csv:
        if row["variant"] not in {"base", "heldout_selected", "best_miou", "best_picture"}:
            continue
        if row["pair"] not in {"picture:wall", "counter:cabinet", "desk:table", "sink:cabinet"}:
            continue
        lines.append(
            f"| {row['domain']} | {row['variant']} | {row['pair']} | "
            f"{row['a_to_b_frac']:.4f} | {row['b_to_a_frac']:.4f} | "
            f"{row['a_correct_frac']:.4f} | {row['b_correct_frac']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{args.output_dir / 'coda_transfer_class_iou.csv'}`",
            f"- `{args.output_dir / 'coda_transfer_pair_confusion.csv'}`",
            f"- `{args.output_dir / 'coda_transfer_feature_shift.csv'}`",
            f"- `{args.output_dir / 'coda_transfer_pair_centroids.csv'}`",
            f"- `{args.output_dir / 'coda_transfer_picture_rank_drift.csv'}`",
            f"- `{args.output_dir / 'metadata.json'}`",
            "",
        ]
    )
    (args.output_dir / "coda_transfer_failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
