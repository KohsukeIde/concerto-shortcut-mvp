#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
        description=(
            "Region-level expert-choice coherence diagnostic for ScanNet20 fusion. "
            "This tests whether cross-model complementarity is spatially coherent enough "
            "for region-smoothed expert selection."
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
    parser.add_argument("--region-sizes", default="4,8,16")
    parser.add_argument("--min-region-points", type=int, default=16)
    parser.add_argument("--current-model", action="append", default=[])
    parser.add_argument("--include-utonia", action="store_true")
    parser.add_argument("--utonia-weight", type=Path, default=Path("data/weights/utonia/utonia.pth"))
    parser.add_argument("--utonia-head", type=Path, default=Path("data/weights/utonia/utonia_linear_prob_head_sc.pth"))
    parser.add_argument("--disable-utonia-flash", action="store_true")
    parser.add_argument("--cached-expert", action="append", default=[], help="Repeated spec name::cache_dir.")
    parser.add_argument(
        "--default-expert-name",
        default=None,
        help="Expert used as the default for region-defer variants. If omitted, the first loaded expert is used.",
    )
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_weak_ids(text: str) -> list[int]:
    return [NAME_TO_ID[x.strip()] for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def picture_to_wall(conf: np.ndarray) -> float:
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    den = conf[pic].sum()
    return float(conf[pic, wall] / den) if den else float("nan")


def raw_grid_from_batch(batch: dict) -> torch.Tensor:
    if "grid_coord" not in batch or "inverse" not in batch:
        raise RuntimeError("grid_coord and inverse are required for region coherence")
    return batch["grid_coord"].long()[batch["inverse"].long()].cpu()


def region_inverse(grid: torch.Tensor, size: int) -> torch.Tensor:
    rg = torch.div(grid, size, rounding_mode="floor")
    _, inv = torch.unique(rg, dim=0, return_inverse=True)
    return inv.long()


def choose_region_best_expert(preds: torch.Tensor, labels: torch.Tensor, inv: torch.Tensor, n_regions: int) -> torch.Tensor:
    # preds: [E, N]. Choose one expert per region by max correct count. Ties
    # keep expert 0 due torch.argmax.
    correct = preds.eq(labels[None, :])
    counts = torch.zeros((n_regions, preds.shape[0]), dtype=torch.long)
    for e in range(preds.shape[0]):
        counts[:, e] = torch.bincount(inv, weights=correct[e].float(), minlength=n_regions).long()
    return counts.argmax(dim=1)


def apply_region_choice(preds: torch.Tensor, inv: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
    region_expert = choice[inv]
    return preds.gather(0, region_expert[None, :]).squeeze(0)


def oracle_point_pred(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    correct = preds.eq(labels[None, :])
    any_correct = correct.any(dim=0)
    out = preds[0].clone()
    out[any_correct] = labels[any_correct]
    return out


def region_target_purity(preds: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor, inv: torch.Tensor, n_regions: int) -> tuple[float, float]:
    correct = preds.eq(labels[None, :])
    gather = labels.clamp_min(0).clamp_max(probs.shape[2] - 1)
    correct_prob = probs[:, torch.arange(labels.numel()), gather].transpose(0, 1)
    masked = correct_prob.masked_fill(~correct.transpose(0, 1), -1.0)
    any_correct = correct.any(dim=0)
    target = torch.where(any_correct, masked.argmax(dim=1), torch.zeros_like(labels))
    n_experts = preds.shape[0]
    region_counts = torch.bincount(inv, minlength=n_regions).float()
    flat = inv * n_experts + target.long()
    counts = torch.bincount(flat, minlength=n_regions * n_experts).reshape(n_regions, n_experts).float()
    max_counts = counts.max(dim=1).values
    purity = max_counts / region_counts.clamp_min(1.0)
    valid_region = region_counts > 0
    point_sum = float(region_counts[valid_region].sum().item())
    weighted = float(max_counts[valid_region].sum().item())
    high80 = float(region_counts[valid_region & (purity >= 0.8)].sum().item())
    return weighted / max(point_sum, 1.0), high80 / max(point_sum, 1.0)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = resolve(repo_root, args.data_root)
    out_dir = resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_prefix = resolve(repo_root, args.summary_prefix)
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    region_sizes = parse_int_list(args.region_sizes)
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
    confs: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((num_classes, num_classes), dtype=torch.long))
    purity_sum = {s: 0.0 for s in region_sizes}
    purity80_sum = {s: 0.0 for s in region_sizes}
    point_sum = {s: 0 for s in region_sizes}
    aux_gain_rows = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            labels = batch["origin_segment"].long().cpu()
            grid = raw_grid_from_batch(batch)
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

            if args.default_expert_name is not None:
                if args.default_expert_name not in logits_by_name:
                    raise KeyError(
                        f"--default-expert-name={args.default_expert_name!r} is not among loaded experts: "
                        f"{list(logits_by_name)}"
                    )
                names = [args.default_expert_name] + [n for n in logits_by_name if n != args.default_expert_name]
            else:
                names = list(logits_by_name)
            preds = torch.stack([logits_by_name[n].argmax(dim=1).cpu() for n in names], dim=0)
            probs = torch.stack([torch.softmax(logits_by_name[n].float(), dim=1).cpu() for n in names], dim=0)
            valid = (labels >= 0) & (labels < num_classes)
            labels_v = labels[valid]
            grid_v = grid[valid]
            preds_v = preds[:, valid]
            probs_v = probs[:, valid]
            update_confusion(confs["point_oracle_all"], oracle_point_pred(preds_v, labels_v), labels_v, num_classes, -1)
            update_confusion(confs[f"single::{names[0]}"], preds_v[0], labels_v, num_classes, -1)

            for s in region_sizes:
                inv = region_inverse(grid_v, s)
                n_regions = int(inv.max().item()) + 1 if inv.numel() else 0
                choice = choose_region_best_expert(preds_v, labels_v, inv, n_regions)
                pred_region = apply_region_choice(preds_v, inv, choice)
                update_confusion(confs[f"region_oracle_all_s{s}"], pred_region, labels_v, num_classes, -1)
                purity, purity80 = region_target_purity(preds_v, probs_v, labels_v, inv, n_regions)
                purity_sum[s] += purity * int(labels_v.numel())
                purity80_sum[s] += purity80 * int(labels_v.numel())
                point_sum[s] += int(labels_v.numel())

                default_correct = preds_v[0] == labels_v
                for e, name in enumerate(names[1:], start=1):
                    aux_correct = preds_v[e] == labels_v
                    desired = (~default_correct) & aux_correct
                    danger = default_correct & (~aux_correct)
                    # Region oracle defers to aux only where aux improves the
                    # region-level correct count over the default.
                    def_counts = torch.bincount(inv, weights=default_correct.float(), minlength=n_regions)
                    aux_counts = torch.bincount(inv, weights=aux_correct.float(), minlength=n_regions)
                    defer_region = aux_counts > def_counts
                    pred = preds_v[0].clone()
                    mask = defer_region[inv]
                    pred[mask] = preds_v[e][mask]
                    update_confusion(confs[f"region_defer_oracle_s{s}::{name}"], pred, labels_v, num_classes, -1)
                    aux_gain_rows.append(
                        {
                            "scene": scene_name,
                            "region_size": s,
                            "expert": name,
                            "desired_B_points": int(desired.sum().item()),
                            "danger_A_points": int(danger.sum().item()),
                            "desired_in_defer_region": int((desired & mask).sum().item()),
                            "danger_in_defer_region": int((danger & mask).sum().item()),
                            "defer_point_frac": float(mask.float().mean().item()),
                        }
                    )
            if (batch_idx + 1) % 25 == 0:
                print(f"[region] scenes={batch_idx + 1}/{len(loader.dataset)}", flush=True)

    rows = []
    for variant, conf_t in sorted(confs.items()):
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
            }
        )
    purity_rows = [
        {
            "region_size": s,
            "target_expert_purity_point_weighted": purity_sum[s] / max(point_sum[s], 1),
            "points_in_regions_with_target_purity_ge_0p8": purity80_sum[s] / max(point_sum[s], 1),
            "points": point_sum[s],
        }
        for s in region_sizes
    ]
    write_csv(summary_prefix.with_suffix(".csv"), rows)
    write_csv(summary_prefix.with_name(summary_prefix.name + "_purity").with_suffix(".csv"), purity_rows)
    write_csv(summary_prefix.with_name(summary_prefix.name + "_aux_region_gain").with_suffix(".csv"), aux_gain_rows)
    write_csv(out_dir / "region_coherence_summary.csv", rows)
    write_csv(out_dir / "region_target_purity.csv", purity_rows)
    write_csv(out_dir / "region_aux_gain_by_scene.csv", aux_gain_rows)

    md = [
        "# Cross-Model Region Coherence Diagnostic",
        "",
        "This uses labels only as an oracle diagnostic. It asks whether expert choice is spatially coherent enough that region-smoothed expert selection could plausibly recover cross-model complementarity.",
        "",
        "## Oracle / Region Expert Choice",
        "",
        "| variant | mIoU | allAcc | weak mIoU | picture | p->wall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: float(r["mIoU"]), reverse=True):
        md.append(
            f"| `{row['variant']}` | `{float(row['mIoU']):.4f}` | `{float(row['allAcc']):.4f}` | "
            f"`{float(row['weak_mean_iou']):.4f}` | `{float(row['picture_iou']):.4f}` | `{float(row['picture_to_wall']):.4f}` |"
        )
    md.extend(["", "## Target Expert Region Purity", "", "| region size | weighted purity | points in purity>=0.8 regions |", "|---:|---:|---:|"])
    for row in purity_rows:
        md.append(
            f"| `{row['region_size']}` | `{row['target_expert_purity_point_weighted']:.4f}` | "
            f"`{row['points_in_regions_with_target_purity_ge_0p8']:.4f}` |"
        )
    summary_prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "models": list(confs.keys()),
                "region_sizes": region_sizes,
                "current_models": [spec.__dict__ | {"config": str(spec.config), "weight": str(spec.weight)} for spec in current_specs],
                "include_utonia": args.include_utonia,
                "cached_experts": [(name, str(path)) for name, path in cached_experts],
                "note": "Oracle diagnostic only; region choices use validation labels.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {summary_prefix.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
