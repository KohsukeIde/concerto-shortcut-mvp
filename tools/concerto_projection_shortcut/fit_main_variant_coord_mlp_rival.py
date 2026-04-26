#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from main_variant_step05_utils import (
    build_loader,
    build_main_variant_model,
    cosine_loss_scaled,
    iter_limited,
    move_batch_to_cuda,
    parse_csv,
    repo_root_from_here,
    save_json,
    seed_everything,
    select_dataset_specs,
    shifted_pred,
)


def collect_dataset_cache(
    model,
    cfg,
    spec,
    split_kind: str,
    cache_path: Path,
    max_batches: int,
    max_rows_per_batch: int,
    batch_size: int,
    num_worker: int,
    force: bool,
    dry_run: bool,
):
    if cache_path.is_file() and not force and not dry_run:
        print(f"[cache] reuse {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    loader = build_loader(
        cfg,
        spec,
        split_kind,
        batch_size=batch_size,
        num_worker=num_worker,
        shuffle=(split_kind == "train"),
    )
    coords = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in iter_limited(loader, max_batches):
            batch = move_batch_to_cuda(batch)
            extracted = model.extract_enc2d_coord_target(
                batch, max_rows=max_rows_per_batch
            )
            coord = extracted["coord"].detach().float().cpu()
            target = extracted["target"].detach().half().cpu()
            if coord.numel() == 0:
                continue
            coords.append(coord)
            targets.append(target)
            if (batch_idx + 1) % 50 == 0:
                print(
                    f"[cache] {spec.name}/{split_kind} batches={batch_idx + 1} "
                    f"rows={sum(item.shape[0] for item in coords)}"
                )

    if not coords:
        raise RuntimeError(f"no rows collected for {spec.name}/{split_kind}")
    payload = {
        "dataset": spec.name,
        "split_kind": split_kind,
        "coord": torch.cat(coords, dim=0),
        "target": torch.cat(targets, dim=0),
        "target_dim": int(targets[0].shape[1]),
        "target_shifted": bool(model.enc2d_cos_shift),
        "coord_normalize": bool(model.shortcut_probe["coord_normalize"]),
    }
    print(
        f"[cache] {spec.name}/{split_kind} rows={payload['coord'].shape[0]} "
        f"target_dim={payload['target_dim']} dry_run={dry_run}"
    )
    if not dry_run:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
        print(f"[cache] wrote {cache_path}")
    return payload


def merge_caches(caches: list[dict], dataset_to_id: dict[str, int]) -> dict:
    coords = []
    targets = []
    dataset_ids = []
    for cache in caches:
        coord = cache["coord"].float()
        target = cache["target"]
        coords.append(coord)
        targets.append(target)
        dataset_ids.append(
            torch.full(
                (coord.shape[0],),
                dataset_to_id[cache["dataset"]],
                dtype=torch.long,
            )
        )
    return {
        "coord": torch.cat(coords, dim=0),
        "target": torch.cat(targets, dim=0),
        "dataset_id": torch.cat(dataset_ids, dim=0),
        "target_dim": int(caches[0]["target_dim"]),
        "target_shifted": bool(caches[0]["target_shifted"]),
    }


def compute_coord_stats(caches: list[dict], dataset_to_id: dict[str, int]) -> dict:
    stats = {}
    for cache in caches:
        coord = cache["coord"].float()
        stats[cache["dataset"]] = {
            "dataset_id": dataset_to_id[cache["dataset"]],
            "mean": coord.mean(dim=0),
            "std": coord.std(dim=0, unbiased=False).clamp_min(1e-6),
            "rows": int(coord.shape[0]),
        }
    return stats


def normalize_coord(coord: torch.Tensor, dataset_id: torch.Tensor, stats: dict, id_to_name: dict[int, str]):
    out = torch.empty_like(coord)
    for did in torch.unique(dataset_id).tolist():
        mask = dataset_id == did
        item = stats[id_to_name[int(did)]]
        mean = item["mean"].to(coord.device)
        std = item["std"].to(coord.device)
        out[mask] = (coord[mask] - mean) / std
    return out


def iter_minibatches(num_rows: int, batch_size: int, shuffle: bool):
    order = torch.randperm(num_rows) if shuffle else torch.arange(num_rows)
    for start in range(0, num_rows, batch_size):
        yield order[start : start + batch_size]


def evaluate_prior(prior, cache: dict, stats: dict, id_to_name: dict[int, str], batch_size: int):
    prior.eval()
    losses = []
    with torch.inference_mode():
        for idx in iter_minibatches(cache["coord"].shape[0], batch_size, shuffle=False):
            coord = cache["coord"][idx].cuda(non_blocking=True)
            dataset_id = cache["dataset_id"][idx].cuda(non_blocking=True)
            target = cache["target"][idx].cuda(non_blocking=True).float()
            norm_coord = normalize_coord(coord, dataset_id, stats, id_to_name)
            pred = shifted_pred(prior(norm_coord), bool(cache["target_shifted"]))
            losses.append(cosine_loss_scaled(pred, target).detach().cpu())
    return float(torch.stack(losses).mean().item())


def read_causal_reference(path: Path) -> dict:
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = {}
    for row in rows:
        dataset = row.get("dataset")
        mode = row.get("mode")
        if not dataset or not mode:
            continue
        out[(dataset, mode)] = float(row["enc2d_loss_mean"])
    return out


def write_results(
    rows: list[dict],
    csv_path: Path,
    md_path: Path,
    causal_reference: dict,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "coord_mlp_loss",
        "full_baseline_loss",
        "target_swap_mean_loss",
        "distance_to_baseline",
        "distance_to_swap_mean",
        "relative_position",
        "gate_hint",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Main-Variant Coord-MLP Rival Fit",
        "",
        "| dataset | coord MLP loss | full baseline | target-swap mean | distance to baseline | distance to swap | relative position | gate hint |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['coord_mlp_loss']} | {row['full_baseline_loss']} | "
            f"{row['target_swap_mean_loss']} | {row['distance_to_baseline']} | "
            f"{row['distance_to_swap_mean']} | {row['relative_position']} | {row['gate_hint']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `relative_position = (coord_loss - full_baseline) / (target_swap_mean - full_baseline)`.",
            "- Values near 0 mean the coord rival is close to the full head-refit baseline; values near 1 mean it is near target-swap damage.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a frozen tiny coord-MLP rival for main-variant Step 0.5."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="pretrain-concerto-v1m1-0-probe-enc2d-baseline")
    parser.add_argument("--weight", type=Path, default=None)
    parser.add_argument(
        "--datasets",
        default="arkit,scannet,scannetpp,s3dis,hm3d,structured3d",
    )
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tag", default="main-origin-coord-mlp-rival")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-train-batches-per-dataset", type=int, default=256)
    parser.add_argument("--max-val-batches-per-dataset", type=int, default=64)
    parser.add_argument("--max-rows-per-batch", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=2)
    parser.add_argument("--prior-epochs", type=int, default=20)
    parser.add_argument("--prior-batch-size", type=int, default=8192)
    parser.add_argument("--prior-hidden-channels", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--causal-csv", type=Path, default=None)
    parser.add_argument(
        "--skip-repo-results",
        action="store_true",
        help="Write results only under output-root; do not overwrite the repo-level canonical coord-rival CSV/MD.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    seed_everything(args.seed)
    if args.weight is None:
        args.weight = repo_root / "data" / "weights" / "concerto" / "concerto_base_origin.pth"
    if args.output_root is None:
        args.output_root = repo_root / "data" / "runs" / "main_variant_coord_mlp_rival" / args.tag
    if args.cache_root is None:
        args.cache_root = args.output_root / "cache"
    if args.causal_csv is None:
        args.causal_csv = (
            repo_root
            / "tools"
            / "concerto_projection_shortcut"
            / "results_main_variant_causal_battery.csv"
        )

    cfg, model, load_info = build_main_variant_model(repo_root, args.config, args.weight)
    model.requires_grad_(False)
    model.eval()

    specs = select_dataset_specs(
        repo_root,
        parse_csv(args.datasets),
        allow_missing=args.allow_missing_datasets,
    )
    if not specs:
        raise RuntimeError("no datasets are available")
    dataset_to_id = {spec.name: i for i, spec in enumerate(specs)}
    id_to_name = {i: name for name, i in dataset_to_id.items()}

    train_caches = []
    val_caches = []
    for spec in specs:
        train_caches.append(
            collect_dataset_cache(
                model,
                cfg,
                spec,
                "train",
                args.cache_root / f"{spec.name}_train.pt",
                max_batches=2 if args.dry_run else args.max_train_batches_per_dataset,
                max_rows_per_batch=args.max_rows_per_batch,
                batch_size=args.extract_batch_size,
                num_worker=args.num_worker,
                force=args.force_cache,
                dry_run=args.dry_run,
            )
        )
        val_caches.append(
            collect_dataset_cache(
                model,
                cfg,
                spec,
                "val",
                args.cache_root / f"{spec.name}_val.pt",
                max_batches=2 if args.dry_run else args.max_val_batches_per_dataset,
                max_rows_per_batch=args.max_rows_per_batch,
                batch_size=args.extract_batch_size,
                num_worker=args.num_worker,
                force=args.force_cache,
                dry_run=args.dry_run,
            )
        )

    del model
    torch.cuda.empty_cache()

    stats = compute_coord_stats(train_caches, dataset_to_id)
    train_cache = merge_caches(train_caches, dataset_to_id)
    val_cache = merge_caches(val_caches, dataset_to_id)

    from pointcept.models.concerto.concerto_v1m1_base import CoordPrior

    prior = CoordPrior(
        "mlp",
        int(train_cache["target_dim"]),
        hidden_channels=args.prior_hidden_channels,
    ).cuda()
    optimizer = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=1e-4)
    epochs = 1 if args.dry_run else args.prior_epochs
    for epoch in range(epochs):
        prior.train()
        epoch_losses = []
        for idx in iter_minibatches(
            train_cache["coord"].shape[0], args.prior_batch_size, shuffle=True
        ):
            coord = train_cache["coord"][idx].cuda(non_blocking=True)
            dataset_id = train_cache["dataset_id"][idx].cuda(non_blocking=True)
            target = train_cache["target"][idx].cuda(non_blocking=True).float()
            norm_coord = normalize_coord(coord, dataset_id, stats, id_to_name)
            pred = shifted_pred(prior(norm_coord), bool(train_cache["target_shifted"]))
            loss = cosine_loss_scaled(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        print(
            f"[fit] epoch={epoch + 1}/{epochs} "
            f"train_scaled_cosine_loss={sum(epoch_losses) / max(len(epoch_losses), 1):.6f}"
        )

    per_dataset_losses = {}
    for cache in val_caches:
        merged = merge_caches([cache], dataset_to_id)
        per_dataset_losses[cache["dataset"]] = evaluate_prior(
            prior,
            merged,
            stats,
            id_to_name,
            args.prior_batch_size,
        )
    all_val_loss = evaluate_prior(prior, val_cache, stats, id_to_name, args.prior_batch_size)
    print(f"[eval] all_val_scaled_cosine_loss={all_val_loss:.6f}")

    causal = read_causal_reference(args.causal_csv)
    result_rows = []
    for dataset, loss in per_dataset_losses.items():
        baseline = causal.get((dataset, "none"))
        swap_values = [
            value
            for (ds, mode), value in causal.items()
            if ds == dataset and mode != "none"
        ]
        swap_mean = sum(swap_values) / len(swap_values) if swap_values else None
        if baseline is None or swap_mean is None or abs(swap_mean - baseline) < 1e-9:
            rel = None
            hint = "no_reference"
            dist_base = None
            dist_swap = None
        elif swap_mean <= baseline:
            dist_base = loss - baseline
            dist_swap = swap_mean - loss
            rel = None
            hint = "target_swaps_not_positive"
        else:
            dist_base = loss - baseline
            dist_swap = swap_mean - loss
            rel = (loss - baseline) / (swap_mean - baseline)
            if rel <= 0.35:
                hint = "strong_go"
            elif rel < 0.75:
                hint = "partial"
            else:
                hint = "no_go"
        result_rows.append(
            {
                "dataset": dataset,
                "coord_mlp_loss": f"{loss:.6f}",
                "full_baseline_loss": "" if baseline is None else f"{baseline:.6f}",
                "target_swap_mean_loss": "" if swap_mean is None else f"{swap_mean:.6f}",
                "distance_to_baseline": "" if dist_base is None else f"{dist_base:.6f}",
                "distance_to_swap_mean": "" if dist_swap is None else f"{dist_swap:.6f}",
                "relative_position": "" if rel is None else f"{rel:.6f}",
                "gate_hint": hint,
            }
        )

    metadata = {
        "kind": "main_variant_coord_mlp_rival",
        "config": args.config,
        "source_weight": str(args.weight.resolve()),
        "load_info": load_info,
        "datasets": [spec.name for spec in specs],
        "all_val_scaled_cosine_loss": all_val_loss,
        "per_dataset_scaled_cosine_loss": per_dataset_losses,
        "dataset_to_id": dataset_to_id,
        "coord_stats": {
            name: {
                "dataset_id": int(item["dataset_id"]),
                "mean": [float(x) for x in item["mean"].tolist()],
                "std": [float(x) for x in item["std"].tolist()],
                "rows": int(item["rows"]),
            }
            for name, item in stats.items()
        },
        "dry_run": bool(args.dry_run),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(args.output_root / "metrics.json", metadata)
    if not args.dry_run:
        torch.save(
            {
                "state_dict": prior.cpu().state_dict(),
                "arch": "mlp",
                "target_dim": int(train_cache["target_dim"]),
                "hidden_channels": int(args.prior_hidden_channels),
                "dataset_to_id": dataset_to_id,
                "coord_stats": {
                    name: {
                        "mean": item["mean"],
                        "std": item["std"],
                        "rows": int(item["rows"]),
                    }
                    for name, item in stats.items()
                },
                "metadata": metadata,
            },
            args.output_root / "model_last.pth",
        )
        print(f"[fit] wrote {args.output_root / 'model_last.pth'}")

    if not args.dry_run and not args.skip_repo_results:
        write_results(
            result_rows,
            repo_root / "tools" / "concerto_projection_shortcut" / "results_official_coord_mlp_rival.csv",
            repo_root / "tools" / "concerto_projection_shortcut" / "results_official_coord_mlp_rival.md",
            causal,
        )
    write_results(
        result_rows,
        args.output_root / "results_official_coord_mlp_rival.csv",
        args.output_root / "results_official_coord_mlp_rival.md",
        causal,
    )
    print(f"[results] wrote {args.output_root / 'results_official_coord_mlp_rival.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
