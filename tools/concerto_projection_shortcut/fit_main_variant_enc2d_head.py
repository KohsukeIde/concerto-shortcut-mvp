#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from main_variant_step05_utils import (
    build_loader,
    build_main_variant_model,
    evaluate_enc2d_modes,
    freeze_all_but_patch_proj,
    parse_csv,
    repo_root_from_here,
    save_json,
    seed_everything,
    select_dataset_specs,
    trainable_parameter_count,
    write_causal_results,
    move_batch_to_cuda,
    iter_limited,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the missing enc2d alignment projection for the Concerto paper "
            "main-variant backbone, keeping the backbone and DINO image encoder frozen."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="pretrain-concerto-v1m1-0-probe-enc2d-baseline")
    parser.add_argument(
        "--weight",
        type=Path,
        default=None,
        help="Main-variant backbone weight. Defaults to data/weights/concerto/concerto_base_origin.pth.",
    )
    parser.add_argument(
        "--datasets",
        default="arkit,scannet,scannetpp,s3dis,hm3d,structured3d",
        help="Comma-separated six-dataset subset for fitting/eval.",
    )
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tag", default="main-origin-headfit")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches-per-dataset", type=int, default=128)
    parser.add_argument("--max-val-batches-per-dataset", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-worker", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-datasets", default="arkit,scannet")
    parser.add_argument(
        "--modes",
        default="none,global_target_permutation,cross_image_target_swap,cross_scene_target_swap",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    seed_everything(args.seed)
    if args.weight is None:
        args.weight = repo_root / "data" / "weights" / "concerto" / "concerto_base_origin.pth"
    if args.output_root is None:
        args.output_root = repo_root / "data" / "runs" / "main_variant_enc2d_headfit" / args.tag
    args.output_root.mkdir(parents=True, exist_ok=True)

    cfg, model, load_info = build_main_variant_model(repo_root, args.config, args.weight)
    if load_info.get("bare_backbone"):
        print("[info] input weight is a bare backbone release; fitting/evaluating the missing enc2d head on top.")
    else:
        print(
            "[info] input weight already includes refit enc2d/patch-proj modules; "
            "running continued diagnostic eval on the head-refit checkpoint."
        )
    freeze_all_but_patch_proj(model)
    model.eval()
    print(f"[model] trainable_parameters={trainable_parameter_count(model)}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_specs = select_dataset_specs(
        repo_root,
        parse_csv(args.datasets),
        allow_missing=args.allow_missing_datasets,
    )
    if not train_specs:
        raise RuntimeError("no training datasets are available")
    train_loaders = {
        spec.name: build_loader(
            cfg,
            spec,
            "train",
            args.train_batch_size,
            args.num_worker,
            shuffle=True,
        )
        for spec in train_specs
    }

    steps = 0
    epoch_summaries = []
    train_limit = (
        min(args.max_train_batches_per_dataset, 8)
        if args.dry_run
        else args.max_train_batches_per_dataset
    )
    val_limit = (
        min(args.max_val_batches_per_dataset, 4)
        if args.dry_run
        else args.max_val_batches_per_dataset
    )
    for epoch in range(1 if args.dry_run else args.epochs):
        per_dataset = {}
        for name, loader in train_loaders.items():
            losses = []
            for _, batch in iter_limited(loader, train_limit):
                batch = move_batch_to_cuda(batch)
                output = model(batch)
                loss = output.get("loss_for_backward", output["loss"])
                if not loss.requires_grad:
                    print(
                        f"[fit] skip dataset={name} batch_without_trainable_enc2d_loss "
                        f"enc2d_loss={float(output['enc2d_loss'].detach().cpu().item()):.6f}"
                    )
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                value = float(output["enc2d_loss"].detach().cpu().item())
                losses.append(value)
                steps += 1
            per_dataset[name] = {
                "batches": len(losses),
                "enc2d_loss_mean": sum(losses) / max(len(losses), 1),
            }
            print(
                f"[fit] epoch={epoch + 1} dataset={name} "
                f"batches={per_dataset[name]['batches']} "
                f"enc2d_loss={per_dataset[name]['enc2d_loss_mean']:.6f}"
            )
        epoch_summaries.append({"epoch": epoch + 1, "datasets": per_dataset})

    checkpoint_path = args.output_root / "model_last.pth"
    metadata = {
        "kind": "main_variant_frozen_backbone_enc2d_headfit",
        "config": args.config,
        "source_weight": str(args.weight.resolve()),
        "load_info": load_info,
        "epochs": 1 if args.dry_run else args.epochs,
        "steps": steps,
        "trainable_parameters": trainable_parameter_count(model),
        "train_datasets": [spec.name for spec in train_specs],
        "epoch_summaries": epoch_summaries,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "metadata": metadata,
            },
            checkpoint_path,
        )
        print(f"[fit] wrote {checkpoint_path}")
        model.cuda()
    save_json(args.output_root / "metadata.json", metadata)

    eval_specs = select_dataset_specs(
        repo_root,
        parse_csv(args.eval_datasets),
        allow_missing=args.allow_missing_datasets,
    )
    eval_loaders = {
        spec.name: build_loader(
            cfg,
            spec,
            "val",
            args.eval_batch_size,
            args.num_worker,
            shuffle=False,
        )
        for spec in eval_specs
    }
    rows = evaluate_enc2d_modes(
        model,
        eval_loaders,
        parse_csv(args.modes),
        val_limit,
    )
    raw_path = args.output_root / "causal_rows.json"
    save_json(raw_path, {"rows": rows})
    if not args.dry_run:
        write_causal_results(
            rows,
            repo_root / "tools" / "concerto_projection_shortcut" / "results_main_variant_causal_battery.csv",
            repo_root / "tools" / "concerto_projection_shortcut" / "results_main_variant_causal_battery.md",
            "Main-Variant Frozen-Backbone Head-Refit Causal Battery",
        )
    write_causal_results(
        rows,
        args.output_root / "results_main_variant_causal_battery.csv",
        args.output_root / "results_main_variant_causal_battery.md",
        "Main-Variant Frozen-Backbone Head-Refit Causal Battery",
    )
    print(f"[eval] wrote {args.output_root / 'results_main_variant_causal_battery.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
