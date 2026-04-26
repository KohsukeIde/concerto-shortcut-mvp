#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from fit_main_variant_coord_mlp_rival import (
    compute_coord_stats,
    cosine_loss_scaled,
    merge_caches,
    normalize_coord,
    read_causal_reference,
    shifted_pred,
)
from pointcept.models.concerto.concerto_v1m1_base import CoordPrior


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tools/concerto_projection_shortcut"


def fmt(x: float | str | None, digits: int = 6) -> str:
    if x is None or x == "":
        return ""
    return f"{float(x):.{digits}f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def iter_minibatches(num_rows: int, batch_size: int, shuffle: bool):
    order = torch.randperm(num_rows) if shuffle else torch.arange(num_rows)
    for start in range(0, num_rows, batch_size):
        yield order[start : start + batch_size]


def loss_for_constant(pred: torch.Tensor, target: torch.Tensor, batch_size: int = 4096) -> float:
    losses = []
    with torch.inference_mode():
        for idx in iter_minibatches(target.shape[0], batch_size, shuffle=False):
            batch_pred = pred.unsqueeze(0).expand(idx.numel(), -1).to(target.device)
            losses.append(cosine_loss_scaled(batch_pred, target[idx].float()).detach().cpu())
    return float(torch.stack(losses).mean().item())


def eval_dataset(
    model,
    cache: dict,
    stats: dict,
    dataset_to_id: dict[str, int],
    batch_size: int,
) -> float:
    merged = merge_caches([cache], dataset_to_id)
    id_to_name = {i: name for name, i in dataset_to_id.items()}
    device = next(model.parameters()).device
    model.eval()
    losses = []
    with torch.inference_mode():
        for idx in iter_minibatches(merged["coord"].shape[0], batch_size, shuffle=False):
            coord = merged["coord"][idx].to(device)
            dataset_id = merged["dataset_id"][idx].to(device)
            target = merged["target"][idx].to(device).float()
            norm_coord = normalize_coord(coord, dataset_id, stats, id_to_name)
            pred = shifted_pred(model(norm_coord), bool(merged["target_shifted"]))
            losses.append(cosine_loss_scaled(pred, target).detach().cpu())
    return float(torch.stack(losses).mean().item())


def train_s3dis_only(
    train_cache: dict,
    val_cache: dict,
    epochs: int,
    batch_size: int,
    hidden_channels: int,
    lr: float,
) -> tuple[CoordPrior, list[dict[str, float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dim = int(train_cache["target_dim"])
    model = CoordPrior("mlp", target_dim, hidden_channels=hidden_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    coord_mean = train_cache["coord"].float().mean(dim=0).to(device)
    coord_std = train_cache["coord"].float().std(dim=0, unbiased=False).clamp_min(1e-6).to(device)

    train_coord = train_cache["coord"].float()
    train_target = train_cache["target"].float()
    val_coord = val_cache["coord"].float()
    val_target = val_cache["target"].float()
    target_shifted = bool(train_cache["target_shifted"])
    history = []

    def eval_split(coord: torch.Tensor, target: torch.Tensor) -> float:
        model.eval()
        losses = []
        with torch.inference_mode():
            for idx in iter_minibatches(coord.shape[0], batch_size, shuffle=False):
                x = ((coord[idx].to(device) - coord_mean) / coord_std)
                y = target[idx].to(device)
                pred = shifted_pred(model(x), target_shifted)
                losses.append(cosine_loss_scaled(pred, y).detach().cpu())
        return float(torch.stack(losses).mean().item())

    for epoch in range(epochs):
        model.train()
        losses = []
        for idx in iter_minibatches(train_coord.shape[0], batch_size, shuffle=True):
            x = ((train_coord[idx].to(device) - coord_mean) / coord_std)
            y = train_target[idx].to(device)
            pred = shifted_pred(model(x), target_shifted)
            loss = cosine_loss_scaled(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch in {0, 4, 9, epochs - 1}:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": sum(losses) / max(1, len(losses)),
                    "val_loss": eval_split(val_coord, val_target),
                }
            )
    return model.cpu(), history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "data/runs/main_variant_coord_mlp_rival/main-origin-six-step05",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-channels", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cache_root = args.run_root / "cache"
    datasets = ["arkit", "scannet", "scannetpp", "s3dis", "hm3d", "structured3d"]
    train_caches = {
        ds: torch.load(cache_root / f"{ds}_train.pt", map_location="cpu", weights_only=False)
        for ds in datasets
    }
    val_caches = {
        ds: torch.load(cache_root / f"{ds}_val.pt", map_location="cpu", weights_only=False)
        for ds in datasets
    }
    dataset_to_id = {ds: i for i, ds in enumerate(datasets)}
    stats = compute_coord_stats([train_caches[ds] for ds in datasets], dataset_to_id)

    ckpt = torch.load(args.run_root / "model_last.pth", map_location="cpu", weights_only=False)
    shared = CoordPrior("mlp", int(ckpt["target_dim"]), hidden_channels=int(ckpt["hidden_channels"]))
    shared.load_state_dict(ckpt["state_dict"])
    if torch.cuda.is_available():
        shared = shared.cuda()

    causal = read_causal_reference(OUT_DIR / "results_main_variant_causal_battery.csv")
    coord_rows = {r["dataset"]: r for r in read_csv(OUT_DIR / "results_coord_mlp_rival_six_dataset_calibration.csv")}

    rows = []
    for ds in datasets:
        train_cache = train_caches[ds]
        val_cache = val_caches[ds]
        train_coord = train_cache["coord"].float()
        val_coord = val_cache["coord"].float()
        train_target = train_cache["target"].float()
        val_target = val_cache["target"].float()
        target_mean_cos = torch.nn.functional.cosine_similarity(
            train_target.mean(dim=0, keepdim=True),
            val_target.mean(dim=0, keepdim=True),
        ).item()
        train_loss = eval_dataset(shared, train_cache, stats, dataset_to_id, args.batch_size)
        val_loss = eval_dataset(shared, val_cache, stats, dataset_to_id, args.batch_size)
        train_mean_loss = loss_for_constant(train_target.mean(dim=0), train_target)
        val_mean_loss = loss_for_constant(train_target.mean(dim=0), val_target)
        rows.append(
            {
                "dataset": ds,
                "train_rows": int(train_cache["coord"].shape[0]),
                "val_rows": int(val_cache["coord"].shape[0]),
                "coord_mean_shift_l2": float((train_coord.mean(dim=0) - val_coord.mean(dim=0)).norm().item()),
                "coord_std_shift_l2": float((train_coord.std(dim=0, unbiased=False) - val_coord.std(dim=0, unbiased=False)).norm().item()),
                "target_mean_cosine_train_val": float(target_mean_cos),
                "shared_mlp_train_loss": train_loss,
                "shared_mlp_val_loss": val_loss,
                "train_target_mean_train_loss": train_mean_loss,
                "train_target_mean_val_loss": val_mean_loss,
                "clean_loss": causal[(ds, "none")],
                "mean_corruption_loss": float(coord_rows[ds]["mean_corruption_loss"]),
                "closure_fraction_mean": float(coord_rows[ds]["closure_fraction_mean"]),
            }
        )

    s3_model, s3_history = train_s3dis_only(
        train_caches["s3dis"],
        val_caches["s3dis"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        lr=args.lr,
    )

    # Evaluate the S3DIS-only model on S3DIS train/val using its own stats.
    s3_dataset_to_id = {"s3dis": 0}
    s3_stats = compute_coord_stats([train_caches["s3dis"]], s3_dataset_to_id)
    if torch.cuda.is_available():
        s3_model = s3_model.cuda()
    s3_train_loss = eval_dataset(s3_model, train_caches["s3dis"], s3_stats, s3_dataset_to_id, args.batch_size)
    s3_val_loss = eval_dataset(s3_model, val_caches["s3dis"], s3_stats, s3_dataset_to_id, args.batch_size)

    s3_clean = causal[("s3dis", "none")]
    s3_corrupt = float(coord_rows["s3dis"]["mean_corruption_loss"])
    denom = s3_corrupt - s3_clean
    s3_only_rel = (s3_val_loss - s3_clean) / denom if denom > 0 else ""
    s3_only_closure = 1.0 - s3_only_rel if s3_only_rel != "" else ""

    out = {
        "cache_rows": rows,
        "s3dis_only_history": s3_history,
        "s3dis_only_final": {
            "train_loss": s3_train_loss,
            "val_loss": s3_val_loss,
            "clean_loss": s3_clean,
            "mean_corruption_loss": s3_corrupt,
            "relative_position_mean": s3_only_rel,
            "closure_fraction_mean": s3_only_closure,
        },
    }
    json_path = OUT_DIR / "results_coord_mlp_s3dis_failure_diagnostic.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    csv_path = OUT_DIR / "results_coord_mlp_s3dis_failure_diagnostic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# S3DIS Coord-MLP Rival Failure Diagnostic",
        "",
        "This diagnostic checks whether the negative S3DIS coordinate-rival closure is a genuine no-coordinate signal or an evaluation/cache artifact.",
        "",
        "| dataset | train rows | val rows | coord mean shift | target mean cosine | shared MLP train | shared MLP val | train-mean val | clean | mean corrupt | closure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['dataset']}` | `{r['train_rows']}` | `{r['val_rows']}` | `{fmt(r['coord_mean_shift_l2'])}` | "
            f"`{fmt(r['target_mean_cosine_train_val'])}` | `{fmt(r['shared_mlp_train_loss'])}` | "
            f"`{fmt(r['shared_mlp_val_loss'])}` | `{fmt(r['train_target_mean_val_loss'])}` | "
            f"`{fmt(r['clean_loss'])}` | `{fmt(r['mean_corruption_loss'])}` | `{100 * r['closure_fraction_mean']:.1f}%` |"
        )
    lines.extend(
        [
            "",
            "## S3DIS-only coord MLP",
            "",
            "| epoch | train loss | val loss |",
            "|---:|---:|---:|",
        ]
    )
    for h in s3_history:
        lines.append(f"| `{int(h['epoch'])}` | `{fmt(h['train_loss'])}` | `{fmt(h['val_loss'])}` |")
    lines.extend(
        [
            "",
            f"- Final S3DIS-only train loss: `{fmt(s3_train_loss)}`.",
            f"- Final S3DIS-only val loss: `{fmt(s3_val_loss)}`.",
            f"- S3DIS-only closure against mean corruption: `{100 * s3_only_closure:.1f}%`." if s3_only_closure != "" else "- S3DIS-only closure unavailable.",
            "",
            "## Interpretation",
            "",
            "- The S3DIS coord-rival validation cache is extremely small compared with the other five datasets.",
            "- The S3DIS clean-to-corruption gap is also small, so normalized closure is unstable and can become strongly negative from modest absolute loss differences.",
            "- If the S3DIS-only MLP improves train loss but not validation loss, the safest reading is train/val or sample-selection mismatch rather than a clean six-dataset coordinate-only closure.",
        ]
    )
    (OUT_DIR / "results_coord_mlp_s3dis_failure_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {json_path}")
    print(f"[write] {csv_path}")
    print(f"[write] {OUT_DIR / 'results_coord_mlp_s3dis_failure_diagnostic.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
