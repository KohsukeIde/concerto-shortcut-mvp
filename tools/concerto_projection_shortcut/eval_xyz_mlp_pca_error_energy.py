#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_xyz_mlp_pca_rasa import (  # noqa: E402
    XYZMLP,
    build_loader,
    build_model,
    eval_tensors_with_coord,
    forward_features,
    load_config,
    move_to_cuda,
    projection_energy,
    predict_with_bias,
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Error-conditioned energy analysis for the XYZ-MLP PCA RASA "
            "diagnostic. Reuses the trained xyz-only MLP/PCA/probe state and "
            "compares coordinate-factor projection energy on correct hard "
            "points, hard-to-majority errors, and correct majority points."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py")
    parser.add_argument("--weight", type=Path, default=Path("data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth"))
    parser.add_argument("--state", type=Path, default=Path("data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_rasa_reservoir/xyz_mlp_pca_rasa_state.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_error_energy"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("tools/concerto_projection_shortcut/results_xyz_mlp_pca_error_energy"))
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--hard-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--majority-classes", default="wall,floor,cabinet,table,chair,door")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_names(text: str) -> list[int]:
    ids: list[int] = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {name}")
        ids.append(NAME_TO_ID[name])
    return ids


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    repo_root = args.repo_root.resolve()
    args.config = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.state = (repo_root / args.state).resolve() if not args.state.is_absolute() else args.state
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    return args


class StreamingStats:
    def __init__(self, sample_cap: int = 500000) -> None:
        self.count = 0
        self.sample_cap = sample_cap
        self.energy_sum = 0.0
        self.uhat_norm_sum = 0.0
        self.y_sum = torch.zeros(2, dtype=torch.float64)
        self.y2_sum = torch.zeros(2, dtype=torch.float64)
        self.res2_sum = torch.zeros(2, dtype=torch.float64)
        self.energy_values: list[torch.Tensor] = []
        self.uhat_values: list[torch.Tensor] = []

    def append_samples(self, energy: torch.Tensor, uhat_norm: torch.Tensor) -> None:
        if energy.numel() > self.sample_cap:
            idx = torch.randperm(energy.numel())[: self.sample_cap]
            energy = energy[idx]
            uhat_norm = uhat_norm[idx]
        self.energy_values.append(energy)
        self.uhat_values.append(uhat_norm)
        total = sum(int(x.numel()) for x in self.energy_values)
        if total <= self.sample_cap:
            return
        e_all = torch.cat(self.energy_values)
        u_all = torch.cat(self.uhat_values)
        idx = torch.randperm(e_all.numel())[: self.sample_cap]
        self.energy_values = [e_all[idx]]
        self.uhat_values = [u_all[idx]]

    def update(self, energy: torch.Tensor, u_true: torch.Tensor, u_hat: torch.Tensor, mask: torch.Tensor) -> None:
        n = int(mask.sum().item())
        if n == 0:
            return
        e = energy[mask].detach().cpu().float()
        uh = u_hat[mask].detach().cpu().float()
        yt = u_true[mask].detach().cpu().double()
        yp = uh.double()
        uh_norm = uh.norm(dim=1)
        self.count += n
        self.energy_sum += float(e.sum().item())
        self.uhat_norm_sum += float(uh_norm.sum().item())
        self.y_sum += yt.sum(dim=0)
        self.y2_sum += (yt * yt).sum(dim=0)
        self.res2_sum += ((yt - yp) ** 2).sum(dim=0)
        self.append_samples(e, uh_norm)

    def row(self, subset: str) -> dict:
        if self.count == 0:
            return {
                "subset": subset,
                "count": 0,
                "energy_mean": "",
                "energy_p50": "",
                "energy_p90": "",
                "u_hat_norm_mean": "",
                "u_hat_norm_p50": "",
                "u_hat_norm_p90": "",
                "r2": "",
                "r2_dim0": "",
                "r2_dim1": "",
            }
        energy = torch.cat(self.energy_values)
        uhat = torch.cat(self.uhat_values)
        ss_tot = (self.y2_sum - (self.y_sum * self.y_sum) / max(self.count, 1)).clamp_min(1e-8)
        r2_dim = 1.0 - self.res2_sum / ss_tot
        r2 = 1.0 - self.res2_sum.sum() / ss_tot.sum().clamp_min(1e-8)
        return {
            "subset": subset,
            "count": self.count,
            "energy_mean": self.energy_sum / self.count,
            "energy_p50": float(torch.quantile(energy, 0.50).item()),
            "energy_p90": float(torch.quantile(energy, 0.90).item()),
            "u_hat_norm_mean": self.uhat_norm_sum / self.count,
            "u_hat_norm_p50": float(torch.quantile(uhat, 0.50).item()),
            "u_hat_norm_p90": float(torch.quantile(uhat, 0.90).item()),
            "r2": float(r2.item()),
            "r2_dim0": float(r2_dim[0].item()),
            "r2_dim1": float(r2_dim[1].item()),
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "subset",
        "count",
        "energy_mean",
        "energy_p50",
        "energy_p90",
        "u_hat_norm_mean",
        "u_hat_norm_p50",
        "u_hat_norm_p90",
        "r2",
        "r2_dim0",
        "r2_dim1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k, "")
                out[k] = f"{v:.8f}" if isinstance(v, float) else v
            writer.writerow(out)


def add_mask(stats: dict[str, StreamingStats], name: str, energy: torch.Tensor, u_true: torch.Tensor, u_hat: torch.Tensor, mask: torch.Tensor) -> None:
    stats[name].update(energy, u_true, u_hat, mask)


def main() -> int:
    args = resolve_args(parse_args())
    cfg = load_config(args.config)
    num_classes = int(cfg.data.num_classes)
    hard_classes = parse_names(args.hard_classes)
    majority_classes = parse_names(args.majority_classes)

    if args.dry_run:
        loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        return 0

    hard_set = torch.tensor(hard_classes, dtype=torch.long, device="cuda")
    majority_set = torch.tensor(majority_classes, dtype=torch.long, device="cuda")

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    hidden_dim = int(state["metadata"]["xyz_hidden_dim"])
    xyz_model = XYZMLP(hidden_dim, num_classes).cuda().eval()
    xyz_model.load_state_dict(state["xyz_mlp_state_dict"], strict=True)
    hidden_mean = state["hidden_mean"].cuda().float()
    pca_components = state["pca_components"].cuda().float()
    pca_mean = state["pca_mean"].cuda().float()
    pca_std = state["pca_std"].cuda().float()
    feature_mean = state["feature_mean"].cuda().float()
    feature_std = state["feature_std"].cuda().float()
    probe_weight = state["probe_weight"].cuda().float()
    nuisance_basis = state["nuisance_basis"].cuda().float()

    model = build_model(cfg, args.weight)
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    stats: dict[str, StreamingStats] = defaultdict(StreamingStats)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    seen_batches = 0
    seen_points = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            feat, logits, label, batch = forward_features(model, batch)
            feat, logits, coord, label = eval_tensors_with_coord(feat, logits, label, batch)
            valid = (label >= 0) & (label < num_classes)
            if int(valid.sum().item()) == 0:
                continue
            feat = feat[valid].float()
            logits = logits[valid].float()
            coord = coord[valid].float()
            label = label[valid].long()
            pred = logits.argmax(dim=1).long()

            x = (feat - feature_mean) / feature_std
            energy = projection_energy(x, nuisance_basis)
            u_hat = predict_with_bias(x, probe_weight)
            hidden = xyz_model.hidden(coord)
            u_true = ((hidden - hidden_mean) @ pca_components - pca_mean) / pca_std

            bins = torch.bincount(label.cpu() * num_classes + pred.cpu(), minlength=num_classes * num_classes)
            confusion += bins.reshape(num_classes, num_classes)

            is_hard = torch.isin(label, hard_set)
            is_majority_gt = torch.isin(label, majority_set)
            pred_majority = torch.isin(pred, majority_set)
            correct = pred == label
            error = ~correct

            add_mask(stats, "all", energy, u_true, u_hat, torch.ones_like(label, dtype=torch.bool))
            add_mask(stats, "hard_all", energy, u_true, u_hat, is_hard)
            add_mask(stats, "hard_correct_all", energy, u_true, u_hat, is_hard & correct)
            add_mask(stats, "hard_error_any", energy, u_true, u_hat, is_hard & error)
            add_mask(stats, "hard_error_to_majority", energy, u_true, u_hat, is_hard & error & pred_majority)
            add_mask(stats, "majority_correct_all", energy, u_true, u_hat, is_majority_gt & correct)
            add_mask(stats, "majority_error_any", energy, u_true, u_hat, is_majority_gt & error)

            for cls in hard_classes:
                name = SCANNET20_CLASS_NAMES[cls].replace(" ", "_")
                cls_mask = label == cls
                add_mask(stats, f"{name}_all", energy, u_true, u_hat, cls_mask)
                add_mask(stats, f"{name}_correct", energy, u_true, u_hat, cls_mask & correct)
                add_mask(stats, f"{name}_error_any", energy, u_true, u_hat, cls_mask & error)
                add_mask(stats, f"{name}_error_to_majority", energy, u_true, u_hat, cls_mask & error & pred_majority)
            add_mask(stats, "picture_to_wall", energy, u_true, u_hat, (label == NAME_TO_ID["picture"]) & (pred == NAME_TO_ID["wall"]))
            add_mask(stats, "door_to_wall", energy, u_true, u_hat, (label == NAME_TO_ID["door"]) & (pred == NAME_TO_ID["wall"]))
            add_mask(stats, "counter_to_cabinet", energy, u_true, u_hat, (label == NAME_TO_ID["counter"]) & (pred == NAME_TO_ID["cabinet"]))
            add_mask(stats, "sink_to_cabinet", energy, u_true, u_hat, (label == NAME_TO_ID["sink"]) & (pred == NAME_TO_ID["cabinet"]))
            add_mask(stats, "desk_to_table", energy, u_true, u_hat, (label == NAME_TO_ID["desk"]) & (pred == NAME_TO_ID["table"]))
            add_mask(stats, "shower_curtain_to_wall", energy, u_true, u_hat, (label == NAME_TO_ID["shower curtain"]) & (pred == NAME_TO_ID["wall"]))

            seen_batches += 1
            seen_points += int(label.numel())
            if (batch_idx + 1) % 25 == 0:
                print(f"[eval] batch={batch_idx+1} points={seen_points}", flush=True)

    rows = [stats[name].row(name) for name in sorted(stats)]
    csv_path = args.output_dir / "xyz_mlp_pca_error_energy.csv"
    write_csv(csv_path, rows)
    summary_csv = args.summary_prefix.with_suffix(".csv")
    write_csv(summary_csv, rows)

    conf_rows = []
    for gt in hard_classes:
        total = int(confusion[gt].sum().item())
        top = torch.topk(confusion[gt].float(), k=min(5, num_classes)).indices.tolist() if total else []
        for pred_id in top:
            conf_rows.append(
                {
                    "gt": SCANNET20_CLASS_NAMES[gt],
                    "pred": SCANNET20_CLASS_NAMES[pred_id],
                    "count": int(confusion[gt, pred_id].item()),
                    "frac_of_gt": float(confusion[gt, pred_id].item() / max(total, 1)),
                }
            )
    conf_path = args.output_dir / "xyz_mlp_pca_error_confusions.csv"
    with conf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gt", "pred", "count", "frac_of_gt"])
        writer.writeheader()
        for row in conf_rows:
            writer.writerow({**row, "frac_of_gt": f"{row['frac_of_gt']:.8f}"})

    by_name = {row["subset"]: row for row in rows}
    def fmt(name: str, key: str) -> str:
        val = by_name.get(name, {}).get(key, "")
        return "NA" if val == "" else f"{float(val):.4f}"

    md_lines = [
        "# XYZ-MLP PCA Error-Conditioned Energy",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- state: `{args.state}`",
        f"- val batches: `{seen_batches}`",
        f"- val points: `{seen_points}`",
        "",
        "## Key Subsets",
        "",
        "| subset | count | energy mean | energy p90 | uhat norm mean | R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    key_names = [
        "all",
        "hard_correct_all",
        "hard_error_to_majority",
        "hard_error_any",
        "majority_correct_all",
        "picture_correct",
        "picture_to_wall",
        "counter_to_cabinet",
        "sink_to_cabinet",
        "desk_to_table",
        "door_to_wall",
        "shower_curtain_to_wall",
    ]
    for name in key_names:
        row = by_name.get(name)
        if not row:
            continue
        md_lines.append(
            f"| `{name}` | `{row['count']}` | `{fmt(name, 'energy_mean')}` | `{fmt(name, 'energy_p90')}` | `{fmt(name, 'u_hat_norm_mean')}` | `{fmt(name, 'r2')}` |"
        )

    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This table tests whether the task-conditioned coordinate factor is elevated specifically on hard-class errors.",
            "- A positive shortcut-error explanation would predict higher projection energy or better coordinate-target predictability on `hard_error_to_majority` than on `hard_correct_all` / `majority_correct_all`.",
            "- Use this as a diagnostic follow-up to the rank-2 removal no-go, not as an independent method result.",
            "",
            "## Output Files",
            "",
            f"- `{csv_path}`",
            f"- `{conf_path}`",
        ]
    )
    md_path = args.summary_prefix.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (args.output_dir / "xyz_mlp_pca_error_energy.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    metadata = {
        "config": str(args.config),
        "weight": str(args.weight),
        "state": str(args.state),
        "val_batches": seen_batches,
        "val_points": seen_points,
        "hard_classes": [SCANNET20_CLASS_NAMES[i] for i in hard_classes],
        "majority_classes": [SCANNET20_CLASS_NAMES[i] for i in majority_classes],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[write] {md_path}", flush=True)
    print(f"[write] {summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
