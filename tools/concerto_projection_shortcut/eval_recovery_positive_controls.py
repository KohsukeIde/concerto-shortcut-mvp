#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_oracle_actionability_analysis import (  # noqa: E402
    candidate_mask,
    summarize_confusion,
    update_confusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Positive-control recovery sanity checks. We induce a known "
            "class-prior/logit-bias error on cached raw probabilities, then "
            "verify that the frozen class-prior recovery family can recover "
            "a nonzero fraction of the induced oracle headroom."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--train-cache-dir", type=Path, default=Path("data/runs/ptv3_v151_raw_probs_scannet20/train"))
    parser.add_argument("--val-cache-dir", type=Path, default=Path("data/runs/ptv3_v151_raw_probs_scannet20/full"))
    parser.add_argument("--output-prefix", type=Path, default=Path("tools/concerto_projection_shortcut/results_recovery_positive_controls"))
    parser.add_argument("--weak-classes", default="picture,counter,door,sink,shower curtain")
    parser.add_argument("--synthetic-mode", choices=["prior", "weak"], default="prior")
    parser.add_argument("--prior-bias-strength", type=float, default=1.0)
    parser.add_argument("--recover-alphas", default="0,0.25,0.5,0.75,1,1.25,1.5")
    parser.add_argument("--weak-bias", type=float, default=2.0)
    parser.add_argument("--max-train-points", type=int, default=1000000)
    parser.add_argument("--bias-steps", type=int, default=1000)
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260427)
    return parser.parse_args()


def parse_names(text: str) -> list[int]:
    out = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise KeyError(f"unknown class name: {name}")
        out.append(NAME_TO_ID[name])
    return out


def resolve(root: Path, path: Path) -> Path:
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def scene_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise RuntimeError(f"no npz caches under {path}")
    return files


def sample_train_logits_labels(files: list[Path], max_points: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    total = 0
    shuffled = list(files)
    rng.shuffle(shuffled)
    for path in shuffled:
        arr = np.load(path)
        probs = arr["probs"].astype(np.float32)
        labels = arr["labels"].astype(np.int64)
        valid = (labels >= 0) & (labels < len(SCANNET20_CLASS_NAMES))
        probs = probs[valid]
        labels = labels[valid]
        if labels.size == 0:
            continue
        room = max_points - total
        if room <= 0:
            break
        if labels.size > room:
            idx = np.arange(labels.size)
            rng.shuffle(idx)
            idx = idx[:room]
            probs = probs[idx]
            labels = labels[idx]
        chunks.append((probs, labels))
        total += labels.size
    if not chunks:
        raise RuntimeError("empty training sample")
    probs = np.concatenate([x[0] for x in chunks], axis=0)
    labels = np.concatenate([x[1] for x in chunks], axis=0)
    logits = torch.from_numpy(np.log(np.clip(probs, 1e-8, 1.0))).float()
    return logits, torch.from_numpy(labels).long()


def fit_bias(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, steps: int, lr: float, balanced: bool) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = logits.to(device)
    labels = labels.to(device)
    bias = torch.zeros(num_classes, device=device, requires_grad=True)
    weight = None
    if balanced:
        counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
        weight = counts.sum() / (num_classes * counts)
        weight = weight / weight.mean()
    opt = torch.optim.AdamW([bias], lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits + bias, labels, weight=weight)
        loss.backward()
        opt.step()
    return bias.detach().cpu()


def make_synthetic_bias(num_classes: int, weak_classes: list[int], weak_bias: float) -> torch.Tensor:
    bias = torch.zeros(num_classes, dtype=torch.float32)
    for cls in weak_classes:
        bias[cls] -= weak_bias
    return bias


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def eval_variants(files: list[Path], variant_biases: dict[str, torch.Tensor]) -> dict[str, dict]:
    num_classes = len(SCANNET20_CLASS_NAMES)
    confs = {name: torch.zeros((num_classes, num_classes), dtype=torch.int64) for name in variant_biases}
    confs["biased_oracle_top2"] = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for path in files:
        arr = np.load(path)
        probs = torch.from_numpy(arr["probs"].astype(np.float32))
        labels = torch.from_numpy(arr["labels"].astype(np.int64)).long()
        valid = (labels >= 0) & (labels < num_classes)
        if not valid.any():
            continue
        logits = torch.log(probs[valid].clamp_min(1e-8))
        labels = labels[valid]
        biased = logits + variant_biases["biased"]
        pred_biased = biased.argmax(dim=1)
        cand = candidate_mask(biased, 2)
        pred_oracle = torch.where(cand.gather(1, labels[:, None]).squeeze(1), labels, pred_biased)
        for name, bias in variant_biases.items():
            pred = (logits + bias).argmax(dim=1)
            update_confusion(confs[name], pred, labels, num_classes, -1)
        update_confusion(confs["biased_oracle_top2"], pred_oracle, labels, num_classes, -1)
    return {name: summarize_confusion(conf.numpy(), list(SCANNET20_CLASS_NAMES)) for name, conf in confs.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    train_dir = resolve(root, args.train_cache_dir)
    val_dir = resolve(root, args.val_cache_dir)
    out_prefix = resolve(root, args.output_prefix)
    weak_classes = parse_names(args.weak_classes)
    num_classes = len(SCANNET20_CLASS_NAMES)
    train_files = scene_files(train_dir)
    val_files = scene_files(val_dir)
    train_logits, train_labels = sample_train_logits_labels(train_files, args.max_train_points, args.seed)
    if args.synthetic_mode == "prior":
        counts = torch.bincount(train_labels, minlength=num_classes).float().clamp_min(1.0)
        direction = (counts / counts.sum()).log()
        direction = direction - direction.mean()
        synthetic_bias = args.prior_bias_strength * direction
    else:
        synthetic_bias = make_synthetic_bias(num_classes, weak_classes, args.weak_bias)
    biased_train = train_logits + synthetic_bias
    learned_bias = fit_bias(biased_train, train_labels, num_classes, args.bias_steps, args.bias_lr, balanced=False)
    balanced_bias = fit_bias(biased_train, train_labels, num_classes, args.bias_steps, args.bias_lr, balanced=True)
    variant_biases = {
        "clean": torch.zeros(num_classes, dtype=torch.float32),
        "biased": synthetic_bias,
        "learned_bias": synthetic_bias + learned_bias,
        "balanced_bias": synthetic_bias + balanced_bias,
    }
    for alpha in parse_float_list(args.recover_alphas):
        variant_biases[f"known_direction_recover_alpha{alpha:g}".replace(".", "p")] = synthetic_bias - alpha * synthetic_bias
    summaries = eval_variants(val_files, variant_biases)
    base = summaries["biased"]
    clean = summaries["clean"]
    rows = []
    picture = NAME_TO_ID["picture"]
    for name, summary in summaries.items():
        delta = summary["mIoU"] - base["mIoU"]
        denom = clean["mIoU"] - base["mIoU"]
        pic_delta = float(summary["iou"][picture] - base["iou"][picture])
        pic_denom = float(clean["iou"][picture] - base["iou"][picture])
        rows.append(
            {
                "variant": name,
                "mIoU": f"{summary['mIoU']:.8f}",
                "delta_vs_biased": f"{delta:.8f}",
                "fraction_to_clean": f"{(delta / denom if denom > 0 else 0.0):.8f}",
                "picture_iou": f"{float(summary['iou'][picture]):.8f}",
                "picture_delta_vs_biased": f"{pic_delta:.8f}",
                "picture_fraction_to_clean": f"{(pic_delta / pic_denom if pic_denom > 0 else 0.0):.8f}",
            }
        )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_prefix.with_suffix(".csv"), rows)
    metadata = {
        "train_cache_dir": str(train_dir),
        "val_cache_dir": str(val_dir),
        "weak_classes": [SCANNET20_CLASS_NAMES[i] for i in weak_classes],
        "synthetic_mode": args.synthetic_mode,
        "prior_bias_strength": args.prior_bias_strength,
        "weak_bias": args.weak_bias,
        "recover_alphas": parse_float_list(args.recover_alphas),
        "max_train_points": args.max_train_points,
        "train_points_used": int(train_labels.numel()),
        "num_val_scenes": len(val_files),
        "synthetic_bias": synthetic_bias.tolist(),
        "learned_bias": learned_bias.tolist(),
        "balanced_bias": balanced_bias.tolist(),
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Recovery Positive Controls",
        "",
        "Synthetic class-prior/logit-bias sanity check over cached PTv3 raw probabilities.",
        "We induce a known class-prior/logit-bias direction, then evaluate whether class-prior correction variants can recover the induced error.",
        "",
        "| variant | mIoU | delta vs biased | fraction to clean | picture | picture delta | picture fraction to clean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | `{row['mIoU']}` | `{row['delta_vs_biased']}` | `{row['fraction_to_clean']}` | "
            f"`{row['picture_iou']}` | `{row['picture_delta_vs_biased']}` | `{row['picture_fraction_to_clean']}` |"
        )
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {out_prefix.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
