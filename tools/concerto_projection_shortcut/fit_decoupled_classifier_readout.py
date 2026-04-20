#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    PointBank,
    build_loader,
    build_model,
    eval_tensors,
    forward_features,
    load_config,
    move_to_cuda,
    picture_to_wall,
    summarize_confusion,
    update_confusion,
    weak_mean,
)


@dataclass
class ClassifierBundle:
    name: str
    weight: torch.Tensor
    bias: torch.Tensor
    train_loss: float | None = None


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decoupled classifier readout pilot for the concerto_base_origin "
            "decoder-probe checkpoint. Freezes decoder features and tests "
            "tau-normalization, logit adjustment, cRT, and Balanced Softmax."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=256)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-bank-points", type=int, default=200000)
    parser.add_argument("--max-per-class", type=int, default=10000)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced-sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--tau-values", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--logit-adjust-alphas", default="0.25,0.5,1")
    parser.add_argument("--mix-lambdas", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_decoupled_classifier_readout"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_names(text: str) -> list[int]:
    out = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {name}")
        out.append(NAME_TO_ID[name])
    if not out:
        raise ValueError("empty weak class list")
    return out


def base_classifier(model, feat_dim: int, num_classes: int) -> ClassifierBundle:
    if not hasattr(model, "seg_head") or not isinstance(model.seg_head, nn.Linear):
        raise RuntimeError("model must expose a linear seg_head for decoupled classifier readout")
    weight = model.seg_head.weight.detach().float().cpu()
    bias = (
        model.seg_head.bias.detach().float().cpu()
        if model.seg_head.bias is not None
        else torch.zeros(num_classes, dtype=torch.float32)
    )
    if weight.shape != (num_classes, feat_dim):
        raise RuntimeError(f"unexpected classifier shape {tuple(weight.shape)} for feat_dim={feat_dim}")
    return ClassifierBundle("base_head", weight, bias)


def tau_normalized(bundle: ClassifierBundle, tau: float, no_bias: bool = False) -> ClassifierBundle:
    weight = bundle.weight.clone()
    norm = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
    weight = weight / norm.pow(tau)
    bias = torch.zeros_like(bundle.bias) if no_bias else bundle.bias.clone()
    suffix = "nobias" if no_bias else "bias"
    return ClassifierBundle(f"tau{tau:g}_{suffix}".replace(".", "p"), weight, bias)


def class_counts(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.bincount(labels.clamp(0, num_classes - 1), minlength=num_classes).float()


def append_to_bank_and_count(raw: dict, feat: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, max_points: int, max_per_class: int, num_classes: int) -> None:
    valid = (labels >= 0) & (labels < num_classes)
    raw["raw_counts"] += torch.bincount(labels[valid].detach().cpu(), minlength=num_classes).long()
    total = raw["total"]
    if total >= max_points:
        return
    keep_indices = []
    for cls in range(num_classes):
        room_cls = max_per_class - raw["class_counts"][cls]
        if room_cls <= 0:
            continue
        idx = ((labels == cls) & valid).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        room_total = max_points - total - sum(x.numel() for x in keep_indices)
        if room_total <= 0:
            break
        cap = min(room_cls, room_total)
        if idx.numel() > cap:
            idx = idx[:cap]
        keep_indices.append(idx)
        raw["class_counts"][cls] += int(idx.numel())
    if keep_indices:
        keep = torch.cat(keep_indices, dim=0)
        raw["feat"].append(feat[keep].detach().cpu())
        raw["logits"].append(logits[keep].detach().cpu())
        raw["labels"].append(labels[keep].detach().cpu())
        raw["total"] += int(keep.numel())


def collect_bank_with_raw_counts(args: argparse.Namespace, model, cfg, num_classes: int) -> tuple[PointBank, torch.Tensor]:
    raw = {
        "feat": [],
        "logits": [],
        "labels": [],
        "class_counts": {i: 0 for i in range(num_classes)},
        "total": 0,
        "raw_counts": torch.zeros(num_classes, dtype=torch.long),
    }
    loader = build_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, _ = forward_features(model, batch)
            append_to_bank_and_count(raw, feat, logits, labels, args.max_bank_points, args.max_per_class, num_classes)
            seen += 1
            if (batch_idx + 1) % 25 == 0:
                counts = " ".join(f"{SCANNET20_CLASS_NAMES[k]}={v}" for k, v in raw["class_counts"].items() if v)
                raw_counts = " ".join(
                    f"{SCANNET20_CLASS_NAMES[k]}={int(v)}" for k, v in enumerate(raw["raw_counts"].tolist()) if v
                )
                print(f"[bank] batch={batch_idx + 1} total={raw['total']} capped {counts}", flush=True)
                print(f"[prior] batch={batch_idx + 1} raw {raw_counts}", flush=True)
    if not raw["labels"]:
        raise RuntimeError("empty classifier bank")
    bank = PointBank(
        feat=torch.cat(raw["feat"], dim=0).float(),
        logits=torch.cat(raw["logits"], dim=0).float(),
        labels=torch.cat(raw["labels"], dim=0).long(),
        class_counts={SCANNET20_CLASS_NAMES[k]: int(v) for k, v in raw["class_counts"].items()},
        seen_batches=seen,
    )
    print(f"[bank] done points={bank.labels.numel()} seen_batches={seen}", flush=True)
    return bank, raw["raw_counts"].float()


def sample_indices(labels: torch.Tensor, by_class: list[torch.Tensor], batch_size: int, balanced: bool, generator: torch.Generator) -> torch.Tensor:
    if not balanced:
        return torch.randint(0, labels.numel(), (batch_size,), generator=generator, device=labels.device)
    cls = torch.randint(0, len(by_class), (batch_size,), generator=generator, device=labels.device)
    out = torch.empty(batch_size, dtype=torch.long, device=labels.device)
    for c in cls.unique(sorted=False).tolist():
        pos = (cls == c).nonzero(as_tuple=False).flatten()
        pool = by_class[c]
        if pool.numel() == 0:
            out[pos] = torch.randint(0, labels.numel(), (pos.numel(),), generator=generator, device=labels.device)
        else:
            choice = torch.randint(0, pool.numel(), (pos.numel(),), generator=generator, device=labels.device)
            out[pos] = pool[choice]
    return out


def balanced_softmax_loss(logits: torch.Tensor, target: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    log_prior = prior.clamp_min(1e-12).log().to(logits.device)
    return F.cross_entropy(logits + log_prior.unsqueeze(0), target)


def train_classifier(
    name: str,
    bank: PointBank,
    init: ClassifierBundle,
    num_classes: int,
    mode: str,
    train_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    balanced_sampler: bool,
    seed: int,
) -> ClassifierBundle:
    feat = bank.feat.float().cuda()
    labels = bank.labels.long().cuda()
    feat_dim = feat.shape[1]
    clf = nn.Linear(feat_dim, num_classes).cuda()
    with torch.no_grad():
        clf.weight.copy_(init.weight.cuda())
        clf.bias.copy_(init.bias.cuda())
    counts = class_counts(labels.detach().cpu(), num_classes).cuda().clamp_min(1.0)
    prior = counts / counts.sum()
    by_class = [(labels == c).nonzero(as_tuple=False).flatten() for c in range(num_classes)]
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device=labels.device)
    gen.manual_seed(seed)
    last_loss = 0.0
    clf.train()
    for step in range(1, train_steps + 1):
        idx = sample_indices(labels, by_class, batch_size, balanced_sampler, gen)
        logits = clf(feat[idx])
        target = labels[idx]
        if mode == "balanced_softmax":
            loss = balanced_softmax_loss(logits, target, prior)
        elif mode == "ce":
            loss = F.cross_entropy(logits, target)
        else:
            raise ValueError(f"unknown mode: {mode}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().item())
        if step == 1 or step % 250 == 0 or step == train_steps:
            print(f"[train:{name}] step={step}/{train_steps} loss={last_loss:.6f}", flush=True)
    return ClassifierBundle(name, clf.weight.detach().cpu().float(), clf.bias.detach().cpu().float(), train_loss=last_loss)


def logits_from_bundle(feat: torch.Tensor, bundle: ClassifierBundle) -> torch.Tensor:
    return F.linear(feat.float(), bundle.weight.to(feat.device), bundle.bias.to(feat.device))


def init_confusions(names: list[str], num_classes: int) -> dict[str, torch.Tensor]:
    return {name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda") for name in names}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (f"{row[key]:.8f}" if isinstance(row.get(key), float) else row.get(key, ""))
                    for key in fields
                }
            )


def evaluate_variants(args, model, cfg, variants: dict[str, ClassifierBundle], prior: torch.Tensor, weak_classes: list[int]):
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    names = list(cfg.data.names)
    confusions = init_confusions(list(variants.keys()), num_classes)
    logit_adjust_alphas = parse_float_list(args.logit_adjust_alphas)
    mix_lambdas = parse_float_list(args.mix_lambdas)
    for alpha in logit_adjust_alphas:
        confusions[f"base_logit_adjust_alpha{alpha:g}".replace(".", "p")] = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device="cuda"
        )
    for variant_name in variants:
        if variant_name == "base_head" or variant_name.startswith("tau"):
            continue
        for lam in mix_lambdas:
            confusions[f"mix_{variant_name}_lam{lam:g}".replace(".", "p")] = torch.zeros(
                (num_classes, num_classes), dtype=torch.int64, device="cuda"
            )
    log_prior = prior.clamp_min(1e-12).log().cuda()
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, batch = forward_features(model, batch)
            feat_e, logits_e, labels_e = eval_tensors(feat, logits, labels, batch)
            valid = (labels_e >= 0) & (labels_e < num_classes) & (labels_e != ignore_index)
            if not valid.any():
                continue
            feat_e = feat_e[valid].float()
            logits_e = logits_e[valid].float()
            labels_e = labels_e[valid].long()
            for variant_name, bundle in variants.items():
                variant_logits = logits_from_bundle(feat_e, bundle)
                pred = variant_logits.argmax(dim=1)
                update_confusion(confusions[variant_name], pred, labels_e, num_classes, ignore_index)
                if variant_name != "base_head" and not variant_name.startswith("tau"):
                    for lam in mix_lambdas:
                        mixed = (1.0 - lam) * logits_e + lam * variant_logits
                        name = f"mix_{variant_name}_lam{lam:g}".replace(".", "p")
                        update_confusion(confusions[name], mixed.argmax(dim=1), labels_e, num_classes, ignore_index)
            for alpha in logit_adjust_alphas:
                pred = (logits_e - alpha * log_prior.unsqueeze(0)).argmax(dim=1)
                name = f"base_logit_adjust_alpha{alpha:g}".replace(".", "p")
                update_confusion(confusions[name], pred, labels_e, num_classes, ignore_index)
            seen += 1
            if seen % 25 == 0:
                base_sum = summarize_confusion(confusions["base_head"].detach().cpu().numpy(), names)
                print(f"[val] batches={seen} base_mIoU={base_sum['mIoU']:.4f}", flush=True)
    summaries = {name: summarize_confusion(conf.detach().cpu().numpy(), names) for name, conf in confusions.items()}
    base = summaries["base_head"]
    base_conf = confusions["base_head"].detach().cpu().numpy()
    rows = []
    class_rows = []
    for name, summary in summaries.items():
        conf_np = confusions[name].detach().cpu().numpy()
        row = {
            "variant": name,
            "mIoU": summary["mIoU"],
            "delta_mIoU": summary["mIoU"] - base["mIoU"],
            "mAcc": summary["mAcc"],
            "allAcc": summary["allAcc"],
            "weak_mean_iou": weak_mean(summary, weak_classes),
            "delta_weak_mean_iou": weak_mean(summary, weak_classes) - weak_mean(base, weak_classes),
            "picture_iou": float(summary["iou"][NAME_TO_ID["picture"]]),
            "delta_picture_iou": float(summary["iou"][NAME_TO_ID["picture"]] - base["iou"][NAME_TO_ID["picture"]]),
            "picture_to_wall_frac": picture_to_wall(conf_np, summary),
            "delta_picture_to_wall_frac": picture_to_wall(conf_np, summary) - picture_to_wall(base_conf, base),
            "counter_iou": float(summary["iou"][NAME_TO_ID["counter"]]),
            "desk_iou": float(summary["iou"][NAME_TO_ID["desk"]]),
            "sink_iou": float(summary["iou"][NAME_TO_ID["sink"]]),
            "cabinet_iou": float(summary["iou"][NAME_TO_ID["cabinet"]]),
            "door_iou": float(summary["iou"][NAME_TO_ID["door"]]),
            "shower_curtain_iou": float(summary["iou"][NAME_TO_ID["shower curtain"]]),
        }
        rows.append(row)
        for cls, cls_name in enumerate(SCANNET20_CLASS_NAMES):
            class_rows.append(
                {
                    "variant": name,
                    "class_id": cls,
                    "class_name": cls_name,
                    "iou": float(summary["iou"][cls]),
                    "delta_iou": float(summary["iou"][cls] - base["iou"][cls]),
                    "acc": float(summary["acc"][cls]),
                    "target_sum": float(summary["target_sum"][cls]),
                    "pred_sum": float(summary["pred_sum"][cls]),
                }
            )
    rows.sort(key=lambda r: (r["mIoU"], r["picture_iou"]), reverse=True)
    return rows, class_rows, summaries, confusions, seen


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.config = str((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    cfg = load_config(Path(args.config))
    num_classes = int(cfg.data.num_classes)
    weak_classes = parse_names(args.weak_classes)
    if args.dry_run:
        loader = build_loader(cfg, args.val_split, args.data_root, 1, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        return 0

    model = build_model(cfg, args.weight)
    bank, raw_counts = collect_bank_with_raw_counts(args, model, cfg, num_classes)
    init = base_classifier(model, bank.feat.shape[1], num_classes)
    counts = class_counts(bank.labels, num_classes).clamp_min(1.0)
    prior_counts = raw_counts.clamp_min(1.0)
    prior = prior_counts / prior_counts.sum()
    print(
        "[class-counts] "
        + " ".join(f"{SCANNET20_CLASS_NAMES[i]}={int(counts[i].item())}" for i in range(num_classes)),
        flush=True,
    )
    print(
        "[raw-prior-counts] "
        + " ".join(f"{SCANNET20_CLASS_NAMES[i]}={int(prior_counts[i].item())}" for i in range(num_classes)),
        flush=True,
    )

    variants: dict[str, ClassifierBundle] = {"base_head": init}
    for tau in parse_float_list(args.tau_values):
        if tau == 0:
            continue
        variants[tau_normalized(init, tau, no_bias=False).name] = tau_normalized(init, tau, no_bias=False)
        variants[tau_normalized(init, tau, no_bias=True).name] = tau_normalized(init, tau, no_bias=True)

    variants["crt_ce"] = train_classifier(
        "crt_ce",
        bank,
        init,
        num_classes,
        mode="ce",
        train_steps=args.train_steps,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        balanced_sampler=args.balanced_sampler,
        seed=args.seed + 1,
    )
    variants["crt_balanced_softmax"] = train_classifier(
        "crt_balanced_softmax",
        bank,
        init,
        num_classes,
        mode="balanced_softmax",
        train_steps=args.train_steps,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        balanced_sampler=args.balanced_sampler,
        seed=args.seed + 2,
    )

    rows, class_rows, summaries, confusions, seen = evaluate_variants(args, model, cfg, variants, prior, weak_classes)

    fields = [
        "variant",
        "mIoU",
        "delta_mIoU",
        "mAcc",
        "allAcc",
        "weak_mean_iou",
        "delta_weak_mean_iou",
        "picture_iou",
        "delta_picture_iou",
        "picture_to_wall_frac",
        "delta_picture_to_wall_frac",
        "counter_iou",
        "desk_iou",
        "sink_iou",
        "cabinet_iou",
        "door_iou",
        "shower_curtain_iou",
    ]
    write_csv(args.output_dir / "decoupled_classifier_variants.csv", rows, fields)
    write_csv(args.summary_prefix.with_suffix(".csv"), rows, fields)
    write_csv(
        args.output_dir / "decoupled_classifier_class_metrics.csv",
        class_rows,
        ["variant", "class_id", "class_name", "iou", "delta_iou", "acc", "target_sum", "pred_sum"],
    )
    meta = {
        "config": str(args.config),
        "weight": str(args.weight),
        "data_root": str(args.data_root),
        "seen_val_batches": seen,
        "bank_points": int(bank.labels.numel()),
        "bank_class_counts": bank.class_counts,
        "train_steps": args.train_steps,
        "train_batch_size": args.train_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "balanced_sampler": args.balanced_sampler,
        "tau_values": parse_float_list(args.tau_values),
        "logit_adjust_alphas": parse_float_list(args.logit_adjust_alphas),
        "mix_lambdas": parse_float_list(args.mix_lambdas),
        "raw_prior_counts": {SCANNET20_CLASS_NAMES[i]: int(prior_counts[i].item()) for i in range(num_classes)},
        "train_losses": {k: v.train_loss for k, v in variants.items() if v.train_loss is not None},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_row = next(r for r in rows if r["variant"] == "base_head")
    best_miou = rows[0]
    best_picture = max([r for r in rows if r["variant"] != "base_head"], key=lambda r: (r["picture_iou"], r["mIoU"]))
    safe = [r for r in rows if r["variant"] != "base_head" and r["mIoU"] >= base_row["mIoU"] - 0.002]
    best_safe_picture = max(safe, key=lambda r: (r["picture_iou"], r["mIoU"])) if safe else None

    def fmt(row: dict | None) -> str:
        if row is None:
            return "n/a"
        return (
            f"`{row['variant']}` mIoU={row['mIoU']:.4f} "
            f"(Δ{row['delta_mIoU']:+.4f}), picture={row['picture_iou']:.4f} "
            f"(Δ{row['delta_picture_iou']:+.4f}), p->wall={row['picture_to_wall_frac']:.4f} "
            f"(Δ{row['delta_picture_to_wall_frac']:+.4f})"
        )

    md = [
        "# Decoupled Classifier Readout",
        "",
        "Frozen decoder-feature classifier-family pilot for the `concerto_base_origin` decoder probe. "
        "This tests whether long-tail / decoupled classifier learning can recover the actionability headroom "
        "left by fixed-logit, pair-emphasis, retrieval, and LoRA variants.",
        "",
        "## Setup",
        "",
        f"- Config: `{args.config}`",
        f"- Weight: `{args.weight}`",
        f"- Bank points: `{bank.labels.numel()}`",
        f"- Seen val batches: `{seen}`",
        f"- Train steps: `{args.train_steps}`, train batch size `{args.train_batch_size}`, lr `{args.lr}`",
        f"- Balanced sampler: `{args.balanced_sampler}`",
        "",
        "## Headline",
        "",
        f"- Base: {fmt(base_row)}",
        f"- Best mIoU: {fmt(best_miou)}",
        f"- Best picture: {fmt(best_picture)}",
        f"- Best safe picture (mIoU >= base - 0.002): {fmt(best_safe_picture)}",
        "",
        "## Top Variants",
        "",
        "| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        md.append(
            f"| {rank} | `{row['variant']}` | {row['mIoU']:.4f} | {row['delta_mIoU']:+.4f} | "
            f"{row['weak_mean_iou']:.4f} | {row['delta_weak_mean_iou']:+.4f} | "
            f"{row['picture_iou']:.4f} | {row['delta_picture_iou']:+.4f} | "
            f"{row['picture_to_wall_frac']:.4f} | {row['delta_picture_to_wall_frac']:+.4f} |"
        )
    md.extend(
        [
            "",
            "## Interpretation Gate",
            "",
            "- Promising if a variant improves mIoU by >= +0.003, or improves `picture` by >= +0.02 while keeping mIoU within -0.002.",
            "- If no variant passes, decoupled classifier learning is not recovering the oracle/actionability headroom under this offline protocol.",
            "",
            "## Files",
            "",
            f"- Variant CSV: `{args.output_dir / 'decoupled_classifier_variants.csv'}`",
            f"- Class CSV: `{args.output_dir / 'decoupled_classifier_class_metrics.csv'}`",
            f"- Metadata: `{args.output_dir / 'metadata.json'}`",
        ]
    )
    text = "\n".join(md) + "\n"
    (args.output_dir / "decoupled_classifier_readout.md").write_text(text, encoding="utf-8")
    args.summary_prefix.with_suffix(".md").write_text(text, encoding="utf-8")
    print(f"[done] wrote {args.summary_prefix.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
