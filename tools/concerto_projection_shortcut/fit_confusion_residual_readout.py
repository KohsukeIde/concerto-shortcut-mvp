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
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
    parse_pairs,
    pair_name,
)


@dataclass
class PairExpert:
    pair: tuple[int, int]
    mean: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    train_bal_acc: float
    train_auc_proxy: float


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline confusion-graph residual readout for a frozen ScanNet "
            "decoder/semseg checkpoint. Fits pairwise linear experts on point "
            "features and adds antisymmetric corrections to fixed 20-way logits."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-train-batches", type=int, default=256)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--expert-steps", type=int, default=800)
    parser.add_argument("--expert-lr", type=float, default=0.05)
    parser.add_argument("--expert-weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambdas", default="0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--gates", default="none,top1_pair,top2_any,top2_both,uncertain_top2_any")
    parser.add_argument("--uncertainty-margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path):
    from pointcept.utils.config import Config

    return Config.fromfile(str(path))


def split_dataset_cfg(cfg, split: str, data_root: Path):
    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg.split = split
    ds_cfg.data_root = str(data_root)
    ds_cfg.test_mode = False
    return ds_cfg


def build_loader(cfg, split: str, data_root: Path, batch_size: int, num_worker: int):
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn

    dataset = build_dataset(split_dataset_cfg(cfg, split, data_root))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        collate_fn=lambda batch: point_collate_fn(batch, mix_prob=0),
    )


def build_model(cfg, weight_path: Path):
    from pointcept.models.builder import build_model

    model = build_model(cfg.model).cuda().eval()
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    info = model.load_state_dict(cleaned, strict=False)
    print(
        f"[load] weight={weight_path} missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}",
        flush=True,
    )
    if info.missing_keys:
        print(f"[load] first missing={info.missing_keys[:8]}", flush=True)
    if info.unexpected_keys:
        print(f"[load] first unexpected={info.unexpected_keys[:8]}", flush=True)
    return model


def move_to_cuda(input_dict: dict) -> dict:
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.cuda(non_blocking=True)
    return input_dict


@torch.no_grad()
def forward_features(model, batch: dict):
    out = model(batch, return_point=True)
    point = out["point"]
    feat = point.feat.float()
    logits = out["seg_logits"].float()
    labels = batch["segment"].long()
    if feat.shape[0] != logits.shape[0] or feat.shape[0] != labels.shape[0]:
        raise RuntimeError(f"shape mismatch feat={feat.shape} logits={logits.shape} labels={labels.shape}")
    return feat, logits, labels, batch


def collect_train_cache(args: argparse.Namespace, model, cfg, pairs: list[tuple[int, int]]):
    target_classes = sorted({cls for pair in pairs for cls in pair})
    class_features = {cls: [] for cls in target_classes}
    class_counts = {cls: 0 for cls in target_classes}
    loader = build_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker)
    seen_batches = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
                break
            batch = move_to_cuda(batch)
            feat, _, labels, _ = forward_features(model, batch)
            seen_batches += 1
            for cls in target_classes:
                room = args.max_per_class - class_counts[cls]
                if room <= 0:
                    continue
                idx = (labels == cls).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                if idx.numel() > room:
                    idx = idx[:room]
                class_features[cls].append(feat[idx].detach().cpu())
                class_counts[cls] += int(idx.numel())
            if (batch_idx + 1) % 25 == 0:
                counts = " ".join(f"{SCANNET20_CLASS_NAMES[k]}={v}" for k, v in class_counts.items())
                print(f"[collect-train] batch={batch_idx + 1} {counts}", flush=True)
    cache = {}
    for cls in target_classes:
        if not class_features[cls]:
            raise RuntimeError(f"no train features collected for class={SCANNET20_CLASS_NAMES[cls]}")
        cache[cls] = torch.cat(class_features[cls], dim=0)
    print(
        "[collect-train] done "
        + " ".join(f"{SCANNET20_CLASS_NAMES[k]}={v}" for k, v in class_counts.items())
        + f" seen_batches={seen_batches}",
        flush=True,
    )
    return cache, {"seen_batches": seen_batches, "class_counts": {SCANNET20_CLASS_NAMES[k]: v for k, v in class_counts.items()}}


def binary_auc_proxy(scores: torch.Tensor, y: torch.Tensor) -> float:
    # Cheap exact AUC via ranking for training diagnostics.
    scores = scores.detach().cpu()
    y = y.detach().cpu().bool()
    pos = scores[y]
    neg = scores[~y]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_ranks = ranks[y].sum()
    auc = (pos_ranks - pos.numel() * (pos.numel() + 1) / 2.0) / (pos.numel() * neg.numel())
    return float(auc.item())


def fit_expert(pair: tuple[int, int], train_cache: dict[int, torch.Tensor], args: argparse.Namespace) -> PairExpert:
    pos_cls, neg_cls = pair
    x_pos = train_cache[pos_cls]
    x_neg = train_cache[neg_cls]
    n = min(x_pos.shape[0], x_neg.shape[0])
    # Balance the optimization set to make the expert explicitly pair-focused.
    x = torch.cat([x_pos[:n], x_neg[:n]], dim=0).cuda()
    y = torch.cat([torch.ones(n), torch.zeros(n)], dim=0).cuda()
    perm = torch.randperm(x.shape[0], device=x.device)
    x = x[perm]
    y = y[perm]
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    xz = (x - mean) / std
    weight = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    bias = torch.zeros((), device=x.device, requires_grad=True)
    opt = torch.optim.AdamW([weight, bias], lr=args.expert_lr, weight_decay=args.expert_weight_decay)
    for step in range(args.expert_steps):
        opt.zero_grad(set_to_none=True)
        score = xz @ weight + bias
        loss = F.binary_cross_entropy_with_logits(score, y)
        loss.backward()
        opt.step()
        if (step + 1) % max(args.expert_steps // 4, 1) == 0:
            with torch.no_grad():
                pred = (score >= 0).float()
                pos_acc = (pred[y == 1] == 1).float().mean().item()
                neg_acc = (pred[y == 0] == 0).float().mean().item()
                print(
                    f"[expert] {pair_name(pair)} step={step + 1} loss={loss.item():.4f} "
                    f"bal={(pos_acc + neg_acc) / 2:.4f}",
                    flush=True,
                )
    with torch.no_grad():
        score = xz @ weight + bias
        pred = (score >= 0).float()
        pos_acc = (pred[y == 1] == 1).float().mean().item()
        neg_acc = (pred[y == 0] == 0).float().mean().item()
        bal = (pos_acc + neg_acc) / 2.0
        auc = binary_auc_proxy(score, y)
    return PairExpert(
        pair=pair,
        mean=mean.detach(),
        std=std.detach(),
        weight=weight.detach(),
        bias=bias.detach(),
        train_bal_acc=bal,
        train_auc_proxy=auc,
    )


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    valid = target != ignore_index
    pred = pred[valid].long()
    target = target[valid].long()
    in_range = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[in_range]
    target = target[in_range]
    flat = target * num_classes + pred
    confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def correction_gate(logits: torch.Tensor, pair: tuple[int, int], gate: str, uncertainty_margin: float) -> torch.Tensor:
    pos_cls, neg_cls = pair
    top = logits.topk(k=2, dim=1)
    top_idx = top.indices
    top_val = top.values
    in_top1 = (top_idx[:, 0] == pos_cls) | (top_idx[:, 0] == neg_cls)
    in_top2_any = (top_idx == pos_cls).any(dim=1) | (top_idx == neg_cls).any(dim=1)
    in_top2_both = ((top_idx == pos_cls).any(dim=1) & (top_idx == neg_cls).any(dim=1))
    uncertain = (top_val[:, 0] - top_val[:, 1]) <= uncertainty_margin
    if gate == "none":
        return torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    if gate == "top1_pair":
        return in_top1
    if gate == "top2_any":
        return in_top2_any
    if gate == "top2_both":
        return in_top2_both
    if gate == "uncertain_top2_any":
        return in_top2_any & uncertain
    raise ValueError(f"unknown gate: {gate}")


def apply_corrections(
    feat: torch.Tensor,
    logits: torch.Tensor,
    experts: list[PairExpert],
    lam: float,
    gate: str,
    uncertainty_margin: float,
) -> torch.Tensor:
    corrected = logits.clone()
    for expert in experts:
        pos_cls, neg_cls = expert.pair
        xz = (feat - expert.mean.to(feat.device)) / expert.std.to(feat.device)
        score = xz @ expert.weight.to(feat.device) + expert.bias.to(feat.device)
        mask = correction_gate(logits, expert.pair, gate, uncertainty_margin).float()
        delta = lam * score * mask
        corrected[:, pos_cls] += delta
        corrected[:, neg_cls] -= delta
    return corrected


def summarize_confusion(conf: np.ndarray, names: list[str]):
    inter = np.diag(conf).astype(np.float64)
    target_sum = conf.sum(axis=1).astype(np.float64)
    pred_sum = conf.sum(axis=0).astype(np.float64)
    union = target_sum + pred_sum - inter
    iou = inter / (union + 1e-10)
    acc = inter / (target_sum + 1e-10)
    return {
        "mIoU": float(iou.mean()),
        "mAcc": float(acc.mean()),
        "allAcc": float(inter.sum() / (target_sum.sum() + 1e-10)),
        "iou": iou,
        "acc": acc,
        "target_sum": target_sum,
        "pred_sum": pred_sum,
        "intersection": inter,
        "union": union,
        "names": names,
    }


def evaluate_variants(args: argparse.Namespace, model, cfg, experts: list[PairExpert]):
    names = list(cfg.data.names)
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    gates = [x.strip() for x in args.gates.split(",") if x.strip()]
    variants = [("base", 0.0, "base")]
    variants.extend((f"residual_lam{str(lam).replace('.', 'p')}_{gate}", lam, gate) for gate in gates for lam in lambdas)
    confusions = {name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda") for name, _, _ in variants}
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    seen_batches = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, batch = forward_features(model, batch)
            target = labels
            inverse = batch.get("inverse")
            origin_segment = batch.get("origin_segment")
            for variant_name, lam, gate in variants:
                if variant_name == "base":
                    pred = logits.argmax(dim=1)
                else:
                    corrected = apply_corrections(feat, logits, experts, lam, gate, args.uncertainty_margin)
                    pred = corrected.argmax(dim=1)
                eval_target = target
                eval_pred = pred
                if inverse is not None and origin_segment is not None:
                    eval_pred = pred[inverse]
                    eval_target = origin_segment
                update_confusion(confusions[variant_name], eval_pred, eval_target, num_classes, ignore_index)
            seen_batches += 1
            if (batch_idx + 1) % 25 == 0:
                base_sum = summarize_confusion(confusions["base"].detach().cpu().numpy(), names)
                print(f"[eval] batch={batch_idx + 1} base_mIoU={base_sum['mIoU']:.4f}", flush=True)
    summaries = {name: summarize_confusion(conf.detach().cpu().numpy(), names) for name, conf in confusions.items()}
    return summaries, confusions, {"seen_batches": seen_batches, "variants": [name for name, _, _ in variants]}


def write_outputs(args: argparse.Namespace, summaries: dict, confusions: dict, experts: list[PairExpert], metadata: dict):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = summaries["base"]["names"]
    base = summaries["base"]
    summary_path = args.output_dir / "confusion_residual_summary.csv"
    fields = [
        "variant",
        "mIoU",
        "mAcc",
        "allAcc",
        "delta_mIoU",
        "picture_iou",
        "delta_picture_iou",
        "picture_to_wall_frac",
        "wall_to_picture_frac",
    ]
    rows = []
    for variant, summary in summaries.items():
        conf = confusions[variant].detach().cpu().numpy()
        picture_id = NAME_TO_ID["picture"]
        wall_id = NAME_TO_ID["wall"]
        picture_target = max(summary["target_sum"][picture_id], 1.0)
        wall_target = max(summary["target_sum"][wall_id], 1.0)
        row = {
            "variant": variant,
            "mIoU": summary["mIoU"],
            "mAcc": summary["mAcc"],
            "allAcc": summary["allAcc"],
            "delta_mIoU": summary["mIoU"] - base["mIoU"],
            "picture_iou": summary["iou"][picture_id],
            "delta_picture_iou": summary["iou"][picture_id] - base["iou"][picture_id],
            "picture_to_wall_frac": conf[picture_id, wall_id] / picture_target,
            "wall_to_picture_frac": conf[wall_id, picture_id] / wall_target,
        }
        rows.append(row)
    rows_sorted = sorted(rows, key=lambda r: (r["mIoU"], r["picture_iou"]), reverse=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in row.items()})

    class_path = args.output_dir / "confusion_residual_class_metrics.csv"
    with class_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "class_id", "class_name", "iou", "accuracy", "target_count", "pred_count", "delta_iou"],
        )
        writer.writeheader()
        for variant, summary in summaries.items():
            for idx, name in enumerate(names):
                writer.writerow(
                    {
                        "variant": variant,
                        "class_id": idx,
                        "class_name": name,
                        "iou": f"{summary['iou'][idx]:.8f}",
                        "accuracy": f"{summary['acc'][idx]:.8f}",
                        "target_count": int(summary["target_sum"][idx]),
                        "pred_count": int(summary["pred_sum"][idx]),
                        "delta_iou": f"{summary['iou'][idx] - base['iou'][idx]:.8f}",
                    }
                )

    expert_path = args.output_dir / "pair_experts.json"
    expert_payload = [
        {
            "pair": pair_name(expert.pair),
            "positive_class": SCANNET20_CLASS_NAMES[expert.pair[0]],
            "negative_class": SCANNET20_CLASS_NAMES[expert.pair[1]],
            "train_bal_acc": expert.train_bal_acc,
            "train_auc_proxy": expert.train_auc_proxy,
        }
        for expert in experts
    ]
    expert_path.write_text(json.dumps(expert_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metadata_path = args.output_dir / "metadata.json"
    metadata_payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "metadata": metadata,
        "experts": expert_payload,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_path = args.output_dir / "confusion_residual_readout.md"
    best_miou = rows_sorted[0]
    best_picture = sorted(rows, key=lambda r: (r["picture_iou"], r["mIoU"]), reverse=True)[0]
    lines = [
        "# Confusion-Graph Residual Readout",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- class pairs: `{args.class_pairs}`",
        f"- train batches seen: {metadata['train']['seen_batches']}",
        f"- val batches seen: {metadata['eval']['seen_batches']}",
        "",
        "## Experts",
        "",
        "| pair | train bal acc | train AUC proxy |",
        "| --- | ---: | ---: |",
    ]
    for expert in experts:
        lines.append(f"| {pair_name(expert.pair)} | {expert.train_bal_acc:.4f} | {expert.train_auc_proxy:.4f} |")
    lines += [
        "",
        "## Best Variants",
        "",
        "| criterion | variant | mIoU | delta mIoU | picture IoU | delta picture IoU | picture->wall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        "| best mIoU | {variant} | {mIoU:.4f} | {delta_mIoU:+.4f} | {picture_iou:.4f} | {delta_picture_iou:+.4f} | {picture_to_wall_frac:.4f} |".format(
            **best_miou
        ),
        "| best picture | {variant} | {mIoU:.4f} | {delta_mIoU:+.4f} | {picture_iou:.4f} | {delta_picture_iou:+.4f} | {picture_to_wall_frac:.4f} |".format(
            **best_picture
        ),
        "",
        "## Top 10 By mIoU",
        "",
        "| variant | mIoU | delta | picture IoU | picture delta | picture->wall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_sorted[:10]:
        lines.append(
            "| {variant} | {mIoU:.4f} | {delta_mIoU:+.4f} | {picture_iou:.4f} | {delta_picture_iou:+.4f} | {picture_to_wall_frac:.4f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This is an offline same-checkpoint readout correction, not a retrained Concerto model.",
        "- Positive deltas indicate that pairwise separability in the frozen decoder feature can be converted into a better 20-way decision.",
        "- If all deltas are near zero or negative, the naive residual-readout form is not sufficient and needs a more constrained or calibrated gate.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {summary_path}", flush=True)
    print(f"[write] {class_path}", flush=True)
    print(f"[write] {md_path}", flush=True)
    print(f"[best-miou] {best_miou}", flush=True)
    print(f"[best-picture] {best_picture}", flush=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.config = str((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    seed_everything(args.seed)
    pairs = parse_pairs(args.class_pairs)
    cfg = load_config(Path(args.config))
    if args.dry_run:
        loader = build_loader(cfg, args.val_split, args.data_root, 1, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] pairs={[pair_name(pair) for pair in pairs]}", flush=True)
        return 0
    model = build_model(cfg, args.weight)
    train_cache, train_meta = collect_train_cache(args, model, cfg, pairs)
    experts = [fit_expert(pair, train_cache, args) for pair in pairs]
    summaries, confusions, eval_meta = evaluate_variants(args, model, cfg, experts)
    write_outputs(args, summaries, confusions, experts, {"train": train_meta, "eval": eval_meta})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
