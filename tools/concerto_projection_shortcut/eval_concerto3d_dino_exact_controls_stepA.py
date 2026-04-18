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
import torch_scatter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_patch_separation_stepA import (
    SCANNET20_CLASS_NAMES,
    build_loader,
    ensure_repo_on_path,
    ensure_segment_aliases,
    load_cfg,
    load_weight,
    move_batch_to_cuda,
    offset2batch,
    repo_root_from_here,
    standardize,
)


NAME_TO_ID = {name: idx for idx, name in enumerate(SCANNET20_CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Control for Step A'': compare DINO target features and Concerto 3D "
            "features on the exact same ScanNet patch subset, with balanced and "
            "bootstrap controls over multiple confused class pairs."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="pretrain-concerto-v1m1-2-large-video")
    parser.add_argument(
        "--weight",
        type=Path,
        default=Path("data/weights/concerto/pretrain-concerto-v1m1-2-large-video.pth"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/concerto_scannet_imagepoint_absmeta"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,counter:cabinet,desk:wall,desk:table,sink:cabinet,door:wall,shower curtain:wall",
        help="Comma-separated positive:negative class-name pairs.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=256)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--max-per-class", type=int, default=12000)
    parser.add_argument("--min-points-per-patch", type=int, default=4)
    parser.add_argument("--majority-threshold", type=float, default=0.6)
    parser.add_argument("--logreg-steps", type=int, default=600)
    parser.add_argument("--logreg-lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"class pair must be positive:negative, got {raw!r}")
        pos_name, neg_name = [part.strip() for part in raw.split(":", 1)]
        if pos_name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {pos_name}")
        if neg_name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {neg_name}")
        pairs.append((NAME_TO_ID[pos_name], NAME_TO_ID[neg_name]))
    if not pairs:
        raise ValueError("no class pairs were provided")
    return pairs


def pair_name(pair: tuple[int, int]) -> str:
    return f"{SCANNET20_CLASS_NAMES[pair[0]]}_vs_{SCANNET20_CLASS_NAMES[pair[1]]}".replace(" ", "_")


@torch.no_grad()
def extract_batch_features(model, batch: dict, target_classes: set[int], min_points: int, majority_threshold: float):
    from pointcept.models.utils.structure import Point

    global_point = Point(
        feat=batch["global_feat"],
        coord=batch["global_coord"],
        origin_coord=batch["global_origin_coord"],
        offset=batch["global_offset"],
        grid_size=batch["grid_size"][0],
    )
    point = model.student.backbone(global_point)
    point = model.up_cast(point)
    point = model.up_cast(point, upcast_level=model.enc2d_upcast_level - model.up_cast_level)
    to_feature = model.pool_corr(point, batch["global_correspondence"])

    data_dict_global_offset = torch.cat([torch.tensor([0], device=point.feat.device), to_feature["offset"]], dim=0)
    enc2d_count = (
        data_dict_global_offset[1 : len(data_dict_global_offset) : model.num_global_view]
        - data_dict_global_offset[0 : len(data_dict_global_offset) - 1 : model.num_global_view]
    )
    enc2d_mask = torch.cat(
        [
            torch.arange(0, c, device=enc2d_count.device) + data_dict_global_offset[i * model.num_global_view]
            for i, c in enumerate(enc2d_count)
        ],
        dim=0,
    )
    batch_points_3d = offset2batch(torch.cumsum(enc2d_count, dim=0))
    correspondence = to_feature["correspondence"][enc2d_mask]
    corr_mask = torch.any(correspondence != torch.tensor([-1, -1], device=correspondence.device), dim=2)
    valid_index = torch.where(corr_mask)
    if valid_index[0].numel() == 0:
        return None

    offset_img_num = torch.cat([torch.tensor([0], device=point.feat.device), torch.cumsum(batch["img_num"], dim=0)])
    batch_index = batch_points_3d[valid_index[0]]
    batch_img_num = offset_img_num[:-1][batch_index]
    feature_index = torch.cat(
        [
            batch_img_num.unsqueeze(-1),
            valid_index[1].unsqueeze(-1),
            correspondence[valid_index],
        ],
        dim=-1,
    ).long()
    feature_index = (
        feature_index[:, 0] * model.patch_h * model.patch_w
        + feature_index[:, 1] * model.patch_h * model.patch_w
        + feature_index[:, 2] * model.patch_w
        + feature_index[:, 3]
    )
    feature_index, inverse_index = torch.unique(feature_index, sorted=True, return_inverse=True)

    encoder_feat = to_feature["feat"][enc2d_mask][valid_index[0]]
    encoder_feat = torch_scatter.scatter_mean(
        encoder_feat,
        inverse_index,
        dim=0,
        dim_size=feature_index.shape[0],
    )
    patch_proj = model.patch_proj(encoder_feat)
    dino = model.ENC2D_forward(batch["images"])
    dino = dino.contiguous().view(-1, dino.shape[-1])
    dino = dino[feature_index]

    labels = batch["global_segment"][enc2d_mask][valid_index[0]].long()
    valid_label = (labels >= 0) & (labels < 20)
    if not valid_label.any():
        return None
    counts = torch.zeros((feature_index.shape[0], 20), device=labels.device, dtype=torch.float32)
    counts.index_add_(0, inverse_index[valid_label], F.one_hot(labels[valid_label], 20).float())
    totals = counts.sum(dim=1)
    top_count, top_label = counts.max(dim=1)
    confidence = top_count / totals.clamp_min(1.0)
    target_mask = torch.zeros_like(top_label, dtype=torch.bool)
    for cls in target_classes:
        target_mask |= top_label == cls
    keep = (totals >= min_points) & (confidence >= majority_threshold) & target_mask
    if not keep.any():
        return None
    return {
        "encoder_pooled": encoder_feat[keep].float().cpu(),
        "patch_proj": patch_proj[keep].float().cpu(),
        "dino_exact": dino[keep].float().cpu(),
        "class_id": top_label[keep].cpu().long(),
        "points": totals[keep].cpu().long(),
        "confidence": confidence[keep].cpu().float(),
    }


def collect_split(args: argparse.Namespace, model, loader, split: str, max_batches: int, target_classes: set[int]):
    features = {"dino_exact": [], "encoder_pooled": [], "patch_proj": []}
    labels: list[torch.Tensor] = []
    class_counts = {cls: 0 for cls in sorted(target_classes)}
    seen_batches = 0
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches >= 0 and batch_idx >= max_batches:
                break
            batch = move_batch_to_cuda(batch)
            out = extract_batch_features(
                model,
                batch,
                target_classes=target_classes,
                min_points=args.min_points_per_patch,
                majority_threshold=args.majority_threshold,
            )
            seen_batches += 1
            if out is None:
                continue
            for cls in sorted(target_classes):
                mask = out["class_id"] == cls
                if not mask.any():
                    continue
                room = args.max_per_class - class_counts[cls]
                if room <= 0:
                    continue
                idx = mask.nonzero(as_tuple=False).flatten()
                if idx.numel() > room:
                    idx = idx[:room]
                for name in features:
                    features[name].append(out[name][idx])
                labels.append(out["class_id"][idx])
                class_counts[cls] += int(idx.numel())
            if (batch_idx + 1) % 20 == 0:
                counts = " ".join(f"{SCANNET20_CLASS_NAMES[k]}={v}" for k, v in class_counts.items())
                print(f"[collect] split={split} batch={batch_idx + 1} {counts}", flush=True)
    if not labels:
        raise RuntimeError(f"no target patch samples collected for split={split}")
    payload = {
        "class_id": torch.cat(labels, dim=0),
        "seen_batches": seen_batches,
        "class_counts": {SCANNET20_CLASS_NAMES[k]: int(v) for k, v in class_counts.items()},
    }
    for name, tensors in features.items():
        payload[name] = torch.cat(tensors, dim=0)
    print(f"[collect] split={split} done counts={payload['class_counts']}", flush=True)
    return payload


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.detach().cpu().long()
    scores = scores.detach().cpu()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())
    return float(auc.item())


def balanced_train_indices(labels: torch.Tensor, seed: int) -> torch.Tensor:
    pos = (labels == 1).nonzero(as_tuple=False).flatten()
    neg = (labels == 0).nonzero(as_tuple=False).flatten()
    n = min(pos.numel(), neg.numel())
    if n == 0:
        raise RuntimeError("empty class in balanced split")
    generator = torch.Generator().manual_seed(seed)
    idx = torch.cat([pos[torch.randperm(pos.numel(), generator=generator)[:n]], neg[torch.randperm(neg.numel(), generator=generator)[:n]]])
    return idx[torch.randperm(idx.numel(), generator=generator)]


def fit_probe(train_x, train_y, val_x, val_y, args: argparse.Namespace, mode: str):
    train_x, val_x = standardize(train_x, val_x)
    if mode == "balanced":
        idx = balanced_train_indices(train_y, args.seed + 13)
        train_x = train_x[idx]
        train_y = train_y[idx]
        pos_weight = None
    elif mode == "weighted":
        pos = int((train_y == 1).sum().item())
        neg = int((train_y == 0).sum().item())
        pos_weight = float(neg / max(pos, 1))
    else:
        pos_weight = None

    train_x = train_x.cuda()
    train_y = train_y.cuda()
    val_x = val_x.cuda()
    val_y_cuda = val_y.cuda()
    weight = torch.zeros(train_x.shape[1], device="cuda", requires_grad=True)
    bias = torch.zeros((), device="cuda", requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=args.logreg_lr, weight_decay=args.weight_decay)
    with torch.enable_grad():
        for _ in range(args.logreg_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = train_x @ weight + bias
            loss = F.binary_cross_entropy_with_logits(
                logits,
                train_y,
                pos_weight=(torch.tensor(pos_weight, device="cuda") if pos_weight is not None else None),
            )
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = (val_x @ weight + bias).detach().cpu()
        pred = (logits >= 0).float()
        val_y_cpu = val_y.detach().cpu().float()
        pos_mask = val_y_cpu == 1
        neg_mask = val_y_cpu == 0
        pos_acc = (pred[pos_mask] == val_y_cpu[pos_mask]).float().mean().item() if pos_mask.any() else float("nan")
        neg_acc = (pred[neg_mask] == val_y_cpu[neg_mask]).float().mean().item() if neg_mask.any() else float("nan")
        return {
            "logits": logits,
            "labels": val_y_cpu,
            "acc": (pred == val_y_cpu).float().mean().item(),
            "balanced_acc": float(np.nanmean([pos_acc, neg_acc])),
            "positive_acc": pos_acc,
            "negative_acc": neg_acc,
            "auc": binary_auc(logits, val_y_cpu),
            "train_samples_probe": int(train_y.numel()),
            "pos_weight": "" if pos_weight is None else float(pos_weight),
        }


def bootstrap_ci(logits: torch.Tensor, labels: torch.Tensor, iters: int, seed: int):
    rng = np.random.default_rng(seed)
    logits_np = logits.numpy()
    labels_np = labels.numpy().astype(np.int64)
    pos = np.where(labels_np == 1)[0]
    neg = np.where(labels_np == 0)[0]
    if len(pos) == 0 or len(neg) == 0 or iters <= 0:
        return {"bal_acc_std": float("nan"), "bal_acc_ci_low": float("nan"), "bal_acc_ci_high": float("nan"), "auc_std": float("nan"), "auc_ci_low": float("nan"), "auc_ci_high": float("nan")}
    bal_accs = []
    aucs = []
    for _ in range(iters):
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)])
        scores = torch.from_numpy(logits_np[idx])
        y = torch.from_numpy(labels_np[idx]).float()
        pred = (scores >= 0).float()
        pos_mask = y == 1
        neg_mask = y == 0
        bal = 0.5 * (
            (pred[pos_mask] == y[pos_mask]).float().mean().item()
            + (pred[neg_mask] == y[neg_mask]).float().mean().item()
        )
        bal_accs.append(bal)
        aucs.append(binary_auc(scores, y))
    return {
        "bal_acc_std": float(np.std(bal_accs, ddof=1)),
        "bal_acc_ci_low": float(np.quantile(bal_accs, 0.025)),
        "bal_acc_ci_high": float(np.quantile(bal_accs, 0.975)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "auc_ci_low": float(np.quantile(aucs, 0.025)),
        "auc_ci_high": float(np.quantile(aucs, 0.975)),
    }


def evaluate_pair(pair, train, val, args: argparse.Namespace) -> list[dict]:
    pos_cls, neg_cls = pair
    train_mask = (train["class_id"] == pos_cls) | (train["class_id"] == neg_cls)
    val_mask = (val["class_id"] == pos_cls) | (val["class_id"] == neg_cls)
    train_y = (train["class_id"][train_mask] == pos_cls).float()
    val_y = (val["class_id"][val_mask] == pos_cls).float()
    rows = []
    for feature in ("dino_exact", "encoder_pooled", "patch_proj"):
        for probe in ("unweighted", "balanced", "weighted"):
            result = fit_probe(train[feature][train_mask], train_y, val[feature][val_mask], val_y, args, probe)
            ci = bootstrap_ci(result.pop("logits"), result.pop("labels"), args.bootstrap_iters, args.seed + len(rows) + 101)
            row = {
                "pair": pair_name(pair),
                "positive_class": SCANNET20_CLASS_NAMES[pos_cls],
                "negative_class": SCANNET20_CLASS_NAMES[neg_cls],
                "feature": feature,
                "probe": probe,
                "train_positive": int((train_y == 1).sum().item()),
                "train_negative": int((train_y == 0).sum().item()),
                "val_positive": int((val_y == 1).sum().item()),
                "val_negative": int((val_y == 0).sum().item()),
            }
            row.update(result)
            row.update(ci)
            rows.append(row)
    return rows


def write_outputs(args: argparse.Namespace, rows: list[dict], metadata: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "concerto3d_dino_exact_controls_stepA.csv"
    fields = [
        "pair",
        "positive_class",
        "negative_class",
        "feature",
        "probe",
        "balanced_acc",
        "bal_acc_std",
        "bal_acc_ci_low",
        "bal_acc_ci_high",
        "auc",
        "auc_std",
        "auc_ci_low",
        "auc_ci_high",
        "positive_acc",
        "negative_acc",
        "acc",
        "train_positive",
        "train_negative",
        "val_positive",
        "val_negative",
        "train_samples_probe",
        "pos_weight",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    md_path = args.output_dir / "concerto3d_dino_exact_controls_stepA.md"
    lines = [
        "# Concerto 3D / DINO Exact-Patch Controls Step A",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- train batches seen: {metadata['train']['seen_batches']}",
        f"- val batches seen: {metadata['val']['seen_batches']}",
        f"- train class counts: {metadata['train']['class_counts']}",
        f"- val class counts: {metadata['val']['class_counts']}",
        f"- bootstrap iters: {args.bootstrap_iters}",
        "",
        "## Results",
        "",
        "| pair | feature | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {pair} | {feature} | {probe} | {balanced_acc:.4f} | "
            "[{bal_acc_ci_low:.4f}, {bal_acc_ci_high:.4f}] | {auc:.4f} | "
            "[{auc_ci_low:.4f}, {auc_ci_high:.4f}] | {val_positive}/{val_negative} |".format(**row)
        )
    lines += [
        "",
        "## Interpretation Guide",
        "- `dino_exact` is the frozen DINO target feature from `model.ENC2D_forward` on the exact same augmented image patches used for the Concerto rows.",
        "- `encoder_pooled` is the Concerto 3D encoder feature pooled to those same patch ids through point-pixel correspondence.",
        "- `patch_proj` is the Concerto enc2d patch projection of `encoder_pooled`.",
        "- `balanced` trains the binary probe on a class-balanced train subset; validation metrics are still reported on all validation rows through balanced accuracy and AUC.",
        "- Confidence intervals bootstrap validation rows with class-stratified resampling.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[write] {csv_path}", flush=True)
    print(f"[write] {md_path}", flush=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ensure_repo_on_path(repo_root)
    seed_everything(args.seed)
    pairs = parse_pairs(args.class_pairs)
    target_classes = {cls for pair in pairs for cls in pair}
    cfg = load_cfg(repo_root, args.config)
    alias_summary = ensure_segment_aliases(args.data_root, [args.train_split, args.val_split])
    print(f"[segment_alias] {alias_summary}", flush=True)
    print(f"[pairs] {[pair_name(pair) for pair in pairs]}", flush=True)

    if args.dry_run:
        train_loader = build_loader(cfg, args.data_root, args.train_split, 1, 0)
        batch = next(iter(train_loader))
        print(f"[dry] keys={sorted(batch.keys())}")
        print(f"[dry] target_classes={[SCANNET20_CLASS_NAMES[i] for i in sorted(target_classes)]}")
        return 0

    from pointcept.models.builder import build_model

    print(f"[info] building model config={args.config}", flush=True)
    model = build_model(cfg.model).cuda().eval()
    load_weight(model, (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight)
    torch.cuda.empty_cache()

    train_loader = build_loader(cfg, args.data_root, args.train_split, args.batch_size, args.num_worker)
    val_loader = build_loader(cfg, args.data_root, args.val_split, args.batch_size, args.num_worker)
    train = collect_split(args, model, train_loader, args.train_split, args.max_train_batches, target_classes)
    val = collect_split(args, model, val_loader, args.val_split, args.max_val_batches, target_classes)
    rows: list[dict] = []
    for pair in pairs:
        rows.extend(evaluate_pair(pair, train, val, args))
    metadata = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pairs": [pair_name(pair) for pair in pairs],
        "train": {"seen_batches": train["seen_batches"], "class_counts": train["class_counts"]},
        "val": {"seen_batches": val["seen_batches"], "class_counts": val["class_counts"]},
    }
    write_outputs(args, rows, metadata)
    for row in rows:
        if row["probe"] == "balanced":
            print(
                "[result] {pair}/{feature}/{probe} bal={balanced_acc:.4f} "
                "ci=[{bal_acc_ci_low:.4f},{bal_acc_ci_high:.4f}] auc={auc:.4f}".format(**row),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
