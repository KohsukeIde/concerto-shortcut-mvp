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
class TrainCache:
    feat: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    class_counts: dict[str, int]
    seen_batches: int


class TopKPairwiseReranker(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int,
        class_embed_dim: int,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
        logit_mean: torch.Tensor,
        logit_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("feat_mean", feat_mean.float())
        self.register_buffer("feat_std", feat_std.float().clamp_min(1e-6))
        self.register_buffer("logit_mean", logit_mean.float())
        self.register_buffer("logit_std", logit_std.float().clamp_min(1e-6))
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.logit_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + class_embed_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        feat = (feat.float() - self.feat_mean.to(feat.device)) / self.feat_std.to(feat.device)
        logits = logits.float()
        logit_z = (logits - self.logit_mean.to(logits.device)) / self.logit_std.to(logits.device)
        prob = logits.softmax(dim=1)
        top = logits.max(dim=1, keepdim=True).values
        margin_to_top = (logits - top) / self.logit_std.to(logits.device)

        feat_h = self.feat_proj(feat).unsqueeze(1).expand(-1, self.num_classes, -1)
        cls_h = self.class_embed(torch.arange(self.num_classes, device=feat.device)).unsqueeze(0).expand(
            feat.shape[0], -1, -1
        )
        log_h = self.logit_proj(torch.stack([logit_z, prob, margin_to_top], dim=-1))
        return self.scorer(torch.cat([feat_h, cls_h, log_h], dim=-1)).squeeze(-1)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Top-K confusion-graph reranking decoder for the frozen "
            "concerto_base_origin decoder-probe checkpoint. The base 20-way "
            "logits are fixed; a tiny candidate-local reranker only changes "
            "top-K classes and their confusion-neighbor classes."
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
    parser.add_argument("--max-train-points", type=int, default=600000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--train-top-k", type=int, default=3)
    parser.add_argument("--eval-top-ks", default="2,3,5")
    parser.add_argument("--lambdas", default="0.01,0.02,0.05,0.1,0.2,0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--class-embed-dim", type=int, default=32)
    parser.add_argument("--rerank-steps", type=int, default=2000)
    parser.add_argument("--rerank-lr", type=float, default=3e-4)
    parser.add_argument("--rerank-weight-decay", type=float, default=1e-3)
    parser.add_argument("--sample-batch-size", type=int, default=8192)
    parser.add_argument("--residual-l2", type=float, default=1e-3)
    parser.add_argument("--focus-class-weight", type=float, default=3.0)
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


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def build_neighbor_mask(pairs: list[tuple[int, int]], num_classes: int) -> torch.Tensor:
    mask = torch.zeros((num_classes, num_classes), dtype=torch.bool)
    for a, b in pairs:
        mask[a, b] = True
        mask[b, a] = True
    return mask


def build_candidate_mask(logits: torch.Tensor, neighbor_mask: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = min(top_k, logits.shape[1])
    top_idx = logits.topk(k=top_k, dim=1).indices
    candidate = torch.zeros_like(logits, dtype=torch.bool)
    candidate.scatter_(1, top_idx, True)
    neighbor_mask = neighbor_mask.to(logits.device)
    candidate |= neighbor_mask[top_idx].any(dim=1)
    return candidate


def center_candidate_scores(scores: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    count = candidate.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (scores.masked_fill(~candidate, 0.0).sum(dim=1, keepdim=True) / count)
    return (scores - mean).masked_fill(~candidate, 0.0)


def collect_train_cache(args: argparse.Namespace, model, cfg, num_classes: int) -> TrainCache:
    feats: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    class_counts = {idx: 0 for idx in range(num_classes)}
    total_points = 0
    loader = build_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker)
    seen_batches = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_train_batches >= 0 and batch_idx >= args.max_train_batches:
                break
            if total_points >= args.max_train_points:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, _ = forward_features(model, batch)
            seen_batches += 1
            valid = (labels >= 0) & (labels < num_classes)
            if not valid.any():
                continue
            keep_indices = []
            for cls in range(num_classes):
                room = args.max_per_class - class_counts[cls]
                if room <= 0:
                    continue
                idx = ((labels == cls) & valid).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                remaining_total = args.max_train_points - total_points - sum(i.numel() for i in keep_indices)
                if remaining_total <= 0:
                    break
                cap = min(room, remaining_total)
                if idx.numel() > cap:
                    idx = idx[:cap]
                keep_indices.append(idx)
                class_counts[cls] += int(idx.numel())
            if keep_indices:
                keep = torch.cat(keep_indices, dim=0)
                feats.append(feat[keep].detach().cpu())
                logits_list.append(logits[keep].detach().cpu())
                labels_list.append(labels[keep].detach().cpu())
                total_points += int(keep.numel())
            if (batch_idx + 1) % 25 == 0:
                counts = " ".join(f"{SCANNET20_CLASS_NAMES[k]}={v}" for k, v in class_counts.items() if v)
                print(f"[collect-train] batch={batch_idx + 1} total={total_points} {counts}", flush=True)
    if not labels_list:
        raise RuntimeError("no train features collected")
    payload = TrainCache(
        feat=torch.cat(feats, dim=0),
        logits=torch.cat(logits_list, dim=0),
        labels=torch.cat(labels_list, dim=0).long(),
        class_counts={SCANNET20_CLASS_NAMES[k]: int(v) for k, v in class_counts.items()},
        seen_batches=seen_batches,
    )
    print(
        f"[collect-train] done points={payload.labels.numel()} seen_batches={seen_batches} "
        f"counts={payload.class_counts}",
        flush=True,
    )
    return payload


def build_class_weights(labels: torch.Tensor, pairs: list[tuple[int, int]], weight: float, num_classes: int) -> torch.Tensor:
    focus = sorted({cls for pair in pairs for cls in pair})
    weights = torch.ones(num_classes, dtype=torch.float32)
    for cls in focus:
        weights[cls] = weight
    return weights[labels].float()


def train_reranker(
    args: argparse.Namespace,
    cache: TrainCache,
    pairs: list[tuple[int, int]],
    num_classes: int,
) -> tuple[TopKPairwiseReranker, dict]:
    device = torch.device("cuda")
    feat = cache.feat.to(device)
    logits = cache.logits.to(device)
    labels = cache.labels.to(device)
    neighbor_mask = build_neighbor_mask(pairs, num_classes).to(device)
    candidate = build_candidate_mask(logits, neighbor_mask, args.train_top_k)
    valid = candidate.gather(1, labels[:, None]).squeeze(1)
    coverage = float(valid.float().mean().item())
    valid_idx = valid.nonzero(as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        raise RuntimeError("no training examples have their ground-truth class in the candidate set")
    feat = feat[valid_idx]
    logits = logits[valid_idx]
    labels = labels[valid_idx]
    candidate = candidate[valid_idx]
    sample_weights = build_class_weights(labels.detach().cpu(), pairs, args.focus_class_weight, num_classes).to(device)
    feat_mean = feat.mean(dim=0, keepdim=True).detach().cpu()
    feat_std = feat.std(dim=0, keepdim=True).clamp_min(1e-6).detach().cpu()
    logit_mean = logits.mean(dim=0, keepdim=True).detach().cpu()
    logit_std = logits.std(dim=0, keepdim=True).clamp_min(1e-6).detach().cpu()
    model = TopKPairwiseReranker(
        feat_dim=feat.shape[1],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        class_embed_dim=args.class_embed_dim,
        feat_mean=feat_mean,
        feat_std=feat_std,
        logit_mean=logit_mean,
        logit_std=logit_std,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.rerank_lr, weight_decay=args.rerank_weight_decay)
    n = labels.numel()
    history = []
    for step in range(args.rerank_steps):
        idx = torch.randint(0, n, (min(args.sample_batch_size, n),), device=device)
        feat_b = feat[idx]
        logits_b = logits[idx]
        labels_b = labels[idx]
        cand_b = candidate[idx]
        weights_b = sample_weights[idx]
        scores = model(feat_b, logits_b)
        delta = center_candidate_scores(scores, cand_b)
        final = (logits_b + delta).masked_fill(~cand_b, -1e9)
        loss_vec = F.cross_entropy(final, labels_b, reduction="none")
        loss = (loss_vec * weights_b).sum() / weights_b.sum().clamp_min(1e-6)
        if args.residual_l2 > 0:
            loss = loss + args.residual_l2 * (delta[cand_b].pow(2).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step + 1) % max(args.rerank_steps // 10, 1) == 0 or step == 0:
            with torch.no_grad():
                base_pred = logits_b.masked_fill(~cand_b, -1e9).argmax(dim=1)
                pred = final.argmax(dim=1)
                base_acc = (base_pred == labels_b).float().mean().item()
                acc = (pred == labels_b).float().mean().item()
                delta_rms = delta[cand_b].pow(2).mean().sqrt().item()
            row = {
                "step": step + 1,
                "loss": float(loss.item()),
                "candidate_base_acc": base_acc,
                "candidate_rerank_acc": acc,
                "delta_rms": delta_rms,
            }
            history.append(row)
            print(
                f"[rerank] step={row['step']} loss={row['loss']:.4f} "
                f"base_acc={base_acc:.4f} acc={acc:.4f} delta_rms={delta_rms:.4f}",
                flush=True,
            )
    meta = {
        "candidate_train_coverage": coverage,
        "valid_train_points": int(n),
        "history": history,
    }
    return model, meta


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    valid = target != ignore_index
    pred = pred[valid].long()
    target = target[valid].long()
    in_range = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[in_range]
    target = target[in_range]
    flat = target * num_classes + pred
    confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


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


@torch.no_grad()
def apply_rerank(
    reranker: TopKPairwiseReranker,
    feat: torch.Tensor,
    logits: torch.Tensor,
    neighbor_mask: torch.Tensor,
    top_k: int,
    lam: float,
) -> torch.Tensor:
    candidate = build_candidate_mask(logits, neighbor_mask, top_k)
    scores = reranker(feat, logits)
    delta = center_candidate_scores(scores, candidate)
    corrected = logits.clone()
    corrected[candidate] = corrected[candidate] + lam * delta[candidate]
    return corrected


def evaluate_variants(args: argparse.Namespace, model, cfg, reranker: TopKPairwiseReranker, pairs: list[tuple[int, int]]):
    names = list(cfg.data.names)
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    top_ks = parse_int_list(args.eval_top_ks)
    lambdas = parse_float_list(args.lambdas)
    variants = [("base", 0, 0.0)]
    variants.extend((f"topk{k}_lam{str(lam).replace('.', 'p')}", k, lam) for k in top_ks for lam in lambdas)
    confusions = {name: torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda") for name, _, _ in variants}
    neighbor_mask = build_neighbor_mask(pairs, num_classes).cuda()
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker)
    seen_batches = 0
    reranker.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            feat, logits, labels, batch = forward_features(model, batch)
            target = labels
            inverse = batch.get("inverse")
            origin_segment = batch.get("origin_segment")
            for variant_name, top_k, lam in variants:
                if variant_name == "base":
                    pred = logits.argmax(dim=1)
                else:
                    corrected = apply_rerank(reranker, feat, logits, neighbor_mask, top_k, lam)
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


def row_for_variant(variant: str, summary: dict, conf: np.ndarray, base: dict) -> dict:
    picture_id = NAME_TO_ID["picture"]
    wall_id = NAME_TO_ID["wall"]
    picture_target = max(summary["target_sum"][picture_id], 1.0)
    wall_target = max(summary["target_sum"][wall_id], 1.0)
    return {
        "variant": variant,
        "mIoU": summary["mIoU"],
        "mAcc": summary["mAcc"],
        "allAcc": summary["allAcc"],
        "delta_mIoU": summary["mIoU"] - base["mIoU"],
        "picture_iou": summary["iou"][picture_id],
        "delta_picture_iou": summary["iou"][picture_id] - base["iou"][picture_id],
        "picture_to_wall_frac": conf[picture_id, wall_id] / picture_target,
        "wall_to_picture_frac": conf[wall_id, picture_id] / wall_target,
        "counter_iou": summary["iou"][NAME_TO_ID["counter"]],
        "desk_iou": summary["iou"][NAME_TO_ID["desk"]],
        "sink_iou": summary["iou"][NAME_TO_ID["sink"]],
    }


def write_outputs(args: argparse.Namespace, summaries: dict, confusions: dict, reranker: TopKPairwiseReranker, metadata: dict):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = summaries["base"]["names"]
    base = summaries["base"]
    summary_path = args.output_dir / "topk_pairwise_rerank_summary.csv"
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
        "counter_iou",
        "desk_iou",
        "sink_iou",
    ]
    rows = []
    for variant, summary in summaries.items():
        conf = confusions[variant].detach().cpu().numpy()
        rows.append(row_for_variant(variant, summary, conf, base))
    rows_sorted = sorted(rows, key=lambda r: (r["mIoU"], r["picture_iou"]), reverse=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in row.items()})

    class_path = args.output_dir / "topk_pairwise_rerank_class_metrics.csv"
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

    model_path = args.output_dir / "topk_pairwise_reranker.pt"
    torch.save({"state_dict": reranker.state_dict(), "metadata": metadata}, model_path)

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    md_path = args.output_dir / "topk_pairwise_rerank_decoder.md"
    best_miou = rows_sorted[0]
    best_picture = sorted(rows, key=lambda r: (r["picture_iou"], r["mIoU"]), reverse=True)[0]
    lines = [
        "# Top-K Pairwise Re-ranking Decoder",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- class pairs: `{args.class_pairs}`",
        f"- train top-k: `{args.train_top_k}`",
        f"- eval top-k values: `{args.eval_top_ks}`",
        f"- lambdas: `{args.lambdas}`",
        f"- train batches seen: {metadata['train']['seen_batches']}",
        f"- train candidate coverage: {metadata['reranker']['candidate_train_coverage']:.4f}",
        f"- train valid points: {metadata['reranker']['valid_train_points']}",
        f"- val batches seen: {metadata['eval']['seen_batches']}",
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
        "## Training Trace",
        "",
        "| step | loss | candidate base acc | candidate rerank acc | delta rms |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metadata["reranker"]["history"]:
        lines.append(
            f"| {row['step']} | {row['loss']:.4f} | {row['candidate_base_acc']:.4f} | "
            f"{row['candidate_rerank_acc']:.4f} | {row['delta_rms']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This is an offline same-checkpoint readout correction, not a retrained Concerto model.",
        "- The reranker is candidate-local: only top-K base classes and confusion-graph neighbors can change.",
        "- Residual scores are centered within the candidate set so the correction acts as a local reranking term rather than a global class-prior shift.",
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
        print(f"[dry] eval_top_ks={parse_int_list(args.eval_top_ks)} lambdas={parse_float_list(args.lambdas)}", flush=True)
        return 0
    model = build_model(cfg, args.weight)
    num_classes = int(cfg.data.num_classes)
    train_cache = collect_train_cache(args, model, cfg, num_classes)
    reranker, reranker_meta = train_reranker(args, train_cache, pairs, num_classes)
    summaries, confusions, eval_meta = evaluate_variants(args, model, cfg, reranker, pairs)
    write_outputs(
        args,
        summaries,
        confusions,
        reranker,
        {
            "train": {
                "seen_batches": train_cache.seen_batches,
                "class_counts": train_cache.class_counts,
                "num_points": int(train_cache.labels.numel()),
            },
            "reranker": reranker_meta,
            "eval": eval_meta,
            "pairs": [pair_name(pair) for pair in pairs],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
