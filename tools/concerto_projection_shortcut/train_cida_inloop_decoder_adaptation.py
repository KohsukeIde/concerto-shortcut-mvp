#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pointcept.models.utils.structure import Point  # noqa: E402
from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
    parse_pairs,
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CIDA: Confusion-Aware In-Loop Decoder Adaptation. Loads the "
            "origin decoder-probe checkpoint, freezes the encoder, adapts the "
            "decoder/seg head under real ScanNet train augmentation, and anchors "
            "to a frozen base copy via KL and batch predicted-distribution KL."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument(
        "--base-weight",
        type=Path,
        default=None,
        help="Frozen baseline checkpoint for KL anchors/eval comparison. Defaults to --weight.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="cida")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--class-pairs",
        default="picture:wall,desk:table,sink:cabinet,counter:cabinet",
    )
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-train-iters", type=int, default=-1)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-worker", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--weak-class-weight", type=float, default=2.0)
    parser.add_argument("--lambda-pair", type=float, default=0.2)
    parser.add_argument("--lambda-kl", type=float, default=0.05)
    parser.add_argument("--lambda-dist", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--clip-grad", type=float, default=3.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-every-epoch", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    root = args.repo_root.resolve()
    args.config = str((root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    args.weight = (root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    if args.base_weight is None:
        args.base_weight = args.weight
    else:
        args.base_weight = (root / args.base_weight).resolve() if not args.base_weight.is_absolute() else args.base_weight
    args.data_root = (root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path):
    from pointcept.utils.config import Config

    return Config.fromfile(str(path))


def split_dataset_cfg(cfg, split: str, data_root: Path, train: bool):
    ds_cfg = copy.deepcopy(cfg.data.train if train else cfg.data.val)
    ds_cfg.split = split
    ds_cfg.data_root = str(data_root)
    ds_cfg.test_mode = False
    return ds_cfg


def build_loader(cfg, split: str, data_root: Path, batch_size: int, num_worker: int, train: bool):
    from pointcept.datasets.builder import build_dataset
    from pointcept.datasets.utils import point_collate_fn

    ds_cfg = split_dataset_cfg(cfg, split, data_root, train=train)
    dataset = build_dataset(ds_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_worker,
        pin_memory=True,
        drop_last=train,
        collate_fn=lambda batch: point_collate_fn(batch, mix_prob=cfg.mix_prob if train else 0),
    )


def build_model(cfg, weight_path: Path, name: str):
    from pointcept.models.builder import build_model

    model = build_model(cfg.model).cuda()
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    info = model.load_state_dict(cleaned, strict=False)
    print(
        f"[load-{name}] weight={weight_path} missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}",
        flush=True,
    )
    if info.missing_keys:
        print(f"[load-{name}] first missing={info.missing_keys[:8]}", flush=True)
    if info.unexpected_keys:
        print(f"[load-{name}] first unexpected={info.unexpected_keys[:8]}", flush=True)
    return model


def freeze_for_cida(model) -> dict[str, int]:
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model.backbone, "dec"):
        for p in model.backbone.dec.parameters():
            p.requires_grad = True
    for p in model.seg_head.parameters():
        p.requires_grad = True
    counts = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dec_trainable": sum(p.numel() for p in getattr(model.backbone, "dec", torch.nn.Module()).parameters() if p.requires_grad),
        "head_trainable": sum(p.numel() for p in model.seg_head.parameters() if p.requires_grad),
    }
    return counts


def set_cida_train_mode(model) -> None:
    # Match Pointcept's standard trainer behavior: frozen encoder parameters have
    # requires_grad=False, but modules remain in train mode during decoder-probe
    # training. For PTv3 decoder adaptation this avoids diverging from the
    # already validated decoder-probe path.
    model.train()


def move_to_cuda(input_dict: dict) -> dict:
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.cuda(non_blocking=True)
    return input_dict


def forward_logits(model, input_dict: dict):
    point = Point(input_dict)
    point = model.backbone(point)
    if isinstance(point, Point):
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        feat = point.feat
    else:
        feat = point
    return model.seg_head(feat)


def eval_tensors(logits: torch.Tensor, labels: torch.Tensor, batch: dict):
    inverse = batch.get("inverse")
    origin_segment = batch.get("origin_segment")
    if inverse is not None and origin_segment is not None:
        return logits[inverse], origin_segment.long()
    return logits, labels


def parse_names(text: str) -> list[int]:
    ids = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown class name: {name}")
        ids.append(NAME_TO_ID[name])
    return ids


def sample_weights(labels: torch.Tensor, weak_classes: list[int], weak_weight: float, num_classes: int) -> torch.Tensor:
    weights = torch.ones(num_classes, dtype=torch.float32, device=labels.device)
    for cls in weak_classes:
        weights[cls] = weak_weight
    safe_labels = labels.clamp(min=0, max=num_classes - 1)
    return weights[safe_labels]


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weak_classes: list[int], weak_weight: float, ignore_index: int, num_classes: int):
    valid = labels != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits[valid], labels[valid], reduction="none")
    weights = sample_weights(labels[valid], weak_classes, weak_weight, num_classes)
    return (ce * weights).sum() / weights.sum().clamp_min(1e-6)


def pairwise_aux_loss(logits: torch.Tensor, labels: torch.Tensor, pairs: list[tuple[int, int]], ignore_index: int):
    losses = []
    for a, b in pairs:
        mask = ((labels == a) | (labels == b)) & (labels != ignore_index)
        if not mask.any():
            continue
        pair_logits = logits[mask][:, [a, b]]
        pair_target = (labels[mask] == b).long()
        losses.append(F.cross_entropy(pair_logits, pair_target))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def pointwise_kl(logits: torch.Tensor, base_logits: torch.Tensor, labels: torch.Tensor, ignore_index: int, temperature: float):
    valid = labels != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    t = temperature
    base_prob = (base_logits[valid] / t).softmax(dim=1)
    log_prob = (logits[valid] / t).log_softmax(dim=1)
    return F.kl_div(log_prob, base_prob, reduction="batchmean") * (t * t)


def distribution_kl(logits: torch.Tensor, base_logits: torch.Tensor, labels: torch.Tensor, ignore_index: int, temperature: float):
    valid = labels != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    t = temperature
    p = (logits[valid] / t).softmax(dim=1).mean(dim=0).clamp_min(1e-8)
    p0 = (base_logits[valid] / t).softmax(dim=1).mean(dim=0).clamp_min(1e-8)
    kl_p_p0 = (p * (p.log() - p0.log())).sum()
    kl_p0_p = (p0 * (p0.log() - p.log())).sum()
    return 0.5 * (kl_p_p0 + kl_p0_p)


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    valid = target != ignore_index
    pred = pred[valid].long()
    target = target[valid].long()
    in_range = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[in_range]
    target = target[in_range]
    flat = target * num_classes + pred
    confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def summarize_confusion(conf: np.ndarray):
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
    }


def evaluate(model, base_model, cfg, args, label: str):
    model.eval()
    base_model.eval()
    loader = build_loader(cfg, args.val_split, args.data_root, args.batch_size, args.num_worker, train=False)
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")
    base_conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cuda")
    seen = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            logits = forward_logits(model, batch)
            base_logits = forward_logits(base_model, batch)
            labels = batch["segment"].long()
            logits_e, labels_e = eval_tensors(logits, labels, batch)
            base_logits_e, _ = eval_tensors(base_logits, labels, batch)
            update_confusion(conf, logits_e.argmax(dim=1), labels_e, num_classes, ignore_index)
            update_confusion(base_conf, base_logits_e.argmax(dim=1), labels_e, num_classes, ignore_index)
            valid = labels != ignore_index
            if valid.any():
                loss_sum += float(F.cross_entropy(logits[valid], labels[valid]).item())
            seen += 1
            if seen % 25 == 0:
                summary = summarize_confusion(conf.detach().cpu().numpy())
                print(f"[eval:{label}] batches={seen} mIoU={summary['mIoU']:.4f}", flush=True)
    summary = summarize_confusion(conf.detach().cpu().numpy())
    base_summary = summarize_confusion(base_conf.detach().cpu().numpy())
    return summary, base_summary, conf.detach().cpu().numpy(), base_conf.detach().cpu().numpy(), seen, loss_sum / max(seen, 1)


def confusion_frac(conf: np.ndarray, a: int, b: int) -> float:
    denom = max(float(conf[a].sum()), 1.0)
    return float(conf[a, b] / denom)


def row_for_eval(tag: str, summary: dict, base_summary: dict, conf: np.ndarray, base_conf: np.ndarray) -> dict:
    picture = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    weak = [NAME_TO_ID[x] for x in ["picture", "counter", "desk", "sink", "cabinet"]]
    return {
        "tag": tag,
        "mIoU": summary["mIoU"],
        "delta_mIoU": summary["mIoU"] - base_summary["mIoU"],
        "mAcc": summary["mAcc"],
        "allAcc": summary["allAcc"],
        "weak_mIoU": float(np.mean(summary["iou"][weak])),
        "weak_delta_mIoU": float(np.mean(summary["iou"][weak] - base_summary["iou"][weak])),
        "picture_iou": summary["iou"][picture],
        "picture_delta_iou": summary["iou"][picture] - base_summary["iou"][picture],
        "picture_to_wall": confusion_frac(conf, picture, wall),
        "base_picture_to_wall": confusion_frac(base_conf, picture, wall),
        "picture_pred_count": summary["pred_sum"][picture],
        "base_picture_pred_count": base_summary["pred_sum"][picture],
        "counter_delta_iou": summary["iou"][NAME_TO_ID["counter"]] - base_summary["iou"][NAME_TO_ID["counter"]],
        "desk_delta_iou": summary["iou"][NAME_TO_ID["desk"]] - base_summary["iou"][NAME_TO_ID["desk"]],
        "sink_delta_iou": summary["iou"][NAME_TO_ID["sink"]] - base_summary["iou"][NAME_TO_ID["sink"]],
        "cabinet_delta_iou": summary["iou"][NAME_TO_ID["cabinet"]] - base_summary["iou"][NAME_TO_ID["cabinet"]],
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in row.items()})


def main() -> int:
    args = resolve_args(parse_args())
    seed_everything(args.seed)
    cfg = load_config(Path(args.config))
    cfg.data.train.split = args.train_split
    cfg.data.val.split = args.val_split
    cfg.data.train.data_root = str(args.data_root)
    cfg.data.val.data_root = str(args.data_root)
    if args.dry_run:
        loader = build_loader(cfg, args.train_split, args.data_root, 1, 0, train=True)
        batch = next(iter(loader))
        print(f"[dry] train keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] pairs={args.class_pairs}", flush=True)
        print(f"[dry] weak={args.weak_classes}", flush=True)
        return 0

    use_base_anchor = args.lambda_kl > 0 or args.lambda_dist > 0 or args.eval_only
    model = build_model(cfg, args.weight, "cida").cuda()
    base_model = None
    if use_base_anchor:
        base_model = build_model(cfg, args.base_weight, "base").cuda().eval()
        for p in base_model.parameters():
            p.requires_grad = False
    counts = freeze_for_cida(model)
    print(f"[params] {counts}", flush=True)
    pairs = parse_pairs(args.class_pairs)
    weak_classes = parse_names(args.weak_classes)
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)

    if args.eval_only:
        summary, base_summary, conf, base_conf, seen, eval_loss = evaluate(model, base_model, cfg, args, "eval-only")
        eval_row = row_for_eval(args.tag, summary, base_summary, conf, base_conf)
        fields = [
            "tag",
            "mIoU",
            "delta_mIoU",
            "mAcc",
            "allAcc",
            "weak_mIoU",
            "weak_delta_mIoU",
            "picture_iou",
            "picture_delta_iou",
            "picture_to_wall",
            "base_picture_to_wall",
            "picture_pred_count",
            "base_picture_pred_count",
            "counter_delta_iou",
            "desk_delta_iou",
            "sink_delta_iou",
            "cabinet_delta_iou",
        ]
        write_csv(args.output_dir / "cida_eval.csv", [eval_row], fields)
        metadata = {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "params": counts,
            "global_step": 0,
            "val_seen_batches": seen,
            "eval_loss": eval_loss,
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / "cida_inloop_decoder_adaptation.md").write_text(
            "\n".join(
                [
                    "# CIDA In-Loop Decoder Adaptation Eval",
                    "",
                    f"- tag: `{args.tag}`",
                    f"- weight: `{args.weight}`",
                    f"- base weight: `{args.base_weight}`",
                    f"- val seen batches: `{seen}`",
                    "",
                    "| metric | value |",
                    "| --- | ---: |",
                    f"| mIoU | {eval_row['mIoU']:.6f} |",
                    f"| delta mIoU | {eval_row['delta_mIoU']:+.6f} |",
                    f"| weak mIoU | {eval_row['weak_mIoU']:.6f} |",
                    f"| weak delta mIoU | {eval_row['weak_delta_mIoU']:+.6f} |",
                    f"| picture IoU | {eval_row['picture_iou']:.6f} |",
                    f"| picture delta IoU | {eval_row['picture_delta_iou']:+.6f} |",
                    f"| picture -> wall | {eval_row['picture_to_wall']:.6f} |",
                    f"| base picture -> wall | {eval_row['base_picture_to_wall']:.6f} |",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[done] eval-only wrote {args.output_dir}", flush=True)
        return 0

    train_loader = build_loader(cfg, args.train_split, args.data_root, args.batch_size, args.num_worker, train=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    total_iters = args.epochs * len(train_loader)
    if args.max_train_iters > 0:
        total_iters = min(total_iters, args.max_train_iters)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_iters, 1), eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    history = []
    global_step = 0
    stop = False
    for epoch in range(args.epochs):
        set_cida_train_mode(model)
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_iters > 0 and global_step >= args.max_train_iters:
                stop = True
                break
            batch = move_to_cuda(batch)
            labels = batch["segment"].long()
            optimizer.zero_grad(set_to_none=True)
            if base_model is not None:
                with torch.inference_mode():
                    base_logits = forward_logits(base_model, batch)
            else:
                base_logits = None
            with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.float16):
                logits = forward_logits(model, batch)
                loss_ce = weighted_ce(logits, labels, weak_classes, args.weak_class_weight, ignore_index, num_classes)
                loss_pair = pairwise_aux_loss(logits, labels, pairs, ignore_index)
                if base_model is not None:
                    loss_kl = pointwise_kl(logits, base_logits, labels, ignore_index, args.temperature)
                    loss_dist = distribution_kl(logits, base_logits, labels, ignore_index, args.temperature)
                else:
                    loss_kl = logits.sum() * 0.0
                    loss_dist = logits.sum() * 0.0
                loss = loss_ce + args.lambda_pair * loss_pair + args.lambda_kl * loss_kl + args.lambda_dist * loss_dist
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if global_step == 1 or global_step % 100 == 0:
                with torch.no_grad():
                    valid = labels != ignore_index
                    pred = logits[valid].argmax(dim=1) if valid.any() else torch.empty(0, device=labels.device, dtype=torch.long)
                    base_pred = base_logits[valid].argmax(dim=1) if (base_logits is not None and valid.any()) else torch.empty(0, device=labels.device, dtype=torch.long)
                    target = labels[valid]
                    train_acc = float((pred == target).float().mean().item()) if target.numel() else 0.0
                    base_acc = float((base_pred == target).float().mean().item()) if (base_logits is not None and target.numel()) else 0.0
                    p = logits[valid].softmax(dim=1).mean(dim=0) if valid.any() else torch.zeros(num_classes, device=labels.device)
                    if base_logits is not None and valid.any():
                        p0 = base_logits[valid].softmax(dim=1).mean(dim=0)
                        hist_l1 = float((p - p0).abs().sum().item())
                    else:
                        hist_l1 = 0.0
                row = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "loss": float(loss.detach().item()),
                    "ce": float(loss_ce.detach().item()),
                    "pair": float(loss_pair.detach().item()),
                    "kl": float(loss_kl.detach().item()),
                    "dist": float(loss_dist.detach().item()),
                    "base_acc": base_acc,
                    "train_acc": train_acc,
                    "hist_l1": hist_l1,
                }
                history.append(row)
                print(
                    f"[train] epoch={row['epoch']} step={row['step']} loss={row['loss']:.4f} "
                    f"ce={row['ce']:.4f} pair={row['pair']:.4f} kl={row['kl']:.4f} dist={row['dist']:.4f} "
                    f"base={row['base_acc']:.4f} acc={row['train_acc']:.4f} hist_l1={row['hist_l1']:.4f}",
                    flush=True,
                )
        if stop:
            break
        if args.eval_every_epoch > 0 and (epoch + 1) % args.eval_every_epoch == 0:
            summary, base_summary, _, _, seen, _ = evaluate(model, base_model or model, cfg, args, f"e{epoch + 1}")
            print(
                f"[eval-epoch] epoch={epoch + 1} seen={seen} mIoU={summary['mIoU']:.4f} "
                f"delta={summary['mIoU'] - base_summary['mIoU']:+.4f}",
                flush=True,
            )
    if base_model is None:
        base_model = build_model(cfg, args.base_weight, "base-eval").cuda().eval()
        for p in base_model.parameters():
            p.requires_grad = False
    summary, base_summary, conf, base_conf, seen, eval_loss = evaluate(model, base_model, cfg, args, "final")
    eval_row = row_for_eval(args.tag, summary, base_summary, conf, base_conf)
    fields = [
        "tag",
        "mIoU",
        "delta_mIoU",
        "mAcc",
        "allAcc",
        "weak_mIoU",
        "weak_delta_mIoU",
        "picture_iou",
        "picture_delta_iou",
        "picture_to_wall",
        "base_picture_to_wall",
        "picture_pred_count",
        "base_picture_pred_count",
        "counter_delta_iou",
        "desk_delta_iou",
        "sink_delta_iou",
        "cabinet_delta_iou",
    ]
    write_csv(args.output_dir / "cida_eval.csv", [eval_row], fields)
    write_csv(
        args.output_dir / "cida_train_history.csv",
        history,
        ["epoch", "step", "lr", "loss", "ce", "pair", "kl", "dist", "base_acc", "train_acc", "hist_l1"],
    )
    if args.save_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": args.epochs,
                "global_step": global_step,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            args.output_dir / "model_cida_last.pth",
        )
    metadata = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "params": counts,
        "global_step": global_step,
        "val_seen_batches": seen,
        "eval_loss": eval_loss,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# CIDA In-Loop Decoder Adaptation",
        "",
        "## Setup",
        f"- tag: `{args.tag}`",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- data root: `{args.data_root}`",
        f"- epochs requested: `{args.epochs}`",
        f"- global steps: `{global_step}`",
        f"- batch size: `{args.batch_size}`",
        f"- trainable params: `{counts['trainable']}`",
        f"- lambda pair/KL/dist: `{args.lambda_pair}` / `{args.lambda_kl}` / `{args.lambda_dist}`",
        f"- weak class weight: `{args.weak_class_weight}`",
        f"- temperature: `{args.temperature}`",
        "",
        "## Final ScanNet Val",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| mIoU | {eval_row['mIoU']:.6f} |",
        f"| delta mIoU | {eval_row['delta_mIoU']:+.6f} |",
        f"| weak mIoU | {eval_row['weak_mIoU']:.6f} |",
        f"| weak delta mIoU | {eval_row['weak_delta_mIoU']:+.6f} |",
        f"| picture IoU | {eval_row['picture_iou']:.6f} |",
        f"| picture delta IoU | {eval_row['picture_delta_iou']:+.6f} |",
        f"| picture -> wall | {eval_row['picture_to_wall']:.6f} |",
        f"| base picture -> wall | {eval_row['base_picture_to_wall']:.6f} |",
        f"| picture pred count | {eval_row['picture_pred_count']:.0f} |",
        f"| base picture pred count | {eval_row['base_picture_pred_count']:.0f} |",
        "",
        "## Last Training Logs",
        "",
        "| epoch | step | loss | CE | pair | KL | dist | base acc | train acc | hist L1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history[-10:]:
        lines.append(
            f"| {row['epoch']} | {row['step']} | {row['loss']:.4f} | {row['ce']:.4f} | "
            f"{row['pair']:.4f} | {row['kl']:.4f} | {row['dist']:.4f} | "
            f"{row['base_acc']:.4f} | {row['train_acc']:.4f} | {row['hist_l1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{args.output_dir / 'cida_eval.csv'}`",
            f"- `{args.output_dir / 'cida_train_history.csv'}`",
            f"- `{args.output_dir / 'metadata.json'}`",
            f"- `{args.output_dir / 'model_cida_last.pth'}`",
            "",
        ]
    )
    (args.output_dir / "cida_inloop_decoder_adaptation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
