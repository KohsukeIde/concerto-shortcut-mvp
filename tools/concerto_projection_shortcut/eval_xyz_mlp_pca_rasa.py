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
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    picture_to_wall,
    summarize_confusion,
    update_confusion,
    weak_mean,
)


@dataclass
class PointCache:
    feat: torch.Tensor
    logits: torch.Tensor
    coord: torch.Tensor
    label: torch.Tensor
    class_counts: dict[str, int]
    seen_batches: int


class XYZMLP(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def hidden(self, xyz: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.fc1(xyz)), inplace=True)
        x = F.relu(self.norm2(self.fc2(x)), inplace=True)
        return x

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(xyz))


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task-conditioned xyz nuisance probe for the Concerto origin "
            "decoder features. Trains an xyz-only MLP on ScanNet labels, "
            "compresses its hidden features to a 2D PCA target, predicts that "
            "target from Concerto features, then evaluates RASA-style removal "
            "and split/add-back variants."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py")
    parser.add_argument("--weight", type=Path, default=Path("data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_rasa_reservoir"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("tools/concerto_projection_shortcut/results_xyz_mlp_pca_rasa_reservoir"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=-1)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-train-points", type=int, default=1200000)
    parser.add_argument("--max-val-points", type=int, default=2000000)
    parser.add_argument("--max-train-per-class", type=int, default=60000)
    parser.add_argument("--val-sampling", choices=("natural", "reservoir", "balanced"), default="reservoir")
    parser.add_argument("--xyz-hidden-dim", type=int, default=128)
    parser.add_argument("--xyz-epochs", type=int, default=30)
    parser.add_argument("--xyz-batch-size", type=int, default=65536)
    parser.add_argument("--xyz-lr", type=float, default=1e-3)
    parser.add_argument("--xyz-weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--classifier-ridge", type=float, default=1e-2)
    parser.add_argument("--betas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--addback-lambdas", default="0.25,0.5,1.0")
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--seed", type=int, default=20260424)
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
        raise ValueError("no weak classes provided")
    return out


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
    label = batch["segment"].long()
    if feat.shape[0] != logits.shape[0] or feat.shape[0] != label.shape[0]:
        raise RuntimeError(f"shape mismatch feat={feat.shape} logits={logits.shape} labels={label.shape}")
    return feat, logits, label, batch


def scene_normalized_xyz(batch: dict) -> torch.Tensor:
    coord = batch["coord"].float()
    cmin = coord.min(dim=0, keepdim=True).values
    cmax = coord.max(dim=0, keepdim=True).values
    span = (cmax - cmin).clamp_min(1e-4)
    return (coord - cmin) / span


def eval_tensors_with_coord(feat: torch.Tensor, logits: torch.Tensor, label: torch.Tensor, batch: dict):
    xyz = scene_normalized_xyz(batch)
    inverse = batch.get("inverse")
    origin_segment = batch.get("origin_segment")
    if inverse is not None and origin_segment is not None:
        return feat[inverse], logits[inverse], xyz[inverse], origin_segment.long()
    return feat, logits, xyz, label


def append_balanced(raw: dict, feat: torch.Tensor, logits: torch.Tensor, coord: torch.Tensor, label: torch.Tensor, max_points: int, max_per_class: int, num_classes: int) -> None:
    total = raw["total"]
    if total >= max_points:
        return
    valid = (label >= 0) & (label < num_classes)
    keep_parts = []
    used = 0
    for cls in range(num_classes):
        room_cls = max_per_class - raw["class_counts"][cls]
        if room_cls <= 0:
            continue
        idx = ((label == cls) & valid).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        room_total = max_points - total - used
        if room_total <= 0:
            break
        cap = min(room_cls, room_total)
        if idx.numel() > cap:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:cap]]
        keep_parts.append(idx)
        raw["class_counts"][cls] += int(idx.numel())
        used += int(idx.numel())
    if keep_parts:
        keep = torch.cat(keep_parts, dim=0)
        raw["feat"].append(feat[keep].detach().cpu())
        raw["logits"].append(logits[keep].detach().cpu())
        raw["coord"].append(coord[keep].detach().cpu())
        raw["label"].append(label[keep].detach().cpu())
        raw["total"] += int(keep.numel())


def append_natural(raw: dict, feat: torch.Tensor, logits: torch.Tensor, coord: torch.Tensor, label: torch.Tensor, max_points: int, num_classes: int) -> None:
    if raw["total"] >= max_points:
        return
    valid = (label >= 0) & (label < num_classes)
    idx = valid.nonzero(as_tuple=False).flatten()
    room = max_points - raw["total"]
    if idx.numel() > room:
        idx = idx[:room]
    if idx.numel() == 0:
        return
    for cls in range(num_classes):
        raw["class_counts"][cls] += int((label[idx] == cls).sum().item())
    raw["feat"].append(feat[idx].detach().cpu())
    raw["logits"].append(logits[idx].detach().cpu())
    raw["coord"].append(coord[idx].detach().cpu())
    raw["label"].append(label[idx].detach().cpu())
    raw["total"] += int(idx.numel())


def append_reservoir(raw: dict, feat: torch.Tensor, logits: torch.Tensor, coord: torch.Tensor, label: torch.Tensor, max_points: int, num_classes: int) -> None:
    valid = (label >= 0) & (label < num_classes)
    idx = valid.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return
    feat_new = feat[idx].detach().cpu()
    logits_new = logits[idx].detach().cpu()
    coord_new = coord[idx].detach().cpu()
    label_new = label[idx].detach().cpu()
    if raw["feat"]:
        feat_all = torch.cat([raw["feat"][0], feat_new], dim=0)
        logits_all = torch.cat([raw["logits"][0], logits_new], dim=0)
        coord_all = torch.cat([raw["coord"][0], coord_new], dim=0)
        label_all = torch.cat([raw["label"][0], label_new], dim=0)
    else:
        feat_all, logits_all, coord_all, label_all = feat_new, logits_new, coord_new, label_new
    if feat_all.shape[0] > max_points:
        keep = torch.randperm(feat_all.shape[0])[:max_points]
        feat_all = feat_all[keep]
        logits_all = logits_all[keep]
        coord_all = coord_all[keep]
        label_all = label_all[keep]
    raw["feat"] = [feat_all]
    raw["logits"] = [logits_all]
    raw["coord"] = [coord_all]
    raw["label"] = [label_all]
    raw["total"] = int(label_all.numel())
    counts = torch.bincount(label_all.clamp_min(0), minlength=num_classes)
    raw["class_counts"] = {i: int(counts[i].item()) for i in range(num_classes)}


def class_cap_room(raw: dict, max_per_class: int, num_classes: int) -> bool:
    return any(raw["class_counts"][cls] < max_per_class for cls in range(num_classes))


def finalize_cache(raw: dict, seen: int) -> PointCache:
    return PointCache(
        feat=torch.cat(raw["feat"], dim=0).float(),
        logits=torch.cat(raw["logits"], dim=0).float(),
        coord=torch.cat(raw["coord"], dim=0).float(),
        label=torch.cat(raw["label"], dim=0).long(),
        class_counts={SCANNET20_CLASS_NAMES[k]: int(v) for k, v in raw["class_counts"].items()},
        seen_batches=seen,
    )


def collect_cache(args: argparse.Namespace, model, cfg, split: str, balanced: bool, max_batches: int, max_points: int, num_classes: int) -> PointCache:
    raw = {
        "feat": [],
        "logits": [],
        "coord": [],
        "label": [],
        "class_counts": {i: 0 for i in range(num_classes)},
        "total": 0,
    }
    loader = build_loader(cfg, split, args.data_root, args.batch_size, args.num_worker)
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches >= 0 and batch_idx >= max_batches:
                break
            if raw["total"] >= max_points:
                if balanced and class_cap_room(raw, args.max_train_per_class, num_classes):
                    pass
                elif (not balanced) and args.val_sampling == "reservoir":
                    pass
                else:
                    break
            batch = move_to_cuda(batch)
            feat, logits, label, batch = forward_features(model, batch)
            feat, logits, coord, label = eval_tensors_with_coord(feat, logits, label, batch)
            if balanced:
                append_balanced(raw, feat, logits, coord, label, max_points, args.max_train_per_class, num_classes)
            elif args.val_sampling == "reservoir":
                append_reservoir(raw, feat, logits, coord, label, max_points, num_classes)
            else:
                append_natural(raw, feat, logits, coord, label, max_points, num_classes)
            seen += 1
            if (batch_idx + 1) % 25 == 0:
                print(f"[cache:{split}] batch={batch_idx+1} points={raw['total']}", flush=True)
    if not raw["label"]:
        raise RuntimeError(f"empty cache for split={split}")
    cache = finalize_cache(raw, seen)
    print(f"[cache:{split}] done points={cache.label.numel()} seen_batches={seen}", flush=True)
    return cache


def train_xyz_mlp(args: argparse.Namespace, train: PointCache, num_classes: int) -> XYZMLP:
    model = XYZMLP(args.xyz_hidden_dim, num_classes).cuda()
    valid = (train.label >= 0) & (train.label < num_classes)
    x = train.coord[valid].float()
    y = train.label[valid].long()
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = (counts.sum() / counts.clamp_min(1.0)).clamp_max(50.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=args.xyz_batch_size, shuffle=True, num_workers=0, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.xyz_lr, weight_decay=args.xyz_weight_decay)
    weights = weights.cuda()
    for epoch in range(args.xyz_epochs):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for xb, yb in loader:
            xb = xb.cuda(non_blocking=True)
            yb = yb.cuda(non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(yb.numel())
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_seen += int(yb.numel())
        if epoch in {0, 1, 2, 4, 9, args.xyz_epochs - 1} or (epoch + 1) % 10 == 0:
            print(
                f"[xyz-mlp] epoch={epoch+1}/{args.xyz_epochs} loss={total_loss/max(total_seen,1):.6f} acc={total_correct/max(total_seen,1):.4f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def hidden_features(model: XYZMLP, coord: torch.Tensor, batch_size: int) -> torch.Tensor:
    outs = []
    for start in range(0, coord.shape[0], batch_size):
        xb = coord[start : start + batch_size].cuda(non_blocking=True)
        outs.append(model.hidden(xb).detach().cpu())
    return torch.cat(outs, dim=0).float()


def fit_pca2(train_hidden: torch.Tensor, val_hidden: torch.Tensor):
    mean = train_hidden.mean(dim=0, keepdim=True)
    train_c = train_hidden - mean
    val_c = val_hidden - mean
    _, _, vh = torch.linalg.svd(train_c, full_matrices=False)
    comp = vh[:2].T.contiguous()
    train_u = train_c @ comp
    val_u = val_c @ comp
    u_mean = train_u.mean(dim=0, keepdim=True)
    u_std = train_u.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_u - u_mean) / u_std, (val_u - u_mean) / u_std, comp, mean.squeeze(0), u_mean.squeeze(0), u_std.squeeze(0)


def standardize_features(train_feat: torch.Tensor, val_feat: torch.Tensor):
    mean = train_feat.mean(dim=0, keepdim=True)
    std = train_feat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_feat - mean) / std, (val_feat - mean) / std, mean.squeeze(0), std.squeeze(0)


def ridge_regression(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    out_dtype = x.dtype
    x = x.double()
    y = y.double()
    d = x.shape[1]
    xtx = x.T @ x
    reg = ridge * torch.eye(d, dtype=x.dtype, device=x.device)
    xty = x.T @ y
    system = xtx + reg
    try:
        return torch.linalg.solve(system, xty).to(out_dtype)
    except torch._C._LinAlgError:
        return (torch.linalg.pinv(system) @ xty).to(out_dtype)


def add_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)], dim=1)


def fit_ridge_with_bias(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    return ridge_regression(add_bias(x), y, ridge)


def predict_with_bias(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return add_bias(x) @ w


def r2_score(y: torch.Tensor, pred: torch.Tensor) -> tuple[float, list[float]]:
    y = y.float()
    pred = pred.float()
    ss_res = ((y - pred) ** 2).sum(dim=0)
    ss_tot = ((y - y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-8)
    per_dim = 1.0 - ss_res / ss_tot
    overall = 1.0 - ss_res.sum() / ss_tot.sum().clamp_min(1e-8)
    return float(overall.item()), [float(x) for x in per_dim.tolist()]


def orth_basis(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if mat.numel() == 0:
        return mat.new_zeros((mat.shape[0], 0))
    u, s, _ = torch.linalg.svd(mat, full_matrices=False)
    rank = int((s > eps).sum().item())
    return u[:, :rank]


def one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = torch.zeros((label.shape[0], num_classes), dtype=torch.float32)
    valid = (label >= 0) & (label < num_classes)
    y[valid, label[valid].long()] = 1.0
    return y


def fit_classifier_logits(x: torch.Tensor, label: torch.Tensor, num_classes: int, ridge: float) -> torch.Tensor:
    valid = (label >= 0) & (label < num_classes)
    return fit_ridge_with_bias(x[valid], one_hot(label[valid], num_classes), ridge)


def eval_logits(logits: torch.Tensor, label: torch.Tensor, num_classes: int, ignore_index: int):
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    update_confusion(conf, logits.argmax(dim=1), label, num_classes, ignore_index)
    return conf.numpy(), summarize_confusion(conf.numpy(), SCANNET20_CLASS_NAMES)


def class_iou(summary: dict, name: str) -> float:
    return float(summary["iou"][NAME_TO_ID[name]])


def projection_energy(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return torch.zeros(x.shape[0], dtype=torch.float32)
    numer = ((x @ basis) ** 2).sum(dim=1)
    denom = (x ** 2).sum(dim=1).clamp_min(1e-8)
    return numer / denom


def subset_r2_rows(y: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, name: str) -> dict:
    if int(mask.sum().item()) < 4:
        return {"subset": name, "count": int(mask.sum().item()), "r2": "", "r2_dim0": "", "r2_dim1": ""}
    r2, dims = r2_score(y[mask], pred[mask])
    return {"subset": name, "count": int(mask.sum().item()), "r2": r2, "r2_dim0": dims[0], "r2_dim1": dims[1]}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                val = row.get(key, "")
                out[key] = f"{val:.8f}" if isinstance(val, float) else val
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.config = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    args.weight = (repo_root / args.weight).resolve() if not args.weight.is_absolute() else args.weight
    args.data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.summary_prefix = (repo_root / args.summary_prefix).resolve() if not args.summary_prefix.is_absolute() else args.summary_prefix
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    cfg = load_config(args.config)
    num_classes = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    weak_classes = parse_names(args.weak_classes)
    betas = parse_float_list(args.betas)
    addback_lambdas = parse_float_list(args.addback_lambdas)

    if args.dry_run:
        loader = build_loader(cfg, args.val_split, args.data_root, 1, 0)
        batch = next(iter(loader))
        print(f"[dry] keys={sorted(batch.keys())}", flush=True)
        print(f"[dry] betas={betas} addback_lambdas={addback_lambdas}", flush=True)
        return 0

    model = build_model(cfg, args.weight)
    train = collect_cache(args, model, cfg, args.train_split, True, args.max_train_batches, args.max_train_points, num_classes)
    val = collect_cache(args, model, cfg, args.val_split, False, args.max_val_batches, args.max_val_points, num_classes)

    xyz_model = train_xyz_mlp(args, train, num_classes)
    with torch.no_grad():
        train_xyz_logits = []
        val_xyz_logits = []
        for coord, out in [(train.coord, train_xyz_logits), (val.coord, val_xyz_logits)]:
            for start in range(0, coord.shape[0], args.xyz_batch_size):
                out.append(xyz_model(coord[start : start + args.xyz_batch_size].cuda(non_blocking=True)).detach().cpu())
        train_xyz_logits = torch.cat(train_xyz_logits, dim=0)
        val_xyz_logits = torch.cat(val_xyz_logits, dim=0)
    train_xyz_hidden = hidden_features(xyz_model, train.coord, args.xyz_batch_size)
    val_xyz_hidden = hidden_features(xyz_model, val.coord, args.xyz_batch_size)
    train_u, val_u, pca_components, hidden_mean, pca_mean, pca_std = fit_pca2(train_xyz_hidden, val_xyz_hidden)

    train_x, val_x, feat_mean, feat_std = standardize_features(train.feat, val.feat)
    w_u = fit_ridge_with_bias(train_x, train_u, args.ridge)
    train_u_hat = predict_with_bias(train_x, w_u)
    val_u_hat = predict_with_bias(val_x, w_u)
    train_r2, train_r2_dims = r2_score(train_u, train_u_hat)
    val_r2, val_r2_dims = r2_score(val_u, val_u_hat)
    nuisance_basis = orth_basis(w_u[:-1, :])

    base_conf, base_summary = eval_logits(val.logits, val.label, num_classes, ignore_index)
    xyz_conf, xyz_summary = eval_logits(val_xyz_logits, val.label, num_classes, ignore_index)
    refit_w = fit_classifier_logits(train_x, train.label, num_classes, args.classifier_ridge)
    refit_conf, refit_summary = eval_logits(predict_with_bias(val_x, refit_w), val.label, num_classes, ignore_index)

    y_train = one_hot(train.label, num_classes)
    variant_rows = []

    def add_variant(name: str, logits: torch.Tensor, extra: dict | None = None) -> None:
        conf, summary = eval_logits(logits, val.label, num_classes, ignore_index)
        row = {
            "variant": name,
            "mIoU": summary["mIoU"],
            "delta_mIoU_vs_base_logits": summary["mIoU"] - base_summary["mIoU"],
            "delta_mIoU_vs_refit": summary["mIoU"] - refit_summary["mIoU"],
            "mAcc": summary["mAcc"],
            "allAcc": summary["allAcc"],
            "weak_mean_iou": weak_mean(summary, weak_classes),
            "picture_iou": class_iou(summary, "picture"),
            "picture_to_wall_frac": picture_to_wall(conf, summary),
            "counter_iou": class_iou(summary, "counter"),
            "desk_iou": class_iou(summary, "desk"),
            "sink_iou": class_iou(summary, "sink"),
            "cabinet_iou": class_iou(summary, "cabinet"),
            "door_iou": class_iou(summary, "door"),
            "shower_curtain_iou": class_iou(summary, "shower curtain"),
        }
        if extra:
            row.update(extra)
        variant_rows.append(row)

    add_variant("base_decoder_logits", val.logits, {"beta": "", "lambda": "", "r2_refit_after": ""})
    add_variant("xyz_only_mlp", val_xyz_logits, {"beta": "", "lambda": "", "r2_refit_after": ""})
    add_variant("refit_linear_original", predict_with_bias(val_x, refit_w), {"beta": 0.0, "lambda": "", "r2_refit_after": val_r2})

    for beta in betas:
        train_res = train_x - float(beta) * ((train_x @ nuisance_basis) @ nuisance_basis.T)
        val_res = val_x - float(beta) * ((val_x @ nuisance_basis) @ nuisance_basis.T)
        w_res = fit_classifier_logits(train_res, train.label, num_classes, args.classifier_ridge)
        logits_res_train = predict_with_bias(train_res, w_res)
        logits_res_val = predict_with_bias(val_res, w_res)
        w_u_resid = fit_ridge_with_bias(train_u_hat, y_train - logits_res_train, args.classifier_ridge)
        w_probe_after = fit_ridge_with_bias(train_res, train_u, args.ridge)
        r2_after, _ = r2_score(val_u, predict_with_bias(val_res, w_probe_after))
        add_variant(
            f"remove_beta{beta:g}".replace(".", "p"),
            logits_res_val,
            {"beta": beta, "lambda": "", "r2_refit_after": r2_after},
        )
        for lam in addback_lambdas:
            logits_add = logits_res_val + float(lam) * predict_with_bias(val_u_hat, w_u_resid)
            add_variant(
                f"addback_beta{beta:g}_lambda{lam:g}".replace(".", "p"),
                logits_add,
                {"beta": beta, "lambda": lam, "r2_refit_after": r2_after},
            )

    energy = projection_energy(val_x, nuisance_basis)
    base_pred = val.logits.argmax(dim=1)
    picture = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    subset_masks = {
        "all": torch.ones_like(val.label, dtype=torch.bool),
        "picture_all": val.label == picture,
        "picture_correct": (val.label == picture) & (base_pred == picture),
        "picture_to_wall": (val.label == picture) & (base_pred == wall),
        "wall_all": val.label == wall,
    }
    for name, cls in zip(SCANNET20_CLASS_NAMES, range(num_classes)):
        if cls in weak_classes:
            subset_masks[f"class_{name.replace(' ', '_')}"] = val.label == cls

    r2_rows = []
    energy_rows = []
    for name, mask in subset_masks.items():
        row = subset_r2_rows(val_u, val_u_hat, mask, name)
        r2_rows.append(row)
        if int(mask.sum().item()) > 0:
            e = energy[mask]
            pred_norm = val_u_hat[mask].norm(dim=1)
            energy_rows.append(
                {
                    "subset": name,
                    "count": int(mask.sum().item()),
                    "projection_energy_mean": float(e.mean().item()),
                    "projection_energy_p90": float(torch.quantile(e, 0.90).item()),
                    "u_hat_norm_mean": float(pred_norm.mean().item()),
                    "u_hat_norm_p90": float(torch.quantile(pred_norm, 0.90).item()),
                }
            )

    variant_rows.sort(key=lambda r: (r["mIoU"], r["picture_iou"]), reverse=True)
    fields = [
        "variant",
        "beta",
        "lambda",
        "r2_refit_after",
        "mIoU",
        "delta_mIoU_vs_base_logits",
        "delta_mIoU_vs_refit",
        "mAcc",
        "allAcc",
        "weak_mean_iou",
        "picture_iou",
        "picture_to_wall_frac",
        "counter_iou",
        "desk_iou",
        "sink_iou",
        "cabinet_iou",
        "door_iou",
        "shower_curtain_iou",
    ]
    write_csv(args.output_dir / "xyz_mlp_pca_rasa_variants.csv", variant_rows, fields)
    write_csv(args.summary_prefix.with_suffix(".csv"), variant_rows, fields)
    write_csv(args.output_dir / "xyz_mlp_pca_r2_by_subset.csv", r2_rows, ["subset", "count", "r2", "r2_dim0", "r2_dim1"])
    write_csv(args.output_dir / "xyz_mlp_pca_energy_by_subset.csv", energy_rows, ["subset", "count", "projection_energy_mean", "projection_energy_p90", "u_hat_norm_mean", "u_hat_norm_p90"])

    metadata = {
        "config": str(args.config),
        "weight": str(args.weight),
        "data_root": str(args.data_root),
        "train_points": int(train.label.numel()),
        "val_points": int(val.label.numel()),
        "train_batches": int(train.seen_batches),
        "val_batches": int(val.seen_batches),
        "val_sampling": args.val_sampling,
        "train_class_counts": train.class_counts,
        "val_class_counts": val.class_counts,
        "xyz_hidden_dim": int(args.xyz_hidden_dim),
        "xyz_epochs": int(args.xyz_epochs),
        "pca_components_shape": list(pca_components.shape),
        "nuisance_rank": int(nuisance_basis.shape[1]),
        "train_r2": train_r2,
        "train_r2_dim0": train_r2_dims[0],
        "train_r2_dim1": train_r2_dims[1],
        "val_r2": val_r2,
        "val_r2_dim0": val_r2_dims[0],
        "val_r2_dim1": val_r2_dims[1],
        "base_mIoU": base_summary["mIoU"],
        "base_picture_iou": class_iou(base_summary, "picture"),
        "base_picture_to_wall": picture_to_wall(base_conf, base_summary),
        "refit_mIoU": refit_summary["mIoU"],
        "refit_picture_iou": class_iou(refit_summary, "picture"),
        "xyz_only_mIoU": xyz_summary["mIoU"],
        "xyz_only_picture_iou": class_iou(xyz_summary, "picture"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(
        {
            "xyz_mlp_state_dict": xyz_model.state_dict(),
            "pca_components": pca_components,
            "hidden_mean": hidden_mean,
            "pca_mean": pca_mean,
            "pca_std": pca_std,
            "feature_mean": feat_mean,
            "feature_std": feat_std,
            "probe_weight": w_u,
            "nuisance_basis": nuisance_basis,
            "metadata": metadata,
        },
        args.output_dir / "xyz_mlp_pca_rasa_state.pt",
    )

    best = variant_rows[0]
    best_picture = max(variant_rows, key=lambda r: (r["picture_iou"], r["mIoU"]))
    md_lines = [
        "# XYZ-MLP PCA RASA Pilot",
        "",
        "## Setup",
        f"- config: `{args.config}`",
        f"- weight: `{args.weight}`",
        f"- train points: `{train.label.numel()}`",
        f"- val points: `{val.label.numel()}`",
        f"- xyz hidden dim: `{args.xyz_hidden_dim}`",
        f"- xyz epochs: `{args.xyz_epochs}`",
        "",
        "## Predictability",
        "",
        f"- train R2: `{train_r2:.4f}` (dim0 `{train_r2_dims[0]:.4f}`, dim1 `{train_r2_dims[1]:.4f}`)",
        f"- val R2: `{val_r2:.4f}` (dim0 `{val_r2_dims[0]:.4f}`, dim1 `{val_r2_dims[1]:.4f}`)",
        f"- nuisance basis rank: `{nuisance_basis.shape[1]}`",
        "",
        "## Baselines",
        "",
        f"- base decoder logits: mIoU `{base_summary['mIoU']:.4f}`, picture `{class_iou(base_summary, 'picture'):.4f}`, picture->wall `{picture_to_wall(base_conf, base_summary):.4f}`",
        f"- xyz-only MLP: mIoU `{xyz_summary['mIoU']:.4f}`, picture `{class_iou(xyz_summary, 'picture'):.4f}`",
        f"- refit linear on Concerto features: mIoU `{refit_summary['mIoU']:.4f}`, picture `{class_iou(refit_summary, 'picture'):.4f}`",
        "",
        "## Best Variants",
        "",
        f"- best mIoU: `{best['variant']}` mIoU `{best['mIoU']:.4f}`, picture `{best['picture_iou']:.4f}`, picture->wall `{best['picture_to_wall_frac']:.4f}`",
        f"- best picture: `{best_picture['variant']}` mIoU `{best_picture['mIoU']:.4f}`, picture `{best_picture['picture_iou']:.4f}`, picture->wall `{best_picture['picture_to_wall_frac']:.4f}`",
        "",
        "## Output Files",
        "",
        f"- `{args.output_dir / 'xyz_mlp_pca_rasa_variants.csv'}`",
        f"- `{args.output_dir / 'xyz_mlp_pca_r2_by_subset.csv'}`",
        f"- `{args.output_dir / 'xyz_mlp_pca_energy_by_subset.csv'}`",
        f"- `{args.output_dir / 'metadata.json'}`",
    ]
    args.summary_prefix.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (args.output_dir / "xyz_mlp_pca_rasa.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[write] {args.summary_prefix.with_suffix('.md')}", flush=True)
    print(f"[write] {args.summary_prefix.with_suffix('.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
