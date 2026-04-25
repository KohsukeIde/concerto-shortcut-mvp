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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.concerto_projection_shortcut.eval_concerto3d_dino_exact_controls_stepA import (  # noqa: E402
    NAME_TO_ID,
    SCANNET20_CLASS_NAMES,
)
from tools.concerto_projection_shortcut.eval_masking_battery import (  # noqa: E402
    build_loader,
    full_scene_logits_from_support,
    inference_batch,
)
from tools.concerto_projection_shortcut.eval_retrieval_prototype_readout import (  # noqa: E402
    build_model,
    load_config,
    move_to_cuda,
    summarize_confusion,
    update_confusion,
    weak_mean,
)
from tools.concerto_projection_shortcut.eval_utonia_scannet_support_stress import (  # noqa: E402
    build_model as build_utonia_model,
    forward_raw_logits as forward_utonia_raw_logits,
)


@dataclass(frozen=True)
class CurrentModelSpec:
    name: str
    config: Path
    weight: Path


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Raw-point aligned cross-model complementarity and simple fusion "
            "audit for ScanNet20. The first target is Concerto decoder vs "
            "Sonata/Utonia released stacks."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--data-root", type=Path, default=Path("data/scannet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--full-scene-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument(
        "--current-model",
        action="append",
        default=[],
        help="Repeated spec: name::config::weight for current Pointcept models.",
    )
    parser.add_argument("--include-utonia", action="store_true")
    parser.add_argument("--utonia-weight", type=Path, default=Path("data/weights/utonia/utonia.pth"))
    parser.add_argument("--utonia-head", type=Path, default=Path("data/weights/utonia/utonia_linear_prob_head_sc.pth"))
    parser.add_argument("--disable-utonia-flash", action="store_true")
    parser.add_argument(
        "--cached-expert",
        action="append",
        default=[],
        help="Repeated spec name::cache_dir. Cache files are per-scene .npz with probs or logits and labels.",
    )
    parser.add_argument("--weak-classes", default="picture,counter,desk,sink,cabinet,shower curtain,door")
    parser.add_argument("--summary-prefix", type=Path, default=Path("tools/concerto_projection_shortcut/results_cross_model_fusion_scannet20"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_current_specs(args: argparse.Namespace) -> list[CurrentModelSpec]:
    specs = []
    for raw in args.current_model:
        parts = raw.split("::")
        if len(parts) != 3:
            raise ValueError(f"invalid --current-model spec: {raw}")
        name, config, weight = parts
        specs.append(CurrentModelSpec(name=name, config=Path(config), weight=Path(weight)))
    if not specs:
        specs = [
            CurrentModelSpec(
                "Concerto decoder",
                Path("configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py"),
                Path("data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth"),
            ),
            CurrentModelSpec(
                "Sonata linear",
                Path("configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py"),
                Path("data/weights/sonata/sonata_scannet_linear_merged.pth"),
            ),
        ]
    return specs


def parse_weak_ids(text: str) -> list[int]:
    out = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown weak class: {name}")
        out.append(NAME_TO_ID[name])
    return out


def parse_cached_experts(raw_specs: list[str], repo_root: Path) -> list[tuple[str, Path]]:
    out = []
    for raw in raw_specs:
        parts = raw.split("::")
        if len(parts) != 2:
            raise ValueError(f"invalid --cached-expert spec: {raw}")
        name, cache_dir = parts
        out.append((name, resolve(repo_root, Path(cache_dir))))
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(root: Path, path: Path) -> Path:
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def split_dataset_cfg(cfg, split: str, data_root: Path):
    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg.split = split
    ds_cfg.data_root = str(data_root)
    ds_cfg.test_mode = False
    return ds_cfg


@torch.no_grad()
def forward_current_raw_logits(model, batch: dict, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    model_input = inference_batch(batch)
    out = model(model_input, return_point=True)
    logits = out["seg_logits"].float()
    labels = batch["origin_segment"].long()
    # Clean current-Pointcept rows expose an exact inverse from voxel support to
    # the original scene points. Fall back to NN only if a future config lacks it.
    if "inverse" in batch:
        raw_logits = logits[batch["inverse"].long()]
    else:
        raw_logits = full_scene_logits_from_support(logits, batch, batch, chunk_size)
    if raw_logits.shape[0] != labels.shape[0]:
        raise RuntimeError(f"raw logits/labels mismatch: {raw_logits.shape} vs {labels.shape}")
    return raw_logits.cpu(), labels.cpu()


def build_current_models(repo_root: Path, specs: list[CurrentModelSpec]):
    cfg0 = load_config(resolve(repo_root, specs[0].config))
    loader = build_loader(cfg0, "val", resolve(repo_root, Path("data/scannet")), 1, 4)
    models = []
    for spec in specs:
        cfg = load_config(resolve(repo_root, spec.config))
        model = build_model(cfg, resolve(repo_root, spec.weight)).cuda().eval()
        models.append((spec.name, model))
    return cfg0, loader, models


def build_utonia_scene_transform():
    import utonia

    return utonia.transform.default(0.5)


def transform_utonia_scene(transform, scene: dict) -> dict:
    point = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in scene.items()}
    raw_segment = torch.from_numpy(point["segment"].copy()).long()
    out = transform(point)
    out["segment"] = torch.from_numpy(point["segment"]).long()
    out["raw_segment"] = raw_segment
    return out


def current_raw_scene_from_dataset(dataset, idx: int) -> dict:
    scene = dataset.get_data(idx)
    if "segment" not in scene and "segment20" in scene:
        scene["segment"] = scene["segment20"]
    if "instance" not in scene:
        scene_dir = Path(dataset.data_root) / dataset.split / dataset.get_data_name(idx)
        inst = scene_dir / "instance.npy"
        if inst.exists():
            scene["instance"] = np.load(inst)
    return scene


def scene_name_from_dataset(dataset, idx: int) -> str:
    try:
        return str(dataset.get_data_name(idx))
    except Exception:
        return Path(dataset.data_list[idx]).name


def load_cached_expert_logits(cache_dir: Path, scene_name: str, labels_ref: torch.Tensor) -> torch.Tensor:
    path = cache_dir / f"{scene_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing cached expert scene file: {path}")
    with np.load(path) as data:
        labels = torch.from_numpy(data["labels"].astype(np.int64))
        if labels.shape != labels_ref.shape or not torch.equal(labels, labels_ref):
            raise RuntimeError(f"cached expert label mismatch for {path}: {labels.shape} vs {labels_ref.shape}")
        if "logits" in data:
            logits_np = data["logits"].astype(np.float32)
        elif "probs" in data:
            logits_np = np.log(np.clip(data["probs"].astype(np.float32), 1e-8, 1.0))
        else:
            raise KeyError(f"cache file lacks logits/probs: {path}")
    return torch.from_numpy(logits_np)


def softmax_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.softmax(logits.float() / temperature, dim=1)


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)


def margin(probs: torch.Tensor) -> torch.Tensor:
    top2 = probs.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def oracle_pred(logits_by_model: dict[str, torch.Tensor], labels: torch.Tensor, model_names: list[str]) -> torch.Tensor:
    base = logits_by_model[model_names[0]].argmax(dim=1)
    pred = base.clone()
    correct_any = torch.zeros_like(labels, dtype=torch.bool)
    for name in model_names:
        correct_any |= logits_by_model[name].argmax(dim=1) == labels
    pred[correct_any] = labels[correct_any]
    return pred


def top2_contains_pair(logits: torch.Tensor, a: int, b: int) -> torch.Tensor:
    idx = logits.topk(2, dim=1).indices
    has_a = (idx == a).any(dim=1)
    has_b = (idx == b).any(dim=1)
    return has_a | has_b


def build_variant_predictions(logits_by_model: dict[str, torch.Tensor], labels: torch.Tensor, weak_ids: list[int]) -> dict[str, torch.Tensor]:
    names = list(logits_by_model)
    preds = {f"single::{name}": logits.argmax(dim=1) for name, logits in logits_by_model.items()}

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair = [a, b]
            probs = [softmax_logits(logits_by_model[x]) for x in pair]
            preds[f"oracle::{a}+{b}"] = oracle_pred(logits_by_model, labels, pair)
            preds[f"avgprob::{a}+{b}"] = torch.stack(probs, dim=0).mean(dim=0).argmax(dim=1)
            preds[f"avgprobT2::{a}+{b}"] = torch.stack([softmax_logits(logits_by_model[x], 2.0) for x in pair], dim=0).mean(dim=0).argmax(dim=1)
            conf = torch.stack([p.max(dim=1).values for p in probs], dim=1)
            choose_b = conf[:, 1] > conf[:, 0]
            pred = logits_by_model[a].argmax(dim=1)
            pred[choose_b] = logits_by_model[b].argmax(dim=1)[choose_b]
            preds[f"maxconf::{a}+{b}"] = pred

            marg = torch.stack([margin(p) for p in probs], dim=1)
            choose_b = marg[:, 1] > marg[:, 0]
            pred = logits_by_model[a].argmax(dim=1)
            pred[choose_b] = logits_by_model[b].argmax(dim=1)[choose_b]
            preds[f"maxmargin::{a}+{b}"] = pred

    if len(names) >= 3:
        probs_all = [softmax_logits(logits_by_model[x]) for x in names]
        preds[f"oracle::{'+'.join(names)}"] = oracle_pred(logits_by_model, labels, names)
        preds[f"avgprob::{'+'.join(names)}"] = torch.stack(probs_all, dim=0).mean(dim=0).argmax(dim=1)
        conf = torch.stack([p.max(dim=1).values for p in probs_all], dim=1)
        best = conf.argmax(dim=1)
        stacked_pred = torch.stack([logits_by_model[x].argmax(dim=1) for x in names], dim=1)
        preds[f"maxconf::{'+'.join(names)}"] = stacked_pred.gather(1, best[:, None]).squeeze(1)

    # Weak-class gate: use the last model as the specialist when either base or
    # specialist predicts a weak class. In the default order this means Utonia.
    if len(names) >= 2:
        base = names[0]
        specialist = names[-1]
        weak = torch.tensor(weak_ids, dtype=torch.long)
        base_pred = logits_by_model[base].argmax(dim=1)
        spec_pred = logits_by_model[specialist].argmax(dim=1)
        mask = torch.isin(base_pred.cpu(), weak) | torch.isin(spec_pred.cpu(), weak)
        pred = base_pred.clone()
        pred[mask.to(pred.device)] = spec_pred[mask.to(pred.device)]
        preds[f"weakgate::{base}->{specialist}"] = pred

        pic = NAME_TO_ID["picture"]
        wall = NAME_TO_ID["wall"]
        mask = top2_contains_pair(logits_by_model[base], pic, wall) | top2_contains_pair(logits_by_model[specialist], pic, wall)
        pred = base_pred.clone()
        pred[mask] = spec_pred[mask]
        preds[f"picturewall_top2_gate::{base}->{specialist}"] = pred

    return preds


def picture_to_wall(conf: np.ndarray) -> float:
    pic = NAME_TO_ID["picture"]
    wall = NAME_TO_ID["wall"]
    denom = conf[pic].sum()
    return float(conf[pic, wall] / denom) if denom else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    repo_root = args.repo_root.resolve()
    data_root = resolve(repo_root, args.data_root)
    out_dir = resolve(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_prefix = resolve(repo_root, args.summary_prefix)
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    current_specs = parse_current_specs(args)
    cached_experts = parse_cached_experts(args.cached_expert, repo_root)

    if args.dry_run:
        print(f"[dry-run] current={current_specs}")
        print(f"[dry-run] include_utonia={args.include_utonia}")
        return

    cfg0 = load_config(resolve(repo_root, current_specs[0].config))
    loader = build_loader(cfg0, args.val_split, data_root, args.batch_size, args.num_worker)
    if len(loader.dataset) == 0:
        raise RuntimeError(
            f"empty {args.val_split} dataset at data_root={data_root}. "
            "For ScanNet20 fusion use --data-root data/scannet or set "
            "SCANNET_DATA_ROOT/SCANNET_EXTRACT_DIR, not the generic DATA_ROOT."
        )
    current_models = []
    for spec in current_specs:
        cfg = load_config(resolve(repo_root, spec.config))
        current_models.append((spec.name, build_model(cfg, resolve(repo_root, spec.weight)).cuda().eval()))

    utonia_model = utonia_head = utonia_transform = None
    if args.include_utonia:
        utonia_model, utonia_head = build_utonia_model(
            resolve(repo_root, args.utonia_weight),
            resolve(repo_root, args.utonia_head),
            args.disable_utonia_flash,
        )
        utonia_transform = build_utonia_scene_transform()

    weak_ids = parse_weak_ids(args.weak_classes)
    confs: dict[str, torch.Tensor] = {}
    complement_counts: dict[str, dict[str, int]] = {}
    num_classes = len(SCANNET20_CLASS_NAMES)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches >= 0 and batch_idx >= args.max_val_batches:
                break
            batch = move_to_cuda(batch)
            labels_ref = batch["origin_segment"].long().cpu()
            logits_by_model: dict[str, torch.Tensor] = {}
            for name, model in current_models:
                logits, labels = forward_current_raw_logits(model, batch, args.full_scene_chunk_size)
                if labels.shape != labels_ref.shape or not torch.equal(labels, labels_ref):
                    raise RuntimeError(f"label mismatch for {name} at batch {batch_idx}")
                logits_by_model[name] = logits
            if args.include_utonia:
                raw_scene = current_raw_scene_from_dataset(loader.dataset, batch_idx)
                utonia_batch = transform_utonia_scene(utonia_transform, raw_scene)
                logits, labels = forward_utonia_raw_logits(utonia_model, utonia_head, utonia_batch)
                if labels.shape != labels_ref.shape or not torch.equal(labels, labels_ref):
                    raise RuntimeError(f"Utonia label mismatch at batch {batch_idx}: {labels.shape} vs {labels_ref.shape}")
                logits_by_model["Utonia"] = logits
            scene_name = scene_name_from_dataset(loader.dataset, batch_idx)
            for name, cache_dir in cached_experts:
                logits_by_model[name] = load_cached_expert_logits(cache_dir, scene_name, labels_ref)

            preds = build_variant_predictions(logits_by_model, labels_ref, weak_ids)
            for variant, pred in preds.items():
                confs.setdefault(variant, torch.zeros((num_classes, num_classes), dtype=torch.long))
                update_confusion(confs[variant], pred.cpu(), labels_ref.cpu(), num_classes, -1)

            model_names = list(logits_by_model)
            for i, a in enumerate(model_names):
                for b in model_names[i + 1 :]:
                    key = f"{a}+{b}"
                    complement_counts.setdefault(key, {"both_correct": 0, f"{a}_only": 0, f"{b}_only": 0, "both_wrong": 0, "total": 0})
                    ca = logits_by_model[a].argmax(dim=1) == labels_ref
                    cb = logits_by_model[b].argmax(dim=1) == labels_ref
                    complement_counts[key]["both_correct"] += int((ca & cb).sum().item())
                    complement_counts[key][f"{a}_only"] += int((ca & ~cb).sum().item())
                    complement_counts[key][f"{b}_only"] += int((~ca & cb).sum().item())
                    complement_counts[key]["both_wrong"] += int((~ca & ~cb).sum().item())
                    complement_counts[key]["total"] += int(labels_ref.numel())

            if (batch_idx + 1) % 25 == 0:
                first = next(iter(confs))
                miou = summarize_confusion(confs[first].numpy(), SCANNET20_CLASS_NAMES)["mIoU"]
                print(f"[eval] scenes={batch_idx + 1}/{len(loader.dataset)} first={first} mIoU={miou:.4f}", flush=True)

    rows = []
    weak = weak_ids
    for variant, conf_t in sorted(confs.items()):
        conf = conf_t.numpy()
        s = summarize_confusion(conf, SCANNET20_CLASS_NAMES)
        rows.append(
            {
                "variant": variant,
                "mIoU": s["mIoU"],
                "mAcc": s["mAcc"],
                "allAcc": s["allAcc"],
                "weak_mean_iou": weak_mean(s, weak),
                "picture_iou": float(s["iou"][NAME_TO_ID["picture"]]),
                "wall_iou": float(s["iou"][NAME_TO_ID["wall"]]),
                "counter_iou": float(s["iou"][NAME_TO_ID["counter"]]),
                "cabinet_iou": float(s["iou"][NAME_TO_ID["cabinet"]]),
                "door_iou": float(s["iou"][NAME_TO_ID["door"]]),
                "picture_to_wall": picture_to_wall(conf),
            }
        )
    write_csv(out_dir / "cross_model_fusion_summary.csv", rows)
    write_csv(summary_prefix.with_suffix(".csv"), rows)

    comp_rows = []
    for key, counts in complement_counts.items():
        total = counts["total"]
        row = {"pair": key, **counts}
        for name, value in counts.items():
            if name != "total":
                row[f"{name}_frac"] = value / total if total else float("nan")
        comp_rows.append(row)
    write_csv(out_dir / "cross_model_complementarity_counts.csv", comp_rows)

    sorted_rows = sorted(rows, key=lambda r: float(r["mIoU"]), reverse=True)
    md = [
        "# Cross-Model Complementarity / Fusion Audit",
        "",
        "Raw-point aligned ScanNet20 audit. Current-Pointcept models use the validation `inverse` mapping to original scene points; Utonia uses its released inverse-restored raw logits.",
        "",
        "## Results",
        "",
        "| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(sorted_rows, 1):
        md.append(
            f"| {rank} | `{row['variant']}` | `{fmt(row['mIoU'])}` | `{fmt(row['allAcc'])}` | "
            f"`{fmt(row['weak_mean_iou'])}` | `{fmt(row['picture_iou'])}` | `{fmt(row['picture_to_wall'])}` | "
            f"`{fmt(row['counter_iou'])}` | `{fmt(row['cabinet_iou'])}` | `{fmt(row['door_iou'])}` |"
        )
    md.extend(["", "## Pairwise Complementarity", ""])
    md.append("| pair | both correct | first only | second only | both wrong | oracle correct frac |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for row in comp_rows:
        pair = row["pair"]
        parts = pair.split("+")
        first_only = row.get(f"{parts[0]}_only_frac", "")
        second_only = row.get(f"{parts[1]}_only_frac", "") if len(parts) > 1 else ""
        oracle_frac = 1.0 - float(row.get("both_wrong_frac", 0.0))
        md.append(
            f"| `{pair}` | `{fmt(row.get('both_correct_frac'))}` | `{fmt(first_only)}` | "
            f"`{fmt(second_only)}` | `{fmt(row.get('both_wrong_frac'))}` | `{fmt(oracle_frac)}` |"
        )
    md.extend(
        [
            "",
            "## Interpretation Gate",
            "",
            "- If `oracle::Concerto decoder+Utonia` is far above both singles but simple `avgprob`/confidence gates do not move, complementarity exists but requires learned fusion.",
            "- If simple fusion already beats the Concerto full-FT reference (`~0.8075` in this repo, `80.7` in the paper), this becomes a concrete SOTA-method line.",
            "- If two-model oracle is only marginally above the best single model, cross-model fusion is unlikely to be the right SOTA direction.",
        ]
    )
    summary_prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "cross_model_fusion_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "current_models": [spec.__dict__ | {"config": str(spec.config), "weight": str(spec.weight)} for spec in current_specs],
                "include_utonia": args.include_utonia,
                "utonia_weight": str(args.utonia_weight),
                "utonia_head": str(args.utonia_head),
                "cached_experts": [(name, str(path)) for name, path in cached_experts],
                "max_val_batches": args.max_val_batches,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {summary_prefix.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
