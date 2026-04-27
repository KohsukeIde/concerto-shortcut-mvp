#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tools/concerto_projection_shortcut"
NEPA_ROOT = ROOT / "3D-NEPA"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=4) -> str:
    if x is None:
        return ""
    if x == "":
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_pct(x, digits=1) -> str:
    if x is None or x == "":
        return ""
    return f"{100.0 * float(x):.{digits}f}%"


def build_six_dataset_table() -> None:
    src = OUT_DIR / "results_main_variant_causal_battery.csv"
    rows = read_csv(src)
    by_ds: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], {})[row["mode"]] = row

    out = []
    for dataset, modes in by_ds.items():
        base = float(modes["none"]["enc2d_loss_mean"])
        deltas = {
            "global_target_permutation": float(modes["global_target_permutation"]["delta_vs_baseline"]),
            "cross_image_target_swap": float(modes["cross_image_target_swap"]["delta_vs_baseline"]),
            "cross_scene_target_swap": float(modes["cross_scene_target_swap"]["delta_vs_baseline"]),
        }
        max_delta = max(deltas.values())
        out.append(
            {
                "dataset": dataset,
                "baseline_loss": base,
                "global_perm_delta": deltas["global_target_permutation"],
                "cross_image_delta": deltas["cross_image_target_swap"],
                "cross_scene_delta": deltas["cross_scene_target_swap"],
                "b_pre_max_delta": max_delta,
                "b_pre_relative": max_delta / base,
                "interpretation": "positive" if max_delta > 0 else "flat",
            }
        )
    write_csv(OUT_DIR / "results_main_variant_causal_battery_paper_table.csv", out)

    lines = [
        "# Main-Variant Six-Dataset Causal Battery: Paper Table",
        "",
        "This table reformats the completed main-variant head-refit causal battery into a single paper-facing row per dataset.",
        "",
        "| dataset | baseline loss | global perm delta | cross-image delta | cross-scene delta | max delta | relative B_pre |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in out:
        lines.append(
            f"| `{row['dataset']}` | `{fmt(row['baseline_loss'], 4)}` | `{fmt(row['global_perm_delta'], 4)}` | "
            f"`{fmt(row['cross_image_delta'], 4)}` | `{fmt(row['cross_scene_delta'], 4)}` | "
            f"`{fmt(row['b_pre_max_delta'], 4)}` | `{fmt(row['b_pre_relative'], 4)}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The target-swap sensitivity is positive on all six indoor datasets.",
            "- Magnitude is dataset-dependent: `s3dis` is much weaker than ScanNet++ / HM3D / Structured3D, which should be reported rather than hidden.",
            "- Use this as the main scene-level train-side counterfactual table.",
        ]
    )
    (OUT_DIR / "results_main_variant_causal_battery_paper_table.md").write_text("\n".join(lines) + "\n")


def _coord_closure(coord_loss: float, clean_loss: float, corrupt_loss: float) -> tuple[float | str, float | str]:
    denom = corrupt_loss - clean_loss
    if denom <= 0:
        return "", ""
    relative_position = (coord_loss - clean_loss) / denom
    closure = 1.0 - relative_position
    return relative_position, closure


def build_six_dataset_coord_rival_calibration() -> None:
    """Calibrate the existing coord-only rival against six-dataset references.

    `results_official_coord_mlp_rival.csv` already contains coord-MLP losses
    for all six datasets, but only ARKit/ScanNet carried reference losses in
    that original file. The paper-facing claim needs the coord rival normalized
    against the same six-dataset causal battery used for target corruption.
    """

    coord_src = OUT_DIR / "results_official_coord_mlp_rival.csv"
    causal_src = OUT_DIR / "results_main_variant_causal_battery.csv"
    if not coord_src.exists() or not causal_src.exists():
        return

    causal_rows = read_csv(causal_src)
    by_ds: dict[str, dict[str, float]] = {}
    for row in causal_rows:
        by_ds.setdefault(row["dataset"], {})[row["mode"]] = float(row["enc2d_loss_mean"])

    rows = []
    for row in read_csv(coord_src):
        dataset = row["dataset"]
        if dataset not in by_ds:
            continue
        refs = by_ds[dataset]
        coord_loss = float(row["coord_mlp_loss"])
        clean_loss = refs["none"]
        global_loss = refs["global_target_permutation"]
        cross_image_loss = refs["cross_image_target_swap"]
        cross_scene_loss = refs["cross_scene_target_swap"]
        mean_corrupt_loss = (global_loss + cross_image_loss + cross_scene_loss) / 3.0

        rel_mean, closure_mean = _coord_closure(coord_loss, clean_loss, mean_corrupt_loss)
        rel_global, closure_global = _coord_closure(coord_loss, clean_loss, global_loss)
        rel_cross_image, closure_cross_image = _coord_closure(coord_loss, clean_loss, cross_image_loss)
        rel_cross_scene, closure_cross_scene = _coord_closure(coord_loss, clean_loss, cross_scene_loss)

        if closure_mean == "":
            gate = "invalid_reference"
        elif closure_mean < 0:
            gate = "worse_than_mean_corruption"
        elif closure_mean < 0.25:
            gate = "weak"
        elif closure_mean < 0.5:
            gate = "partial"
        else:
            gate = "strong"

        rows.append(
            {
                "dataset": dataset,
                "clean_loss": clean_loss,
                "coord_mlp_loss": coord_loss,
                "global_perm_loss": global_loss,
                "cross_image_loss": cross_image_loss,
                "cross_scene_loss": cross_scene_loss,
                "mean_corruption_loss": mean_corrupt_loss,
                "coord_delta_vs_clean": coord_loss - clean_loss,
                "mean_corruption_delta_vs_clean": mean_corrupt_loss - clean_loss,
                "relative_position_mean": rel_mean,
                "closure_fraction_mean": closure_mean,
                "closure_fraction_global": closure_global,
                "closure_fraction_cross_image": closure_cross_image,
                "closure_fraction_cross_scene": closure_cross_scene,
                "gate_hint": gate,
            }
        )

    write_csv(OUT_DIR / "results_coord_mlp_rival_six_dataset_calibration.csv", rows)

    closures = [float(r["closure_fraction_mean"]) for r in rows if r["closure_fraction_mean"] != ""]
    mean_closure = sum(closures) / len(closures) if closures else ""
    min_closure = min(closures) if closures else ""
    max_closure = max(closures) if closures else ""
    positive_count = sum(1 for x in closures if x > 0.0)

    lines = [
        "# Six-Dataset Coord-MLP Rival Calibration",
        "",
        "This table recalibrates the existing coordinate-only rival against the completed six-dataset main-variant causal battery.",
        "",
        "Definitions:",
        "- `relative_position = (L_coord - L_clean) / (L_corrupt - L_clean)`.",
        "- `closure_fraction = 1 - relative_position`; higher means the coordinate-only rival is closer to the clean reference than the corrupted reference.",
        "- The main summary uses the mean of global permutation, cross-image target swap, and cross-scene target swap as `L_corrupt`.",
        "",
        "| dataset | clean | coord MLP | mean corrupt | rel. position | closure | global closure | cross-image closure | cross-scene closure | hint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['dataset']}` | `{fmt(r['clean_loss'])}` | `{fmt(r['coord_mlp_loss'])}` | "
            f"`{fmt(r['mean_corruption_loss'])}` | `{fmt(r['relative_position_mean'])}` | "
            f"`{fmt_pct(r['closure_fraction_mean'])}` | `{fmt_pct(r['closure_fraction_global'])}` | "
            f"`{fmt_pct(r['closure_fraction_cross_image'])}` | `{fmt_pct(r['closure_fraction_cross_scene'])}` | "
            f"`{r['gate_hint']}` |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Mean closure against mean corruption: `{fmt_pct(mean_closure)}`.",
            f"- Min / max closure: `{fmt_pct(min_closure)}` / `{fmt_pct(max_closure)}`.",
            f"- Positive-closure datasets: `{positive_count}/{len(rows)}`.",
            "",
            "## Interpretation",
            "",
            "- The six-dataset target-corruption sensitivity is positive, but the coordinate-only rival is not uniformly strong.",
            "- The coordinate-only rival closes a nonzero fraction of the clean-to-corrupted gap on most datasets, but it is weak on ARKit and worse than the mean corruption reference on S3DIS.",
            "- Therefore, the safe paper claim is not that coordinate-only explains the six-dataset average. The safe claim is that the objective has a coordinate-satisfiable component whose strength is dataset-dependent.",
        ]
    )
    (OUT_DIR / "results_coord_mlp_rival_six_dataset_calibration.md").write_text("\n".join(lines) + "\n")

    # A later S3DIS-only follow-up reran the coord-MLP extraction with a much
    # larger validation cap. Keep the original canonical table above intact,
    # and emit an explicitly corrected variant for paper interpretation.
    s3dis_highval = ROOT / "data/runs/main_variant_coord_mlp_rival/s3dis-highval-probe/results_official_coord_mlp_rival.md"
    if not s3dis_highval.exists():
        return
    text = s3dis_highval.read_text()
    match = re.search(r"\|\s*s3dis\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)", text)
    if not match:
        return
    highval_coord_loss = float(match.group(1))

    corrected = []
    for row in rows:
        item = dict(row)
        if item["dataset"] == "s3dis":
            clean_loss = float(item["clean_loss"])
            global_loss = float(item["global_perm_loss"])
            cross_image_loss = float(item["cross_image_loss"])
            cross_scene_loss = float(item["cross_scene_loss"])
            mean_corrupt_loss = float(item["mean_corruption_loss"])
            rel_mean, closure_mean = _coord_closure(highval_coord_loss, clean_loss, mean_corrupt_loss)
            _, closure_global = _coord_closure(highval_coord_loss, clean_loss, global_loss)
            _, closure_cross_image = _coord_closure(highval_coord_loss, clean_loss, cross_image_loss)
            _, closure_cross_scene = _coord_closure(highval_coord_loss, clean_loss, cross_scene_loss)
            if closure_mean == "":
                gate = "invalid_reference"
            elif closure_mean < 0:
                gate = "worse_than_mean_corruption"
            elif closure_mean < 0.25:
                gate = "weak_highval"
            elif closure_mean < 0.5:
                gate = "partial_highval"
            else:
                gate = "strong_highval"
            item.update(
                {
                    "coord_mlp_loss": highval_coord_loss,
                    "coord_delta_vs_clean": highval_coord_loss - clean_loss,
                    "relative_position_mean": rel_mean,
                    "closure_fraction_mean": closure_mean,
                    "closure_fraction_global": closure_global,
                    "closure_fraction_cross_image": closure_cross_image,
                    "closure_fraction_cross_scene": closure_cross_scene,
                    "gate_hint": gate,
                }
            )
        corrected.append(item)

    write_csv(OUT_DIR / "results_coord_mlp_rival_six_dataset_calibration_s3dis_highval.csv", corrected)
    corrected_closures = [float(r["closure_fraction_mean"]) for r in corrected if r["closure_fraction_mean"] != ""]
    corrected_mean = sum(corrected_closures) / len(corrected_closures) if corrected_closures else ""
    corrected_min = min(corrected_closures) if corrected_closures else ""
    corrected_max = max(corrected_closures) if corrected_closures else ""
    corrected_positive = sum(1 for x in corrected_closures if x > 0.0)

    corrected_lines = [
        "# Six-Dataset Coord-MLP Rival Calibration with S3DIS High-Val Follow-Up",
        "",
        "This table keeps the same six-dataset causal references as the canonical calibration, but replaces the original tiny-cache S3DIS coord-MLP loss with the S3DIS-only high-validation follow-up.",
        "",
        "| dataset | clean | coord MLP | mean corrupt | rel. position | closure | global closure | cross-image closure | cross-scene closure | hint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in corrected:
        corrected_lines.append(
            f"| `{r['dataset']}` | `{fmt(r['clean_loss'])}` | `{fmt(r['coord_mlp_loss'])}` | "
            f"`{fmt(r['mean_corruption_loss'])}` | `{fmt(r['relative_position_mean'])}` | "
            f"`{fmt_pct(r['closure_fraction_mean'])}` | `{fmt_pct(r['closure_fraction_global'])}` | "
            f"`{fmt_pct(r['closure_fraction_cross_image'])}` | `{fmt_pct(r['closure_fraction_cross_scene'])}` | "
            f"`{r['gate_hint']}` |"
        )
    corrected_lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Mean closure against mean corruption: `{fmt_pct(corrected_mean)}`.",
            f"- Min / max closure: `{fmt_pct(corrected_min)}` / `{fmt_pct(corrected_max)}`.",
            f"- Positive-closure datasets: `{corrected_positive}/{len(corrected)}`.",
            "",
            "## Interpretation",
            "",
            "- The original S3DIS negative closure was mainly a small validation-cache artifact.",
            "- With the high-val S3DIS follow-up, all six datasets have positive closure, but S3DIS remains weak at `13.3%`.",
            "- The safe claim is still dataset-dependent coordinate-satisfiable signal, not a uniformly strong coordinate-only explanation.",
        ]
    )
    (OUT_DIR / "results_coord_mlp_rival_six_dataset_calibration_s3dis_highval.md").write_text(
        "\n".join(corrected_lines) + "\n"
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def object_rows() -> list[dict]:
    rows = []
    specs = [
        (
            "PointGPT-S official",
            NEPA_ROOT / "results/ptgpt_readout_official_objbg_full.json",
            NEPA_ROOT / "results/ptgpt_stress_official_objbg_severity.json",
        ),
        (
            "PointGPT-S no-mask",
            NEPA_ROOT / "results/ptgpt_readout_nomask_objbg_full.json",
            NEPA_ROOT / "results/ptgpt_stress_nomask_objbg_severity.json",
        ),
        (
            "PointGPT-S no-mask order-random",
            NEPA_ROOT / "results/ptgpt_nomask_ordrand_objbg_readout_full.json",
            NEPA_ROOT / "results/ptgpt_stress_nomask_ordrand_objbg_severity.json",
        ),
        (
            "PointGPT-S mask-on order-random",
            NEPA_ROOT / "results/ptgpt_masked_ordrand_objbg_readout_full.json",
            NEPA_ROOT / "results/ptgpt_stress_masked_ordrand_objbg_severity.json",
        ),
    ]
    for name, readout_path, stress_path in specs:
        if not readout_path.exists():
            continue
        readout = load_json(readout_path)
        row = {
            "domain": "object",
            "model": name,
            "task": "ScanObjectNN obj_bg",
            "top1_or_miou": readout.get("top1_acc"),
            "top2_oracle_proxy": readout.get("top2_hit"),
            "top5_oracle_proxy": readout.get("top5_hit"),
            "hardest_pair": f"{readout['hardest_pair']['a_name']}->{readout['hardest_pair']['b_name']}",
            "hardest_pair_confusion": readout["hardest_pair"]["rate"],
            "pair_probe_bal_acc": readout["hardest_pair"]["pair_probe_bal_acc"],
            "random_keep20_down": "",
            "structured_keep20_down": "",
            "feature_zero_down": "",
        }
        if stress_path and stress_path.exists():
            stress = load_json(stress_path)
            by_name = {r["name"]: r["acc"] for r in stress["conditions"]}
            clean = by_name.get("clean")
            row["random_keep20_down"] = clean - by_name["random_keep20"] if clean is not None and "random_keep20" in by_name else ""
            row["structured_keep20_down"] = clean - by_name["structured_keep20"] if clean is not None and "structured_keep20" in by_name else ""
            row["feature_zero_down"] = clean - by_name["xyz_zero"] if clean is not None and "xyz_zero" in by_name else ""
        rows.append(row)
    return rows


def object_shapenetpart_rows() -> list[dict]:
    rows = []
    specs = [
        (
            "PointGPT-S official",
            NEPA_ROOT / "results/ptgpt_shapenetpart_official_support_stress.json",
        ),
        (
            "PointGPT-S no-mask",
            NEPA_ROOT / "results/ptgpt_shapenetpart_nomask_support_stress.json",
        ),
    ]
    for name, path in specs:
        if not path.exists():
            continue
        summary = load_json(path)
        by_name = {r["condition"]: r for r in summary["conditions"]}
        clean = by_name.get("clean", {})
        clean_iou = clean.get("class_avg_iou", "")
        rows.append(
            {
                "domain": "object-support",
                "model": name,
                "task": "ShapeNetPart support stress",
                "top1_or_miou": clean_iou,
                "top2_oracle_proxy": "",
                "top5_oracle_proxy": "",
                "hardest_pair": "",
                "hardest_pair_confusion": "",
                "pair_probe_bal_acc": "",
                "random_keep20_down": clean_iou - by_name["random_keep20"]["class_avg_iou"] if clean_iou != "" and "random_keep20" in by_name else "",
                "structured_keep20_down": clean_iou - by_name["structured_keep20"]["class_avg_iou"] if clean_iou != "" and "structured_keep20" in by_name else "",
                "feature_zero_down": clean_iou - by_name["xyz_zero"]["class_avg_iou"] if clean_iou != "" and "xyz_zero" in by_name else "",
            }
        )
    return rows


def build_shapenetpart_support_stress_table() -> None:
    specs = [
        (
            "PointGPT-S official",
            NEPA_ROOT / "results/ptgpt_shapenetpart_official_support_stress.json",
        ),
        (
            "PointGPT-S no-mask",
            NEPA_ROOT / "results/ptgpt_shapenetpart_nomask_support_stress.json",
        ),
    ]
    rows = []
    for name, path in specs:
        if not path.exists():
            continue
        summary = load_json(path)
        by_name = {r["condition"]: r for r in summary["conditions"]}
        clean = by_name["clean"]
        clean_class = float(clean["class_avg_iou"])
        clean_inst = float(clean["instance_avg_iou"])

        def val(condition: str, key: str = "class_avg_iou") -> float | str:
            if condition not in by_name:
                return ""
            return float(by_name[condition][key])

        random20 = val("random_keep20")
        structured20 = val("structured_keep20")
        part_drop = val("part_drop_largest")
        part_keep = val("part_keep20_per_part")
        xyz_zero = val("xyz_zero")
        rows.append(
            {
                "model": name,
                "clean_class_avg_iou": clean_class,
                "clean_instance_avg_iou": clean_inst,
                "random_keep20_class_avg_iou": random20,
                "structured_keep20_class_avg_iou": structured20,
                "part_drop_largest_class_avg_iou": part_drop,
                "part_keep20_per_part_class_avg_iou": part_keep,
                "xyz_zero_class_avg_iou": xyz_zero,
                "delta_random_keep20": clean_class - random20 if random20 != "" else "",
                "delta_structured_keep20": clean_class - structured20 if structured20 != "" else "",
                "delta_part_drop_largest": clean_class - part_drop if part_drop != "" else "",
                "delta_part_keep20_per_part": clean_class - part_keep if part_keep != "" else "",
                "delta_xyz_zero": clean_class - xyz_zero if xyz_zero != "" else "",
                "source": str(path.relative_to(ROOT)),
            }
        )

    write_csv(NEPA_ROOT / "results/ptgpt_shapenetpart_support_stress_paper_table.csv", rows)
    lines = [
        "# ShapeNetPart Support-Stress Paper Table",
        "",
        "This table reformats the completed ShapeNetPart support-stress audits for the object-level dense-transfer section.",
        "",
        "| model | clean c-mIoU | random keep20 | structured keep20 | part-drop largest | part keep20/part | xyz-zero | Δ random | Δ structured | Δ part-drop | Δ xyz-zero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['model']}` | `{fmt(r['clean_class_avg_iou'])}` | "
            f"`{fmt(r['random_keep20_class_avg_iou'])}` | `{fmt(r['structured_keep20_class_avg_iou'])}` | "
            f"`{fmt(r['part_drop_largest_class_avg_iou'])}` | `{fmt(r['part_keep20_per_part_class_avg_iou'])}` | "
            f"`{fmt(r['xyz_zero_class_avg_iou'])}` | `{fmt(r['delta_random_keep20'])}` | "
            f"`{fmt(r['delta_structured_keep20'])}` | `{fmt(r['delta_part_drop_largest'])}` | "
            f"`{fmt(r['delta_xyz_zero'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The no-mask checkpoint remains close to the official PointGPT-S checkpoint on clean ShapeNetPart, so the no-mask object-side finding is not classification-only.",
            "- Random keep20 is a weaker dense-transfer stress than structured keep20.",
            "- Semantic part removal (`part_drop_largest`) is the strongest support stress in both rows, which supports the claim that structured missing support is more decisive than random point sparsity.",
        ]
    )
    (NEPA_ROOT / "results/ptgpt_shapenetpart_support_stress_paper_table.md").write_text("\n".join(lines) + "\n")


def scene_readout_rows() -> list[dict]:
    src = OUT_DIR / "results_cross_model_downstream_audit_scannet20.csv"
    if not src.exists():
        return []
    rows = []
    for row in read_csv(src):
        if row["pair"] != "picture_vs_wall":
            continue
        rows.append(
            {
                "domain": "scene",
                "model": row["model"],
                "task": "ScanNet20 picture_vs_wall",
                "top1_or_miou": row["base_positive_iou"],
                "top2_oracle_proxy": row["top2_hit_rate"],
                "top5_oracle_proxy": row["top5_hit_rate"],
                "hardest_pair": "picture->wall",
                "hardest_pair_confusion": row["base_positive_to_negative"],
                "pair_probe_bal_acc": row["point_feature_bal_acc"],
                "random_keep20_down": "",
                "structured_keep20_down": "",
                "feature_zero_down": "",
            }
        )
    return rows


def support_rows_from_scene() -> list[dict]:
    severity_specs = [
        ("Concerto decoder", "ScanNet20 support-stress severity", OUT_DIR / "results_support_severity_concerto_decoder.csv"),
        ("Concerto linear", "ScanNet20 support-stress severity", OUT_DIR / "results_support_severity_concerto_linear.csv"),
        ("Sonata linear", "ScanNet20 support-stress severity", OUT_DIR / "results_support_severity_sonata_linear.csv"),
        ("PTv3 ScanNet20", "ScanNet20 support-stress severity", OUT_DIR / "results_support_severity_ptv3_scannet20.csv"),
        ("PTv3 ScanNet200", "ScanNet200 support-stress severity", OUT_DIR / "results_support_severity_ptv3_scannet200.csv"),
        ("PTv3 S3DIS", "S3DIS Area-5 support-stress severity", OUT_DIR / "results_support_severity_ptv3_s3dis.csv"),
    ]
    severity_rows = []
    for model, task, path in severity_specs:
        if not path.exists():
            continue
        vals = {}
        for row in read_csv(path):
            if row.get("score_space") != "full_nn":
                continue
            vals[row["variant"]] = float(row["mIoU"])
        clean = vals.get("clean_voxel")
        if clean is None:
            continue
        severity_rows.append(
            {
                "domain": "scene-support",
                "model": model,
                "task": task,
                "top1_or_miou": clean,
                "top2_oracle_proxy": "",
                "top5_oracle_proxy": "",
                "hardest_pair": "",
                "hardest_pair_confusion": "",
                "pair_probe_bal_acc": "",
                "random_keep20_down": clean - vals["random_keep0p2"] if "random_keep0p2" in vals else "",
                "structured_keep20_down": clean - vals["structured_b64_keep0p2"] if "structured_b64_keep0p2" in vals else "",
                "feature_zero_down": clean - vals["feature_zero1p0"] if "feature_zero1p0" in vals else "",
            }
        )
    if severity_rows:
        return severity_rows

    src = OUT_DIR / "results_masking_fullscene_scoring.csv"
    if not src.exists():
        return []
    rows_by_model = {}
    for row in read_csv(src):
        if row["score_space"] != "full_nn":
            continue
        model = row["method"]
        rows_by_model.setdefault(model, {})[row["variant"]] = float(row["mIoU"])
    out = []
    for model, vals in rows_by_model.items():
        clean = vals.get("clean_voxel")
        if clean is None:
            continue
        out.append(
            {
                "domain": "scene-support",
                "model": model,
                "task": "ScanNet/S3DIS full-scene support stress",
                "top1_or_miou": clean,
                "top2_oracle_proxy": "",
                "top5_oracle_proxy": "",
                "hardest_pair": "",
                "hardest_pair_confusion": "",
                "pair_probe_bal_acc": "",
                "random_keep20_down": clean - vals["random_keep0p2"] if "random_keep0p2" in vals else "",
                "structured_keep20_down": clean - vals["structured_b64_keep0p2"] if "structured_b64_keep0p2" in vals else "",
                "feature_zero_down": clean - vals["feature_zero1p0"] if "feature_zero1p0" in vals else "",
            }
        )
    return out


def support_rows_from_utonia() -> list[dict]:
    src = OUT_DIR / "results_support_severity_utonia/utonia_scannet_support_stress.csv"
    if not src.exists():
        src = OUT_DIR / "results_utonia_scannet_support_stress_featurezero_audit/utonia_scannet_support_stress.csv"
    if not src.exists():
        src = OUT_DIR / "results_utonia_scannet_support_stress/utonia_scannet_support_stress.csv"
    if not src.exists():
        return []
    vals = {row["condition"]: float(row["miou"]) for row in read_csv(src)}
    clean = vals.get("clean")
    if clean is None:
        return []
    return [
        {
            "domain": "scene-support",
            "model": "Utonia",
            "task": "ScanNet20 full-scene support stress",
            "top1_or_miou": clean,
            "top2_oracle_proxy": "",
            "top5_oracle_proxy": "",
            "hardest_pair": "",
            "hardest_pair_confusion": "",
            "pair_probe_bal_acc": "",
            "random_keep20_down": clean - vals["random_keep0p2"] if "random_keep0p2" in vals else "",
            "structured_keep20_down": clean - vals["structured_b1p28m_keep0p2"] if "structured_b1p28m_keep0p2" in vals else "",
            "feature_zero_down": clean - vals["feature_zero"] if "feature_zero" in vals else "",
        }
    ]


def build_binding_profile() -> None:
    rows = scene_readout_rows() + support_rows_from_scene() + support_rows_from_utonia() + object_rows() + object_shapenetpart_rows()
    write_csv(OUT_DIR / "results_binding_profile_summary.csv", rows)
    lines = [
        "# Binding Profile Summary",
        "",
        "This is a paper-facing consolidation table. It does not replace the source result files; it aligns train-side/readout/support quantities into one view for the central binding-profile figure.",
        "",
        "| domain | model | task | base metric | top2/proxy | top5/proxy | pair/confusion | pair probe | random20 damage | structured20 damage | feature-zero damage |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        pair = r["hardest_pair"]
        if r["hardest_pair_confusion"] != "":
            pair = f"{pair} ({fmt(r['hardest_pair_confusion'], 4)})"
        lines.append(
            f"| `{r['domain']}` | `{r['model']}` | `{r['task']}` | `{fmt(r['top1_or_miou'])}` | "
            f"`{fmt(r['top2_oracle_proxy'])}` | `{fmt(r['top5_oracle_proxy'])}` | `{pair}` | "
            f"`{fmt(r['pair_probe_bal_acc'])}` | `{fmt(r['random_keep20_down'])}` | "
            f"`{fmt(r['structured_keep20_down'])}` | `{fmt(r['feature_zero_down'])}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Scene readout rows use `picture_vs_wall` from the cross-model downstream audit.",
            "- Scene-support rows use full-scene nearest-neighbor scoring from the masking battery.",
            "- Utonia support stress uses the released Utonia inference/head path. The default Utonia transform explicitly builds feat=(coord,color,normal); its low feature-zero damage should therefore be treated as an audited robustness/anomaly, not as evidence that the input path omits raw features.",
            "- Object rows use ScanObjectNN `obj_bg` readout and support-stress audits.",
        ]
    )
    (OUT_DIR / "results_binding_profile_summary.md").write_text("\n".join(lines) + "\n")


def _float_or_blank(x: Any) -> float | str:
    if x in (None, ""):
        return ""
    return float(x)


def _recovery(base: float, oracle: float, recovered_delta: float | str) -> float | str:
    if recovered_delta == "":
        return ""
    denom = oracle - base
    if denom <= 0:
        return ""
    return max(0.0, float(recovered_delta)) / denom


def build_recoverability_table() -> None:
    """Build the paper-facing R_rec^max table.

    R_rec^max is the fraction of oracle headroom recovered by the strongest
    non-oracle intervention in a pre-specified fixed suite:

        (best_non_oracle_score - base_score) / (oracle_score - base_score).

    The main suite is intentionally small and family-level:

    Frozen recovery:
      1. class-prior correction / decoupled classifier
      2. nonparametric feature readout: prototype or kNN
      3. candidate-set reranking: constrained top-K reranking

    Adaptation recovery:
      4. capacity-limited adaptation: fixed-rank LoRA
      5. LP-FT: linear-probe-to-fine-tuning warm start
      6. full fine-tuning

    R_rec is protocol-matched: the base representation/readout that defines
    the oracle denominator must also define the recovery numerator. For example,
    the LP-FT rows below belong to the Concerto linear-head family and are not
    mixed into the Concerto decoder-probe R_rec row. Exploratory
    CoDA/CIDA/region/proposal/subgroup lines stay outside the main suite. We
    keep missing external-model suites explicit instead of inventing numbers.
    """

    fixed_suite_rows = [
        {
            "model": "Concerto decoder",
            "base_row": "frozen encoder + decoder probe",
            "suite": "frozen",
            "family": "class-prior correction / decoupled classifier",
            "best_delta_miou": 0.00020,
            "best_delta_picture": -0.00060,
            "source": "results_decoupled_classifier_readout.md",
            "notes": "tests long-tail / class-prior miscalibration; no meaningful recovery",
        },
        {
            "model": "Concerto decoder",
            "base_row": "frozen encoder + decoder probe",
            "suite": "frozen",
            "family": "nonparametric feature readout: prototype or kNN",
            "best_delta_miou": 0.00020,
            "best_delta_picture": 0.00080,
            "source": "results_knn_readout_small.md; results_prototype_readout.md",
            "notes": "tests nonparametric/metric readout geometry; no meaningful recovery",
        },
        {
            "model": "Concerto decoder",
            "base_row": "frozen encoder + decoder probe",
            "suite": "frozen",
            "family": "candidate-set reranking: constrained Top-K",
            "best_delta_miou": 0.00022,
            "best_delta_picture": 0.00130,
            "source": "results_topk_pairwise_rerank_decoder.md; results_constrained_topk_set_decoder.md",
            "notes": "tests whether oracle candidate-set headroom is recoverable by reranking; no meaningful recovery",
        },
        {
            "model": "Concerto decoder",
            "base_row": "frozen encoder + decoder probe",
            "suite": "adaptation",
            "family": "capacity-limited adaptation: fixed-rank LoRA",
            "best_delta_miou": -0.00280,
            "best_delta_picture": -0.00130,
            "source": "results_scannet_dec_lora_origin_perclass.md",
            "notes": "decoder-capacity-matched LoRA; same-head linear LoRA is positive but head-capacity confounded",
        },
        {
            "model": "Concerto linear",
            "base_row": "frozen encoder + linear head",
            "suite": "adaptation",
            "family": "capacity-limited adaptation: fixed-rank LoRA",
            "best_delta_miou": 0.0134,
            "best_delta_picture": 0.0289,
            "source": "results_scannet_lora_origin_perclass.md; results_scannet_linear_origin_oracle_actionability/",
            "notes": "protocol-matched to the linear-head family; strongest picture recovery among linear-head adaptation rows",
        },
        {
            "model": "Concerto linear",
            "base_row": "frozen encoder + linear head",
            "suite": "adaptation",
            "family": "LP-FT warm-start adaptation",
            "best_delta_miou": 0.0166,
            "best_delta_picture": 0.0125,
            "source": "results_scannet_lora_lpft_classsafe.md; results_scannet_lora_lpft_plain_oracle_actionability/",
            "notes": "protocol-matched to the linear-head family; strongest mIoU recovery among linear-head adaptation rows",
        },
        {
            "model": "Concerto decoder",
            "base_row": "frozen encoder + decoder probe",
            "suite": "adaptation",
            "family": "full fine-tuning",
            "best_delta_miou": 0.01870,
            "best_delta_picture": 0.01980,
            "source": "results_scannet_origin_fullft.md; results_scannet_origin_fullft_oracle_actionability/",
            "notes": "maximum practical adaptation budget; improves aggregate but leaves large oracle headroom",
        },
        {
            "model": "Sonata linear",
            "base_row": "released backbone + linear head",
            "suite": "frozen",
            "family": "class-prior correction / decoupled classifier",
            "best_delta_miou": 0.00170,
            "best_delta_picture": -0.00005,
            "source": "results_sonata_recovery_decoupled_classifier.md",
            "notes": "small aggregate gain; picture does not recover",
        },
        {
            "model": "Sonata linear",
            "base_row": "released backbone + linear head",
            "suite": "frozen",
            "family": "candidate-set reranking: constrained Top-K",
            "best_delta_miou": 0.0,
            "best_delta_picture": 0.00064,
            "source": "data/runs/sonata_recovery_topk/topk_pairwise_rerank_decoder.md",
            "notes": "no aggregate recovery; tiny picture-only movement",
        },
        {
            "model": "Sonata full FT",
            "base_row": "released backbone + full fine-tuned head",
            "suite": "adaptation",
            "family": "full fine-tuning",
            "best_delta_miou": 0.0684,
            "best_delta_picture": -0.0074,
            "source": "results_sonata_fullft_oracle_actionability/oracle_actionability_analysis.md",
            "notes": "full-FT improves aggregate relative to Sonata linear under the oracle evaluator, but does not improve picture",
        },
    ]
    for model in ("Utonia released stack", "PTv3 supervised"):
        fixed_suite_rows.append(
            {
                "model": model,
                "base_row": "released stack / protocol-specific head",
                "suite": "frozen/adaptation",
                "family": "6-family recovery suite",
                "best_delta_miou": "",
                "best_delta_picture": "",
                "source": "",
                "notes": "not run in a protocol-matched way yet; external rows report oracle/actionability diagnostics only",
            }
        )
    write_csv(OUT_DIR / "results_recoverability_fixed_suite_methods.csv", fixed_suite_rows)

    oracle_specs = [
        {
            "model": "Concerto decoder",
            "source": "results_oracle_actionability_analysis.md",
            "base_miou": 0.7778,
            "oracle2_miou": 0.9197,
            "oracle5_miou": 0.9775,
            "base_picture": 0.4034,
            "oracle2_picture": 0.8579,
            "oracle5_picture": 0.9427,
            "frozen_delta_miou": 0.00022,
            "frozen_delta_picture": 0.00130,
            "adapt_delta_miou": 0.01870,
            "adapt_delta_picture": 0.01980,
            "recovery_suite": "6-family suite; decoder-compatible families used here: class-prior, prototype/kNN, constrained Top-K, decoder-matched LoRA, full FT. LP-FT is tracked separately for the linear-head base.",
        },
        {
            "model": "Concerto linear",
            "source": "results_scannet_linear_origin_oracle_actionability/; results_scannet_lora_lpft_plain_oracle_actionability/",
            "base_miou": 0.7615,
            "oracle2_miou": 0.9171,
            "oracle5_miou": 0.9839,
            "base_picture": 0.4014,
            "oracle2_picture": 0.8013,
            "oracle5_picture": 0.9394,
            "frozen_delta_miou": "",
            "frozen_delta_picture": "",
            "adapt_delta_miou": 0.0166,
            "adapt_delta_picture": 0.0289,
            "recovery_suite": "6-family suite; linear-head-compatible adaptation families used here: fixed-rank LoRA and LP-FT. Frozen suite pending for this base row.",
        },
        {
            "model": "Sonata linear",
            "source": "results_sonata_scannet_oracle_actionability_analysis.md",
            "base_miou": 0.7086,
            "oracle2_miou": 0.8747,
            "oracle5_miou": 0.9670,
            "base_picture": 0.3582,
            "oracle2_picture": 0.6972,
            "oracle5_picture": 0.8867,
            "frozen_delta_miou": 0.00170,
            "frozen_delta_picture": 0.00064,
            "adapt_delta_miou": 0.0684,
            "adapt_delta_picture": -0.0074,
            "recovery_suite": "6-family suite partially run: class-prior and constrained Top-K frozen rows complete, prototype/kNN pending, full-FT adaptation integrated. LoRA/LP-FT not run.",
        },
        {
            "model": "Utonia released stack",
            "source": "results_utonia_scannet_oracle_actionability/oracle_actionability_analysis.md",
            "base_miou": 0.7574,
            "oracle2_miou": 0.9367,
            "oracle5_miou": 0.9908,
            "base_picture": 0.2952,
            "oracle2_picture": 0.9747,
            "oracle5_picture": 1.0000,
            "frozen_delta_miou": "",
            "frozen_delta_picture": "",
            "adapt_delta_miou": "",
            "adapt_delta_picture": "",
            "recovery_suite": "6-family recovery suite not run in a protocol-matched way; oracle/actionability diagnostics only",
        },
        {
            "model": "PTv3 supervised",
            "source": "results_ptv3_scannet20_oracle_actionability.md",
            "base_miou": 0.7745,
            "oracle2_miou": 0.9038,
            "oracle5_miou": 0.9690,
            "base_picture": 0.4908,
            "oracle2_picture": 0.8785,
            "oracle5_picture": 0.9952,
            "frozen_delta_miou": "",
            "frozen_delta_picture": "",
            "adapt_delta_miou": "",
            "adapt_delta_picture": "",
            "recovery_suite": "6-family recovery suite not run in a protocol-matched way; oracle/actionability diagnostics only",
        },
    ]

    rows = []
    for spec in oracle_specs:
        row = {
            "model": spec["model"],
            "base_miou": spec["base_miou"],
            "oracle2_miou": spec["oracle2_miou"],
            "oracle5_miou": spec["oracle5_miou"],
            "base_picture_iou": spec["base_picture"],
            "oracle2_picture_iou": spec["oracle2_picture"],
            "oracle5_picture_iou": spec["oracle5_picture"],
            "frozen_delta_miou_max": spec["frozen_delta_miou"],
            "frozen_delta_picture_max": spec["frozen_delta_picture"],
            "frozen_Rrec2_miou": _recovery(spec["base_miou"], spec["oracle2_miou"], spec["frozen_delta_miou"]),
            "frozen_Rrec5_miou": _recovery(spec["base_miou"], spec["oracle5_miou"], spec["frozen_delta_miou"]),
            "frozen_Rrec2_picture": _recovery(spec["base_picture"], spec["oracle2_picture"], spec["frozen_delta_picture"]),
            "frozen_Rrec5_picture": _recovery(spec["base_picture"], spec["oracle5_picture"], spec["frozen_delta_picture"]),
            "adapt_delta_miou_max": spec["adapt_delta_miou"],
            "adapt_delta_picture_max": spec["adapt_delta_picture"],
            "adapt_Rrec2_miou": _recovery(spec["base_miou"], spec["oracle2_miou"], spec["adapt_delta_miou"]),
            "adapt_Rrec5_miou": _recovery(spec["base_miou"], spec["oracle5_miou"], spec["adapt_delta_miou"]),
            "adapt_Rrec2_picture": _recovery(spec["base_picture"], spec["oracle2_picture"], spec["adapt_delta_picture"]),
            "adapt_Rrec5_picture": _recovery(spec["base_picture"], spec["oracle5_picture"], spec["adapt_delta_picture"]),
            "recovery_suite": spec["recovery_suite"],
            "source": spec["source"],
        }
        rows.append(row)

    write_csv(OUT_DIR / "results_recoverability_rrec_max.csv", rows)
    lines = [
        "# Recoverability Table: R_rec^max",
        "",
        "Definition: `R_rec^max = (best non-oracle score - base score) / (oracle score - base score)`.",
        "This table separates available oracle headroom from the fraction recovered by the pre-specified six-family recovery suite.",
        "`R_rec` is computed only within protocol-matched base/readout rows; LP-FT belongs to the Concerto linear-head family and is not mixed into the decoder-probe denominator.",
        "",
        "Main-suite recovery families:",
        "",
        "1. Class-prior correction / decoupled classifier.",
        "2. Nonparametric feature readout: prototype or kNN.",
        "3. Candidate-set reranking: constrained Top-K.",
        "4. Capacity-limited adaptation: fixed-rank LoRA.",
        "5. LP-FT warm-start adaptation.",
        "6. Full fine-tuning.",
        "",
        "Exploratory CoDA/CIDA/region/proposal/subgroup attempts are intentionally not included in `R_rec^max`; they belong in the appendix.",
        "",
        "| model | base mIoU | oracle@2 | oracle@5 | frozen Δ mIoU | frozen R_rec@2 | adaptation Δ mIoU | adaptation R_rec@2 | base picture | picture oracle@2 | frozen Δ picture | frozen picture R_rec@2 | adaptation Δ picture | adaptation picture R_rec@2 | suite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['model']}` | `{fmt(r['base_miou'])}` | `{fmt(r['oracle2_miou'])}` | `{fmt(r['oracle5_miou'])}` | "
            f"`{fmt(r['frozen_delta_miou_max'])}` | `{fmt_pct(r['frozen_Rrec2_miou'])}` | "
            f"`{fmt(r['adapt_delta_miou_max'])}` | `{fmt_pct(r['adapt_Rrec2_miou'])}` | "
            f"`{fmt(r['base_picture_iou'])}` | `{fmt(r['oracle2_picture_iou'])}` | "
            f"`{fmt(r['frozen_delta_picture_max'])}` | `{fmt_pct(r['frozen_Rrec2_picture'])}` | "
            f"`{fmt(r['adapt_delta_picture_max'])}` | `{fmt_pct(r['adapt_Rrec2_picture'])}` | `{r['recovery_suite']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Concerto has large top-2/top-5 oracle headroom, but the fixed frozen suite recovers essentially none of it.",
            "- Full fine-tuning recovers a nonzero but still small fraction of the oracle headroom; it improves aggregate accuracy but does not close the actionability gap.",
            "- LP-FT is a linear-head-family adaptation row. It should be reported as protocol-matched to the Concerto linear base, not as recovery for the decoder-probe oracle denominator.",
            "- Sonata now has partial protocol-matched recovery coverage: class-prior and constrained Top-K frozen rows are complete, prototype/kNN is pending, and full fine-tuning provides the high-budget adaptation row. Aggregate recovery is possible under full FT, but picture recovery remains poor.",
            "- For Utonia and PTv3, the six-family recovery suite has not been run in a protocol-matched way. Keep their recovery interpretation limited to oracle/actionability comparisons unless custom recovery paths are added.",
        ]
    )
    (OUT_DIR / "results_recoverability_rrec_max.md").write_text("\n".join(lines) + "\n")

    detail_lines = [
        "# Fixed Recovery Suite: Family-Level Rows",
        "",
        "These are the only recovery families included in the main-paper `R_rec^max` suite.",
        "",
        "| model | base row | suite | family | best ΔmIoU | best Δpicture | source | notes |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for r in fixed_suite_rows:
        detail_lines.append(
            f"| `{r['model']}` | `{r['base_row']}` | `{r['suite']}` | `{r['family']}` | `{fmt(r['best_delta_miou'])}` | "
            f"`{fmt(r['best_delta_picture'])}` | `{r['source']}` | {r['notes']} |"
        )
    detail_lines.extend(
        [
            "",
            "## Appendix-only exploratory families",
            "",
            "CoDA, CIDA, latent subgroup readout, region/superpoint smoothing, PHRD/PVD, proposal boosting, and other process-driven variants should be reported as exploratory recovery attempts, not as part of the pre-specified main suite.",
        ]
    )
    (OUT_DIR / "results_recoverability_fixed_suite_methods.md").write_text("\n".join(detail_lines) + "\n")


def build_binding_profile_figure() -> None:
    src = OUT_DIR / "results_binding_profile_summary.csv"
    if not src.exists():
        return
    rows = read_csv(src)
    # Keep the figure compact by plotting the rows that can be compared without
    # forcing heterogeneous quantities into a single false scalar.
    labels = [r["model"] for r in rows]
    metrics = [
        ("base", "top1_or_miou"),
        ("oracle/top2 proxy", "top2_oracle_proxy"),
        ("oracle/top5 proxy", "top5_oracle_proxy"),
        ("random20 damage", "random_keep20_down"),
        ("structured20 damage", "structured_keep20_down"),
        ("feature/xyz-zero damage", "feature_zero_down"),
    ]
    values = []
    for r in rows:
        row_vals = []
        for _, key in metrics:
            val = r.get(key, "")
            row_vals.append(float(val) if val not in ("", None) else float("nan"))
        values.append(row_vals)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        (OUT_DIR / "results_binding_profile_summary_figure_note.md").write_text(
            f"# Binding Profile Figure\n\nMatplotlib is unavailable, so only CSV/Markdown tables were written.\n\nError: `{exc}`\n"
        )
        return

    arr = np.array(values, dtype=float)
    fig_h = max(6.0, 0.42 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(len(metrics)), [m[0] for m in metrics], rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Binding Profile Summary")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="white" if arr[i, j] > 0.55 else "black", fontsize=8)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", color="#666666", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("score / damage")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results_binding_profile_summary.png", dpi=220)
    fig.savefig(OUT_DIR / "results_binding_profile_summary.pdf")


def build_binding_profile_panel_figure() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        (OUT_DIR / "results_binding_profile_summary_panels_note.md").write_text(
            f"# Binding Profile Panel Figure\n\nMatplotlib is unavailable.\n\nError: `{exc}`\n"
        )
        return

    causal_path = OUT_DIR / "results_main_variant_causal_battery_paper_table.csv"
    coord_path = OUT_DIR / "results_coord_mlp_rival_six_dataset_calibration_s3dis_highval.csv"
    profile_path = OUT_DIR / "results_binding_profile_summary.csv"
    if not causal_path.exists() or not coord_path.exists() or not profile_path.exists():
        return

    causal = {r["dataset"]: r for r in read_csv(causal_path)}
    coord = {r["dataset"]: r for r in read_csv(coord_path)}
    profile = read_csv(profile_path)

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), constrained_layout=True)

    # Panel A: pretext-side target corruption and coordinate-rival closure.
    datasets = [d for d in ("arkit", "scannet", "scannetpp", "s3dis", "hm3d", "structured3d") if d in causal]
    x = np.arange(len(datasets))
    width = 0.38
    bpre = [float(causal[d]["b_pre_relative"]) for d in datasets]
    closures = [float(coord[d]["closure_fraction_mean"]) if d in coord and coord[d]["closure_fraction_mean"] != "" else np.nan for d in datasets]
    axes[0].bar(x - width / 2, bpre, width, label="target-corruption B_pre / clean loss", color="#4666a6")
    axes[0].bar(x + width / 2, closures, width, label="coord-rival closure", color="#c17a2c")
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_xticks(x, datasets, rotation=20, ha="right")
    axes[0].set_ylim(0, max(max(bpre), max(closures)) * 1.25)
    axes[0].set_ylabel("relative score")
    axes[0].set_title("A. Train-side counterfactuals are positive, but coord-rival closure is dataset-dependent")
    axes[0].legend(loc="upper left", ncols=2, fontsize=9)

    # Panel B: actionability.
    action_rows = [r for r in profile if r["domain"] in ("scene", "object")]
    action_labels = [f"{r['model']}\n{r['task'].split()[0]}" for r in action_rows]
    x = np.arange(len(action_rows))
    width = 0.25
    base = [float(r["top1_or_miou"]) for r in action_rows]
    top2 = [float(r["top2_oracle_proxy"]) for r in action_rows]
    top5 = [float(r["top5_oracle_proxy"]) for r in action_rows]
    axes[1].bar(x - width, base, width, label="base", color="#4d7f72")
    axes[1].bar(x, top2, width, label="top-2/proxy", color="#d0a13a")
    axes[1].bar(x + width, top5, width, label="top-5/proxy", color="#8f5aa9")
    axes[1].set_xticks(x, action_labels, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("score / hit rate")
    axes[1].set_title("B. Candidate/actionability headroom remains after standard readout")
    axes[1].legend(loc="lower right", ncols=3, fontsize=9)

    # Panel C: support stress.
    support_rows = [r for r in profile if "support" in r["domain"]]
    support_labels = [r["model"] for r in support_rows]
    x = np.arange(len(support_rows))
    width = 0.25
    random_damage = [float(r["random_keep20_down"]) if r["random_keep20_down"] != "" else np.nan for r in support_rows]
    structured_damage = [float(r["structured_keep20_down"]) if r["structured_keep20_down"] != "" else np.nan for r in support_rows]
    zero_damage = [float(r["feature_zero_down"]) if r["feature_zero_down"] != "" else np.nan for r in support_rows]
    axes[2].bar(x - width, random_damage, width, label="random keep20 damage", color="#5f8ec6")
    axes[2].bar(x, structured_damage, width, label="structured keep20 damage", color="#c45a4f")
    axes[2].bar(x + width, zero_damage, width, label="feature/xyz-zero damage", color="#5e5e5e")
    axes[2].set_xticks(x, support_labels, rotation=25, ha="right")
    axes[2].set_ylim(0, max([v for v in random_damage + structured_damage + zero_damage if np.isfinite(v)]) * 1.2)
    axes[2].set_ylabel("drop from clean")
    axes[2].set_title("C. Random point drop is weaker than structured/semantic support stress")
    axes[2].legend(loc="upper left", ncols=3, fontsize=9)

    fig.savefig(OUT_DIR / "results_binding_profile_summary_panels.png", dpi=220)
    fig.savefig(OUT_DIR / "results_binding_profile_summary_panels.pdf")


EPOCH_RE = re.compile(r"\[Training\] EPOCH:\s+(\d+).*?Losses = \['([^']+)'\]")
DIAG_RE = re.compile(r"diag\(loss_main=([0-9.]+).*?recon_cd_l1=([0-9.]+).*?recon_cd_l2=([0-9.]+)")


def parse_pretrain_log(path: Path) -> dict:
    epochs = {}
    diag_last = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epochs[int(m.group(1))] = float(m.group(2))
        d = DIAG_RE.search(line)
        if d:
            # Keep the last logged batch diagnostic.
            diag_last = {
                "loss_main": float(d.group(1)),
                "recon_cd_l1": float(d.group(2)),
                "recon_cd_l2": float(d.group(3)),
            }
    return {"epochs": epochs, "diag_last": diag_last}


def build_object_pretext_summary() -> None:
    specs = [
        (
            "PointGPT-S official masked",
            None,
            "cdl12 + mask_ratio0.7",
            "official checkpoint only; pretrain log unavailable in rebuilt local environment",
        ),
        (
            "PointGPT-S no-mask",
            NEPA_ROOT / "PointGPT/experiments/pretrain_nomask/PointGPT-S/pgpt_s_nomask_e300/20260422_021328.log",
            "cdl12",
            "",
        ),
        (
            "PointGPT-S no-mask order-random",
            NEPA_ROOT / "PointGPT/experiments/pretrain_nomask_orderrandom/PointGPT-S/pgpt_s_nomask_ordrand_e300/20260423_200226.log",
            "cdl12 + random token order",
            "",
        ),
    ]
    rows = []
    for name, path, variant, note in specs:
        if path is None:
            rows.append(
                {
                    "variant": name,
                    "pretext": variant,
                    "epoch0_loss": "",
                    "epoch50_loss": "",
                    "epoch100_loss": "",
                    "epoch200_loss": "",
                    "final_epoch": "",
                    "final_loss": "",
                    "loss_main_last_logged": "",
                    "recon_cd_l1_last_logged": "",
                    "recon_cd_l2_last_logged": "",
                    "source_log": "",
                    "note": note,
                }
            )
            continue
        if not path.exists():
            continue
        parsed = parse_pretrain_log(path)
        epochs = parsed["epochs"]
        final_epoch = max(epochs) if epochs else None
        rows.append(
            {
                "variant": name,
                "pretext": variant,
                "epoch0_loss": epochs.get(0, ""),
                "epoch50_loss": epochs.get(50, ""),
                "epoch100_loss": epochs.get(100, ""),
                "epoch200_loss": epochs.get(200, ""),
                "final_epoch": final_epoch,
                "final_loss": epochs.get(final_epoch, "") if final_epoch is not None else "",
                "loss_main_last_logged": parsed["diag_last"].get("loss_main", ""),
                "recon_cd_l1_last_logged": parsed["diag_last"].get("recon_cd_l1", ""),
                "recon_cd_l2_last_logged": parsed["diag_last"].get("recon_cd_l2", ""),
                "source_log": str(path.relative_to(NEPA_ROOT)),
                "note": note,
            }
        )
    write_csv(NEPA_ROOT / "results/pointgpt_object_pretext_summary.csv", rows)
    lines = [
        "# PointGPT Object-Side Pretext Summary",
        "",
        "| variant | pretext | epoch0 | epoch50 | epoch100 | epoch200 | final | last loss_main | last cd_l1 | last cd_l2 | note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['variant']}` | `{r['pretext']}` | `{fmt(r['epoch0_loss'], 4)}` | `{fmt(r['epoch50_loss'], 4)}` | "
            f"`{fmt(r['epoch100_loss'], 4)}` | `{fmt(r['epoch200_loss'], 4)}` | `{fmt(r['final_loss'], 4)}` | "
            f"`{fmt(r['loss_main_last_logged'], 5)}` | `{fmt(r['recon_cd_l1_last_logged'], 5)}` | `{fmt(r['recon_cd_l2_last_logged'], 5)}` | {r.get('note', '')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The official masked PointGPT-S row uses the public checkpoint; its original pretraining log is unavailable in this rebuilt environment, so it should not be used for pretext-side trajectory claims.",
            "- Both no-mask and no-mask order-randomized pretrains are non-null optimization runs: loss drops sharply from epoch 0 to the final checkpoint.",
            "- The order-randomized row optimizes less well than the no-mask row, so causal order is more binding than mask removal alone, but it does not collapse downstream utility on ScanObjectNN.",
            "- Older pointNEPA mask-off / vit-shift raw logs referenced in the active ledger are not present in this rebuilt local environment; keep those as sidecar-ledger rows unless the original log archive is restored.",
        ]
    )
    (NEPA_ROOT / "results/pointgpt_object_pretext_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    build_six_dataset_table()
    build_six_dataset_coord_rival_calibration()
    build_shapenetpart_support_stress_table()
    build_binding_profile()
    build_binding_profile_figure()
    build_binding_profile_panel_figure()
    build_recoverability_table()
    build_object_pretext_summary()
    print("[write] tools/concerto_projection_shortcut/results_main_variant_causal_battery_paper_table.md")
    print("[write] tools/concerto_projection_shortcut/results_coord_mlp_rival_six_dataset_calibration.md")
    print("[write] tools/concerto_projection_shortcut/results_binding_profile_summary.md")
    print("[write] tools/concerto_projection_shortcut/results_binding_profile_summary.png")
    print("[write] tools/concerto_projection_shortcut/results_binding_profile_summary_panels.png")
    print("[write] tools/concerto_projection_shortcut/results_recoverability_rrec_max.md")
    print("[write] tools/concerto_projection_shortcut/results_recoverability_fixed_suite_methods.md")
    print("[write] 3D-NEPA/results/ptgpt_shapenetpart_support_stress_paper_table.md")
    print("[write] 3D-NEPA/results/pointgpt_object_pretext_summary.md")


if __name__ == "__main__":
    main()
