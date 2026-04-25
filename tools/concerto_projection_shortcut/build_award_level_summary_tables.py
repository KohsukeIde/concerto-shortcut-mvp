#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=4) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def object_rows() -> list[dict]:
    rows = []
    specs = [
        ("PointGPT-S official", NEPA_ROOT / "results/ptgpt_readout_official_objbg_full.json", None),
        ("PointGPT-S no-mask", NEPA_ROOT / "results/ptgpt_readout_nomask_objbg_full.json", NEPA_ROOT / "results/ptgpt_stress_nomask_objbg_full.json"),
        ("PointGPT-S no-mask order-random", NEPA_ROOT / "results/ptgpt_nomask_ordrand_objbg_readout_full.json", NEPA_ROOT / "results/ptgpt_nomask_ordrand_objbg_stress_full.json"),
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
    rows = scene_readout_rows() + support_rows_from_scene() + support_rows_from_utonia() + object_rows()
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
            "- Utonia support stress uses the released Utonia inference/head path; its feature-zero row is not directly comparable to color-dependent rows because the public Utonia path is largely raw-feature agnostic.",
            "- Object rows use ScanObjectNN `obj_bg` readout and support-stress audits.",
        ]
    )
    (OUT_DIR / "results_binding_profile_summary.md").write_text("\n".join(lines) + "\n")


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
            "PointGPT-S no-mask",
            NEPA_ROOT / "PointGPT/experiments/pretrain_nomask/PointGPT-S/pgpt_s_nomask_e300/20260422_021328.log",
            "cdl12",
        ),
        (
            "PointGPT-S no-mask order-random",
            NEPA_ROOT / "PointGPT/experiments/pretrain_nomask_orderrandom/PointGPT-S/pgpt_s_nomask_ordrand_e300/20260423_200226.log",
            "cdl12 + random token order",
        ),
    ]
    rows = []
    for name, path, variant in specs:
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
            }
        )
    write_csv(NEPA_ROOT / "results/pointgpt_object_pretext_summary.csv", rows)
    lines = [
        "# PointGPT Object-Side Pretext Summary",
        "",
        "| variant | pretext | epoch0 | epoch50 | epoch100 | epoch200 | final | last loss_main | last cd_l1 | last cd_l2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['variant']}` | `{r['pretext']}` | `{fmt(r['epoch0_loss'], 4)}` | `{fmt(r['epoch50_loss'], 4)}` | "
            f"`{fmt(r['epoch100_loss'], 4)}` | `{fmt(r['epoch200_loss'], 4)}` | `{fmt(r['final_loss'], 4)}` | "
            f"`{fmt(r['loss_main_last_logged'], 5)}` | `{fmt(r['recon_cd_l1_last_logged'], 5)}` | `{fmt(r['recon_cd_l2_last_logged'], 5)}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Both no-mask and no-mask order-randomized pretrains are non-null optimization runs: loss drops sharply from epoch 0 to the final checkpoint.",
            "- The order-randomized row optimizes less well than the no-mask row, so causal order is more binding than mask removal alone, but it does not collapse downstream utility on ScanObjectNN.",
            "- Older pointNEPA mask-off / vit-shift raw logs referenced in the active ledger are not present in this rebuilt local environment; keep those as sidecar-ledger rows unless the original log archive is restored.",
        ]
    )
    (NEPA_ROOT / "results/pointgpt_object_pretext_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    build_six_dataset_table()
    build_binding_profile()
    build_object_pretext_summary()
    print("[write] tools/concerto_projection_shortcut/results_main_variant_causal_battery_paper_table.md")
    print("[write] tools/concerto_projection_shortcut/results_binding_profile_summary.md")
    print("[write] 3D-NEPA/results/pointgpt_object_pretext_summary.md")


if __name__ == "__main__":
    main()
