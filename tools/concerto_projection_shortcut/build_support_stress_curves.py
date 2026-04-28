#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "tools/concerto_projection_shortcut/results_support_stress_curves.csv"
OUT_RATIO_CSV = ROOT / "tools/concerto_projection_shortcut/results_support_stress_keep20_ratios.csv"
OUT_MD = ROOT / "tools/concerto_projection_shortcut/results_support_stress_curves.md"
OUT_PNG = ROOT / "tools/concerto_projection_shortcut/results_support_stress_curves.png"
OUT_PDF = ROOT / "tools/concerto_projection_shortcut/results_support_stress_curves.pdf"


SCENE_FILES = [
    (
        "scene",
        "Concerto-D",
        "ScanNet20",
        ROOT / "tools/concerto_projection_shortcut/results_support_severity_concerto_decoder.csv",
    ),
    (
        "scene",
        "Concerto-L",
        "ScanNet20",
        ROOT / "tools/concerto_projection_shortcut/results_support_severity_concerto_linear.csv",
    ),
    (
        "scene",
        "Sonata-L",
        "ScanNet20",
        ROOT / "tools/concerto_projection_shortcut/results_support_severity_sonata_linear.csv",
    ),
    (
        "scene",
        "Utonia",
        "ScanNet20",
        ROOT / "tools/concerto_projection_shortcut/results_support_severity_utonia/utonia_scannet_support_stress.csv",
    ),
    (
        "scene",
        "PTv3",
        "ScanNet20",
        ROOT / "tools/concerto_projection_shortcut/results_support_severity_ptv3_scannet20.csv",
    ),
    (
        "scene",
        "PTv3",
        "ScanNet200",
        ROOT / "tools/concerto_projection_shortcut/results_support_severity_ptv3_scannet200.csv",
    ),
    (
        "scene",
        "PTv3",
        "S3DIS Area-5",
        ROOT / "tools/concerto_projection_shortcut/results_support_severity_ptv3_s3dis.csv",
    ),
]

SCANOBJECTNN_FILES = [
    (
        "object",
        "PointGPT-S official",
        "ScanObjectNN obj_bg",
        ROOT / "3D-NEPA/results/ptgpt_stress_official_objbg_severity.json",
    ),
    (
        "object",
        "PointGPT-S no-mask",
        "ScanObjectNN obj_bg",
        ROOT / "3D-NEPA/results/ptgpt_stress_nomask_objbg_severity.json",
    ),
    (
        "object",
        "PointGPT-S no-mask order-random",
        "ScanObjectNN obj_bg",
        ROOT / "3D-NEPA/results/ptgpt_stress_nomask_ordrand_objbg_severity.json",
    ),
    (
        "object",
        "PointGPT-S mask-on order-random",
        "ScanObjectNN obj_bg",
        ROOT / "3D-NEPA/results/ptgpt_stress_masked_ordrand_objbg_severity.json",
    ),
]

SHAPENETPART_FILES = [
    (
        "object",
        "PointGPT-S official",
        "ShapeNetPart",
        ROOT / "3D-NEPA/results/ptgpt_shapenetpart_official_support_stress.json",
    ),
    (
        "object",
        "PointGPT-S no-mask",
        "ShapeNetPart",
        ROOT / "3D-NEPA/results/ptgpt_shapenetpart_nomask_support_stress.json",
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def condition_name(row: dict[str, object]) -> str:
    return str(row.get("variant") or row.get("condition") or row.get("name") or "")


def parse_stress(name: str, kind: str = "", fixed_count: float = 0.0) -> tuple[str, str]:
    if name in {"clean", "clean_voxel"} or kind == "clean":
        return "clean", "1.0"
    for prefix, stress in [
        ("random_keep", "random_keep"),
        ("structured_keep", "structured_keep"),
        ("classwise_keep", "classwise_keep"),
        ("part_keep", "part_keep_per_part"),
        ("masked_model_keep", "object_style_keep"),
    ]:
        if name.startswith(prefix):
            tail = name[len(prefix) :]
            tail = tail.replace("0p", "0.").replace("_per_part", "")
            if tail in {"80", "50", "20", "10"}:
                tail = f"0.{tail[0]}" if tail != "10" else "0.1"
            return stress, tail
    if "structured" in name and "_keep" in name:
        keep = name.rsplit("_keep", 1)[-1].replace("0p", "0.")
        return "structured_keep", keep
    if name.startswith("fixed_points_"):
        return "fixed_budget", name.replace("fixed_points_", "")
    if name == "part_drop_largest":
        return "part_drop_largest", "largest"
    if name == "xyz_zero":
        return "xyz_zero", "1.0"
    if name.startswith("feature_zero") or name == "feature_zero":
        return "feature_zero", "1.0"
    if fixed_count:
        return "fixed_budget", str(int(fixed_count))
    if kind:
        return kind, ""
    return name, ""


def normalize_score(score: float, clean: float, null: float) -> tuple[float, float]:
    raw_damage = clean - score
    denom = clean - null
    b_down = raw_damage / denom if abs(denom) > 1e-12 else float("nan")
    return raw_damage, b_down


def choose_scene_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    if "score_space" not in rows[0]:
        return rows
    full = [row for row in rows if row.get("score_space") == "full_nn"]
    return full if full else rows


def scene_entries(domain: str, model: str, task: str, path: Path) -> list[dict[str, object]]:
    rows = choose_scene_rows(read_csv(path))
    if not rows:
        return []
    clean_row = next(row for row in rows if condition_name(row) in {"clean", "clean_voxel"})
    clean = safe_float(clean_row.get("mIoU") or clean_row.get("miou"))
    null_candidates = [
        row
        for row in rows
        if parse_stress(condition_name(row), row.get("kind", ""))[0] in {"feature_zero", "xyz_zero"}
    ]
    null = safe_float((null_candidates[-1].get("mIoU") or null_candidates[-1].get("miou"))) if null_candidates else 0.0
    entries = []
    for row in rows:
        name = condition_name(row)
        stress_type, severity = parse_stress(name, row.get("kind", ""), safe_float(row.get("fixed_count"), 0.0))
        score = safe_float(row.get("mIoU") or row.get("miou"))
        if math.isnan(score):
            continue
        raw_damage, b_down = normalize_score(score, clean, null)
        entries.append(
            {
                "domain": domain,
                "model": model,
                "task": task,
                "stress_type": stress_type,
                "severity": severity,
                "condition": name,
                "clean_score": clean,
                "stress_score": score,
                "raw_damage": raw_damage,
                "null_score": null,
                "B_down": b_down,
                "observed_keep_frac": safe_float(row.get("observed_keep_frac") or row.get("mean_keep_frac")),
                "metric": "mIoU",
                "source": str(path.relative_to(ROOT)),
            }
        )
    return entries


def json_conditions(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    return list(data.get("conditions", []))


def scanobject_entries(domain: str, model: str, task: str, path: Path) -> list[dict[str, object]]:
    rows = json_conditions(path)
    clean = safe_float(rows[0].get("acc"))
    null_row = next((row for row in rows if row.get("name") == "xyz_zero"), None)
    null = safe_float(null_row.get("acc")) if null_row else 0.0
    entries = []
    for row in rows:
        name = str(row.get("name", ""))
        stress_type, severity = parse_stress(name)
        score = safe_float(row.get("acc"))
        raw_damage, b_down = normalize_score(score, clean, null)
        entries.append(
            {
                "domain": domain,
                "model": model,
                "task": task,
                "stress_type": stress_type,
                "severity": severity,
                "condition": name,
                "clean_score": clean,
                "stress_score": score,
                "raw_damage": raw_damage,
                "null_score": null,
                "B_down": b_down,
                "observed_keep_frac": "",
                "metric": "accuracy",
                "source": str(path.relative_to(ROOT)),
            }
        )
    return entries


def shapenetpart_entries(domain: str, model: str, task: str, path: Path) -> list[dict[str, object]]:
    rows = json_conditions(path)
    clean_row = next(row for row in rows if row.get("condition") == "clean")
    clean = safe_float(clean_row.get("class_avg_iou"))
    null_row = next((row for row in rows if row.get("condition") == "xyz_zero"), None)
    null = safe_float(null_row.get("class_avg_iou")) if null_row else 0.0
    entries = []
    for row in rows:
        name = str(row.get("condition", ""))
        stress_type, severity = parse_stress(name)
        score = safe_float(row.get("class_avg_iou"))
        raw_damage, b_down = normalize_score(score, clean, null)
        entries.append(
            {
                "domain": domain,
                "model": model,
                "task": task,
                "stress_type": stress_type,
                "severity": severity,
                "condition": name,
                "clean_score": clean,
                "stress_score": score,
                "raw_damage": raw_damage,
                "null_score": null,
                "B_down": b_down,
                "observed_keep_frac": "",
                "metric": "class_avg_iou",
                "source": str(path.relative_to(ROOT)),
            }
        )
    return entries


def all_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for item in SCENE_FILES:
        if item[3].exists():
            entries.extend(scene_entries(*item))
    for item in SCANOBJECTNN_FILES:
        if item[3].exists():
            entries.extend(scanobject_entries(*item))
    for item in SHAPENETPART_FILES:
        if item[3].exists():
            entries.extend(shapenetpart_entries(*item))
    return entries


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "model",
        "task",
        "metric",
        "stress_type",
        "severity",
        "condition",
        "clean_score",
        "stress_score",
        "raw_damage",
        "null_score",
        "B_down",
        "observed_keep_frac",
        "source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def keep20_ratio_rows(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in entries:
        key = (str(row["domain"]), str(row["model"]), str(row["task"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    out = []
    for (domain, model, task), rows in sorted(grouped.items()):
        random20 = rows.get("random_keep0p2") or rows.get("random_keep20")
        structured20 = rows.get("structured_b64_keep0p2") or rows.get("structured_b1p28m_keep0p2") or rows.get("structured_keep20")
        object20 = rows.get("masked_model_keep0p2")
        part_drop = rows.get("part_drop_largest")
        if not random20:
            continue
        random_damage = float(random20["raw_damage"])
        def ratio(other: dict[str, object] | None) -> float:
            if other is None or abs(random_damage) < 1e-12:
                return float("nan")
            return float(other["raw_damage"]) / random_damage
        out.append(
            {
                "domain": domain,
                "model": model,
                "task": task,
                "random20_damage": random_damage,
                "structured20_damage": float(structured20["raw_damage"]) if structured20 else "",
                "structured_over_random": ratio(structured20),
                "object20_or_partdrop_damage": float((object20 or part_drop)["raw_damage"]) if (object20 or part_drop) else "",
                "object_or_part_over_random": ratio(object20 or part_drop),
            }
        )
    return out


def write_ratios(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "domain",
        "model",
        "task",
        "random20_damage",
        "structured20_damage",
        "structured_over_random",
        "object20_or_partdrop_damage",
        "object_or_part_over_random",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(entries: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Scene: ScanNet20", lambda r: r["domain"] == "scene" and r["task"] == "ScanNet20"),
        ("Auxiliary Scene Datasets", lambda r: r["domain"] == "scene" and r["task"] != "ScanNet20"),
        ("Object: ScanObjectNN obj_bg", lambda r: r["task"] == "ScanObjectNN obj_bg"),
        ("Object: ShapeNetPart", lambda r: r["task"] == "ShapeNetPart"),
    ]
    stress_order = ["random_keep", "structured_keep", "object_style_keep", "fixed_budget", "part_drop_largest", "part_keep_per_part"]
    stress_style = {
        "random_keep": ("#2C7BB6", "-"),
        "structured_keep": ("#D7191C", "-"),
        "object_style_keep": ("#FDAE61", "--"),
        "fixed_budget": ("#5E3C99", ":"),
        "part_drop_largest": ("#F46D43", "--"),
        "part_keep_per_part": ("#1A9641", "--"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (title, pred) in zip(axes.ravel(), panels):
        rows = [r for r in entries if pred(r)]
        for model in sorted({str(r["model"]) for r in rows}):
            model_rows = [r for r in rows if r["model"] == model]
            for stress in stress_order:
                sr = [r for r in model_rows if r["stress_type"] == stress]
                numeric = []
                for r in sr:
                    try:
                        numeric.append((float(r["severity"]), float(r["B_down"])))
                    except (TypeError, ValueError):
                        if stress == "part_drop_largest":
                            numeric.append((0.0, float(r["B_down"])))
                if not numeric:
                    continue
                numeric.sort(key=lambda x: x[0])
                color, ls = stress_style.get(stress, ("black", "-"))
                ax.plot(
                    [x for x, _ in numeric],
                    [y for _, y in numeric],
                    label=f"{model} {stress}",
                    color=color,
                    linestyle=ls,
                    marker="o",
                    markersize=3,
                    alpha=0.75,
                )
        ax.set_title(title)
        ax.set_xlabel("severity: keep ratio / fixed budget")
        ax.set_ylabel("B_down = damage / null damage")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=-0.05)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # Keep only stress-type legend entries to avoid an unreadable model x stress legend.
    proxy = []
    proxy_labels = []
    for stress in stress_order:
        if stress in stress_style:
            color, ls = stress_style[stress]
            proxy.append(plt.Line2D([0], [0], color=color, linestyle=ls, marker="o", markersize=4))
            proxy_labels.append(stress)
    fig.legend(proxy, proxy_labels, loc="lower center", ncol=3, frameon=False)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def write_markdown(entries: list[dict[str, object]], ratio_rows: list[dict[str, object]]) -> None:
    def fmt(value: object) -> str:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return "--"
        if math.isnan(value_f):
            return "--"
        return f"{value_f:.2f}"

    lines = [
        "# Support-Stress Curves",
        "",
        "This file normalizes scene and object support-stress results into a single schema.",
        "`B_down` is `(clean_score - stress_score) / (clean_score - null_score)`, where the null row is feature-zero for scene models and xyz-zero for object models when available.",
        "",
        "## Outputs",
        "",
        f"- Unified CSV: `{OUT_CSV.relative_to(ROOT)}`",
        f"- Keep20 ratio CSV: `{OUT_RATIO_CSV.relative_to(ROOT)}`",
        f"- Figure PNG: `{OUT_PNG.relative_to(ROOT)}`",
        f"- Figure PDF: `{OUT_PDF.relative_to(ROOT)}`",
        "",
        "## Keep20 Stress Ratios",
        "",
        "| domain | model | task | random20 damage | structured/random | object-or-part/random |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in ratio_rows:
        lines.append(
            f"| `{row['domain']}` | `{row['model']}` | `{row['task']}` | "
            f"{float(row['random20_damage']):.4f} | {fmt(row['structured_over_random'])} | "
            f"{fmt(row['object_or_part_over_random'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Random keep is consistently a weaker stress than structured or object/part-aware removal in the same task family.",
        "- This supports the scoped claim that random point-drop robustness is an insufficient robustness test.",
        "- These curves do not by themselves prove an architecture-level cause; architecture claims require grouping or patchization ablations.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    entries = all_entries()
    write_csv(OUT_CSV, entries)
    ratio_rows = keep20_ratio_rows(entries)
    write_ratios(OUT_RATIO_CSV, ratio_rows)
    write_plot(entries)
    write_markdown(entries, ratio_rows)
    print(json.dumps({"rows": len(entries), "ratio_rows": len(ratio_rows), "csv": str(OUT_CSV)}, indent=2))


if __name__ == "__main__":
    main()
