#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUTS = [
    ("PointGPT-S official", ROOT / "3D-NEPA/results/ptgpt_shapenetpart_official_grouping_ablation.json"),
    ("PointGPT-S no-mask", ROOT / "3D-NEPA/results/ptgpt_shapenetpart_nomask_grouping_ablation.json"),
]
OUT_CSV = ROOT / "tools/concerto_projection_shortcut/results_shapenetpart_grouping_ablation.csv"
OUT_MD = ROOT / "tools/concerto_projection_shortcut/results_shapenetpart_grouping_ablation.md"


def load_rows():
    rows = []
    for variant, path in INPUTS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        table = {}
        for row in payload["rows"]:
            table.setdefault(row["group_mode"], {})[row["condition"]] = row
        fps_clean = table.get("fps_knn", {}).get("clean", {})
        fps_clean_iou = float(fps_clean.get("class_avg_iou", "nan"))
        for group_mode, values in sorted(table.items()):
            clean = values.get("clean")
            if clean is None:
                continue
            random20 = values.get("random_keep20")
            structured20 = values.get("structured_keep20")
            part_drop = values.get("part_drop_largest")
            part_keep = values.get("part_keep20_per_part")
            xyz_zero = values.get("xyz_zero")
            clean_iou = float(clean["class_avg_iou"])
            random_iou = float(random20["class_avg_iou"]) if random20 else float("nan")
            structured_iou = float(structured20["class_avg_iou"]) if structured20 else float("nan")
            part_drop_iou = float(part_drop["class_avg_iou"]) if part_drop else float("nan")
            part_keep_iou = float(part_keep["class_avg_iou"]) if part_keep else float("nan")
            xyz_zero_iou = float(xyz_zero["class_avg_iou"]) if xyz_zero else float("nan")
            rows.append(
                {
                    "variant": variant,
                    "group_mode": group_mode,
                    "clean_class_iou": clean_iou,
                    "delta_clean_vs_fps": clean_iou - fps_clean_iou,
                    "random_keep20_class_iou": random_iou,
                    "structured_keep20_class_iou": structured_iou,
                    "part_drop_largest_class_iou": part_drop_iou,
                    "part_keep20_per_part_class_iou": part_keep_iou,
                    "xyz_zero_class_iou": xyz_zero_iou,
                    "random20_damage": clean_iou - random_iou,
                    "structured20_damage": clean_iou - structured_iou,
                    "part_drop_damage": clean_iou - part_drop_iou,
                }
            )
    return rows


def write_csv(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "group_mode",
        "clean_class_iou",
        "delta_clean_vs_fps",
        "random_keep20_class_iou",
        "structured_keep20_class_iou",
        "part_drop_largest_class_iou",
        "part_keep20_per_part_class_iou",
        "xyz_zero_class_iou",
        "random20_damage",
        "structured20_damage",
        "part_drop_damage",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(rows):
    lines = [
        "# ShapeNetPart Grouping Ablation",
        "",
        "Inference-time grouping/patchization ablation on ShapeNetPart.",
        "The checkpoint and segmentation head are fixed; only grouping center/neighborhood construction is changed.",
        "",
        "| variant | group mode | clean cls-IoU | Δ clean vs FPS | random keep20 | structured keep20 | part-drop largest | xyz-zero |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | `{row['group_mode']}` | "
            f"`{row['clean_class_iou']:.4f}` | `{row['delta_clean_vs_fps']:+.4f}` | "
            f"`{row['random_keep20_class_iou']:.4f}` | "
            f"`{row['structured_keep20_class_iou']:.4f}` | "
            f"`{row['part_drop_largest_class_iou']:.4f}` | "
            f"`{row['xyz_zero_class_iou']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a dense-task companion to the ScanObjectNN grouping diagnostic.",
            "- `random_group` sharply degrades dense part segmentation while center-selection variants stay close to FPS. The safe architecture statement is that local patch neighborhoods are essential, but FPS center selection alone is not the source of random-drop robustness.",
            "- This is still an inference-time diagnostic; retrained grouping variants are required for a stronger architecture-causality claim.",
            "",
            f"- CSV: `{OUT_CSV.relative_to(ROOT)}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit("No ShapeNetPart grouping ablation JSON files found")
    write_csv(rows)
    write_md(rows)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
