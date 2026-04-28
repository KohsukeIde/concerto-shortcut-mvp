#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUTS = [
    (
        "PointGPT-S official",
        ROOT / "3D-NEPA/results/ptgpt_grouping_official_objbg.csv",
    ),
    (
        "PointGPT-S no-mask",
        ROOT / "3D-NEPA/results/ptgpt_grouping_nomask_objbg.csv",
    ),
]
OUT_CSV = ROOT / "tools/concerto_projection_shortcut/results_pointgpt_grouping_ablation.csv"
OUT_MD = ROOT / "tools/concerto_projection_shortcut/results_pointgpt_grouping_ablation.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pivot(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        out.setdefault(row["group_mode"], {})[row["condition"]] = float(row["acc"])
    return out


def main() -> None:
    summary_rows = []
    for variant, path in INPUTS:
        table = pivot(read_rows(path))
        default_clean = table["fps_knn"]["clean"]
        for group_mode, values in sorted(table.items()):
            clean = values.get("clean", float("nan"))
            random20 = values.get("random_keep20", float("nan"))
            structured20 = values.get("structured_keep20", float("nan"))
            random_damage = clean - random20
            structured_damage = clean - structured20
            ratio = structured_damage / random_damage if abs(random_damage) > 1e-12 else float("nan")
            summary_rows.append(
                {
                    "variant": variant,
                    "group_mode": group_mode,
                    "clean_acc": clean,
                    "delta_clean_vs_fps": clean - default_clean,
                    "random_keep20_acc": random20,
                    "structured_keep20_acc": structured20,
                    "random20_damage": random_damage,
                    "structured20_damage": structured_damage,
                    "structured_over_random": ratio,
                    "source": str(path.relative_to(ROOT)),
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# PointGPT Grouping Ablation",
        "",
        "Inference-time grouping/patchization ablation on ScanObjectNN `obj_bg`.",
        "The checkpoint and downstream head are fixed; only the Group module's center/neighborhood construction is changed.",
        "",
        "| variant | group mode | clean acc | Δ clean vs FPS | random keep20 | structured keep20 | structured/random damage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['variant']}` | `{row['group_mode']}` | "
            f"`{row['clean_acc']:.4f}` | `{row['delta_clean_vs_fps']:+.4f}` | "
            f"`{row['random_keep20_acc']:.4f}` | `{row['structured_keep20_acc']:.4f}` | "
            f"`{row['structured_over_random']:.2f}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Changing FPS center selection to random or voxel-distributed centers while keeping kNN neighborhoods has modest effect on clean accuracy.",
        "- Radius neighborhoods with FPS centers are also close to the default row.",
        "- Destroying local neighborhoods with `random_group` collapses clean accuracy to near chance, so the model is not purely context/class-prior driven.",
        "- Structured keep20 remains much harsher than random keep20 for all non-destructive grouping modes.",
        "- This supports a scoped architecture statement: local grouping is essential, but the observed random-drop robustness is not explained by center selection alone. Stronger claims about current point architectures require retrained grouping variants or scene-side grouping ablations.",
        "",
        f"- CSV: `{OUT_CSV.relative_to(ROOT)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print({"rows": len(summary_rows), "csv": str(OUT_CSV)})


if __name__ == "__main__":
    main()
