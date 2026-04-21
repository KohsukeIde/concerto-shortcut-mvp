#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_specs(text: str) -> list[dict]:
    specs = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split("::")
        if len(parts) != 3:
            raise ValueError(f"invalid spec: {chunk}")
        specs.append({"label": parts[0], "stage_csv": Path(parts[1]), "oracle_dir": Path(parts[2])})
    if not specs:
        raise ValueError("no model specs provided")
    return specs


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        pos, neg = [x.strip() for x in chunk.split(":", 1)]
        pairs.append((pos, neg))
    if not pairs:
        raise ValueError("no class pairs provided")
    return pairs


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lookup_stage_metrics(rows: list[dict], pair_name: str, stage: str, probe: str) -> tuple[str, str]:
    for row in rows:
        if row.get("pair") == pair_name and row.get("stage") == stage and row.get("probe") == probe:
            return row.get("balanced_acc", ""), row.get("auc", "")
    return "", ""


def lookup_confusion_fraction(rows: list[dict], target_name: str, pred_name: str) -> str:
    for row in rows:
        if row.get("target_name") == target_name and row.get("pred_name") == pred_name:
            return row.get("fraction_of_target", "")
    return ""


def lookup_topk(rows: list[dict], class_name: str, kind: str, k: int) -> str:
    for row in rows:
        if row.get("class_name") == class_name and row.get("kind") == kind and int(row.get("k", -1)) == k:
            return row.get("hit_rate", "")
    return ""


def lookup_variant_iou(rows: list[dict], variant: str, class_name: str) -> str:
    key = f"{class_name.replace(' ', '_')}_iou"
    for row in rows:
        if row.get("variant") == variant:
            return row.get(key, "")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize cross-model downstream audit tables from stagewise + oracle result files."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument(
        "--model-specs",
        required=True,
        help="Semicolon-separated label::stage_csv::oracle_dir specs.",
    )
    parser.add_argument(
        "--pairs",
        default="picture:wall,door:wall,counter:cabinet",
        help="Comma-separated positive:negative class-name pairs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_cross_model_downstream_audit"),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    specs = parse_specs(args.model_specs)
    pairs = parse_pairs(args.pairs)
    output_prefix = (repo_root / args.output_prefix).resolve() if not args.output_prefix.is_absolute() else args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for spec in specs:
        stage_csv = (repo_root / spec["stage_csv"]).resolve() if not spec["stage_csv"].is_absolute() else spec["stage_csv"]
        oracle_dir = (repo_root / spec["oracle_dir"]).resolve() if not spec["oracle_dir"].is_absolute() else spec["oracle_dir"]
        stage_rows = read_csv(stage_csv)
        variant_rows = read_csv(oracle_dir / "oracle_variants.csv")
        topk_rows = read_csv(oracle_dir / "oracle_topk_hit_rates.csv")
        confusion_rows = read_csv(oracle_dir / "oracle_confusion_distribution.csv")
        for pos, neg in pairs:
            pair = f"{pos}_vs_{neg}".replace(" ", "_")
            point_bal, point_auc = lookup_stage_metrics(stage_rows, pair, "point_feature", "balanced")
            logit_bal, logit_auc = lookup_stage_metrics(stage_rows, pair, "linear_logits", "balanced")
            direct_bal, direct_auc = lookup_stage_metrics(stage_rows, pair, "linear_logits", "direct_pair_margin")
            out_rows.append(
                {
                    "model": spec["label"],
                    "pair": pair,
                    "positive_class": pos,
                    "negative_class": neg,
                    "point_feature_bal_acc": point_bal,
                    "point_feature_auc": point_auc,
                    "linear_logits_bal_acc": logit_bal,
                    "linear_logits_auc": logit_auc,
                    "direct_pair_margin_bal_acc": direct_bal,
                    "direct_pair_margin_auc": direct_auc,
                    "base_positive_iou": lookup_variant_iou(variant_rows, "base", pos),
                    "base_positive_to_negative": lookup_confusion_fraction(confusion_rows, pos, neg),
                    "top2_hit_rate": lookup_topk(topk_rows, pos, "topk", 2),
                    "top5_hit_rate": lookup_topk(topk_rows, pos, "topk", 5),
                    "oracle_top2_positive_iou": lookup_variant_iou(variant_rows, "oracle_top2", pos),
                    "oracle_top5_positive_iou": lookup_variant_iou(variant_rows, "oracle_top5", pos),
                }
            )

    fields = list(out_rows[0].keys())
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    md_lines = [
        "# Cross-Model Downstream Audit Summary",
        "",
        "| model | pair | point bal acc | point AUC | logit bal acc | logit AUC | direct bal acc | direct AUC | base IoU | base pos->neg | top2 | top5 | oracle top2 IoU | oracle top5 IoU |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in out_rows:
        md_lines.append(
            f"| {row['model']} | {row['pair']} | "
            f"{float(row['point_feature_bal_acc']) if row['point_feature_bal_acc'] else float('nan'):.4f} | "
            f"{float(row['point_feature_auc']) if row['point_feature_auc'] else float('nan'):.4f} | "
            f"{float(row['linear_logits_bal_acc']) if row['linear_logits_bal_acc'] else float('nan'):.4f} | "
            f"{float(row['linear_logits_auc']) if row['linear_logits_auc'] else float('nan'):.4f} | "
            f"{float(row['direct_pair_margin_bal_acc']) if row['direct_pair_margin_bal_acc'] else float('nan'):.4f} | "
            f"{float(row['direct_pair_margin_auc']) if row['direct_pair_margin_auc'] else float('nan'):.4f} | "
            f"{float(row['base_positive_iou']) if row['base_positive_iou'] else float('nan'):.4f} | "
            f"{float(row['base_positive_to_negative']) if row['base_positive_to_negative'] else float('nan'):.4f} | "
            f"{float(row['top2_hit_rate']) if row['top2_hit_rate'] else float('nan'):.4f} | "
            f"{float(row['top5_hit_rate']) if row['top5_hit_rate'] else float('nan'):.4f} | "
            f"{float(row['oracle_top2_positive_iou']) if row['oracle_top2_positive_iou'] else float('nan'):.4f} | "
            f"{float(row['oracle_top5_positive_iou']) if row['oracle_top5_positive_iou'] else float('nan'):.4f} |"
        )
    output_prefix.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[write] {csv_path}")
    print(f"[write] {output_prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
