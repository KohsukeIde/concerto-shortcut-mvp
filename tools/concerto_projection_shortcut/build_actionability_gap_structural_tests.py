#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tools/concerto_projection_shortcut"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def fmt(value, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def build_cross_model_gap_table() -> None:
    src = OUT_DIR / "results_cross_model_downstream_audit_scannet20.csv"
    rows = []
    for row in read_csv(src):
        point_ba = f(row, "point_feature_bal_acc")
        direct_ba = f(row, "direct_pair_margin_bal_acc")
        base_iou = f(row, "base_positive_iou")
        oracle2_iou = f(row, "oracle_top2_positive_iou")
        oracle5_iou = f(row, "oracle_top5_positive_iou")
        rows.append(
            {
                "model": row["model"],
                "pair": row["pair"],
                "positive_class": row["positive_class"],
                "negative_class": row["negative_class"],
                "point_feature_bal_acc": point_ba,
                "direct_pair_margin_bal_acc": direct_ba,
                "direct_minus_point_bal_acc": direct_ba - point_ba,
                "base_positive_iou": base_iou,
                "positive_to_negative": f(row, "base_positive_to_negative"),
                "top2_hit_rate": f(row, "top2_hit_rate"),
                "top5_hit_rate": f(row, "top5_hit_rate"),
                "oracle_top2_positive_iou": oracle2_iou,
                "oracle_top5_positive_iou": oracle5_iou,
                "oracle_top2_headroom": oracle2_iou - base_iou,
                "oracle_top5_headroom": oracle5_iou - base_iou,
            }
        )
    write_csv(OUT_DIR / "results_actionability_gap_cross_model_pairs.csv", rows)

    md = [
        "# Cross-Model Representation-Readout Actionability Gap",
        "",
        "This table separates four quantities that should not be collapsed into a single claim:",
        "",
        "- `point_feature_bal_acc`: whether pairwise semantic information is present in the frozen feature.",
        "- `direct_pair_margin_bal_acc`: whether the released fixed 20-way readout realizes that pairwise information.",
        "- `positive_to_negative`: the observed target-to-confusion error rate.",
        "- `oracle_topK_headroom`: candidate-set/actionability headroom, not a realizable method.",
        "",
        "| model | pair | point BA | direct BA | direct-point | base IoU | pos->neg | top2 | oracle top2 IoU | top2 headroom | top5 headroom |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| `{row['model']}` | `{row['pair']}` | `{fmt(row['point_feature_bal_acc'])}` | "
            f"`{fmt(row['direct_pair_margin_bal_acc'])}` | `{fmt(row['direct_minus_point_bal_acc'])}` | "
            f"`{fmt(row['base_positive_iou'])}` | `{fmt(row['positive_to_negative'])}` | "
            f"`{fmt(row['top2_hit_rate'])}` | `{fmt(row['oracle_top2_positive_iou'])}` | "
            f"`{fmt(row['oracle_top2_headroom'])}` | `{fmt(row['oracle_top5_headroom'])}` |"
        )

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    md.extend(
        [
            "",
            "## Model-Level Means",
            "",
            "| model | mean point BA | mean direct BA | mean pos->neg | mean top2 headroom | picture pos->neg | picture top2 headroom |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    summary_rows = []
    for model, model_rows in by_model.items():
        picture_row = next(r for r in model_rows if r["pair"] == "picture_vs_wall")
        summary = {
            "model": model,
            "mean_point_feature_bal_acc": sum(r["point_feature_bal_acc"] for r in model_rows) / len(model_rows),
            "mean_direct_pair_margin_bal_acc": sum(r["direct_pair_margin_bal_acc"] for r in model_rows) / len(model_rows),
            "mean_positive_to_negative": sum(r["positive_to_negative"] for r in model_rows) / len(model_rows),
            "mean_oracle_top2_headroom": sum(r["oracle_top2_headroom"] for r in model_rows) / len(model_rows),
            "picture_positive_to_negative": picture_row["positive_to_negative"],
            "picture_oracle_top2_headroom": picture_row["oracle_top2_headroom"],
        }
        summary_rows.append(summary)
        md.append(
            f"| `{model}` | `{fmt(summary['mean_point_feature_bal_acc'])}` | "
            f"`{fmt(summary['mean_direct_pair_margin_bal_acc'])}` | "
            f"`{fmt(summary['mean_positive_to_negative'])}` | "
            f"`{fmt(summary['mean_oracle_top2_headroom'])}` | "
            f"`{fmt(summary['picture_positive_to_negative'])}` | "
            f"`{fmt(summary['picture_oracle_top2_headroom'])}` |"
        )
    write_csv(OUT_DIR / "results_actionability_gap_cross_model_model_means.csv", summary_rows)

    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The audited failures are not representation collapse: all rows retain substantial pairwise feature content.",
            "- Concerto/Sonata show large `picture -> wall` confusion and weaker fixed direct margins than the feature/probe evidence would suggest.",
            "- PTv3 and Utonia are cleaner on fixed pairwise readout/confusion, which argues against a universal ScanNet-only artifact.",
            "- Utonia should be described as a constructive comparator: its released stack reduces the audited wall-confusion/fixed-margin gap, but oracle headroom remains and the cause cannot be attributed to a specific Utonia design component without ablation checkpoints.",
        ]
    )
    (OUT_DIR / "results_actionability_gap_cross_model_pairs.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_readout_fix_battery() -> None:
    rows = [
        {
            "family": "fixed-logit residual",
            "method": "confusion residual readout",
            "best_delta_miou": 0.0002,
            "best_delta_picture_iou": 0.00055,
            "picture_to_wall_effect": "reduced slightly",
            "decision": "below gate",
            "source": "results_confusion_residual_readout.md",
        },
        {
            "family": "top-k rerank",
            "method": "top-K pairwise reranker",
            "best_delta_miou": 0.00022,
            "best_delta_picture_iou": 0.00066,
            "picture_to_wall_effect": "reduced at small lambda",
            "decision": "below gate",
            "source": "results_topk_pairwise_rerank_decoder.md",
        },
        {
            "family": "validation-aware set rerank",
            "method": "constrained top-K set decoder",
            "best_delta_miou": 0.00012274,
            "best_delta_picture_iou": 0.00129753,
            "picture_to_wall_effect": "0.4386 -> 0.4277",
            "decision": "below gate",
            "source": "results_constrained_topk_set_decoder.md",
        },
        {
            "family": "feature-conditioned adapter",
            "method": "CoDA residual decoder adapter",
            "best_delta_miou": 0.00023654,
            "best_delta_picture_iou": 0.00167642,
            "picture_to_wall_effect": "0.4436 -> 0.4232",
            "decision": "below gate / transfer failure",
            "source": "results_coda_decoder_adapter.md",
        },
        {
            "family": "in-loop decoder adaptation",
            "method": "CIDA",
            "best_delta_miou": -0.00312027,
            "best_delta_picture_iou": -0.02025931,
            "picture_to_wall_effect": "reduced slightly but collateral damage",
            "decision": "no-go",
            "source": "results_cida_inloop_decoder_adaptation.md",
        },
        {
            "family": "nonparametric retrieval",
            "method": "kNN readout",
            "best_delta_miou": 0.0002,
            "best_delta_picture_iou": 0.0003,
            "picture_to_wall_effect": "minor",
            "decision": "no-go",
            "source": "results_knn_readout_small.md",
        },
        {
            "family": "prototype metric readout",
            "method": "prototype / multi-prototype readout",
            "best_delta_miou": 0.0002,
            "best_delta_picture_iou": 0.0008,
            "picture_to_wall_effect": "minor",
            "decision": "no-go",
            "source": "results_prototype_readout.md",
        },
        {
            "family": "decoupled classifier",
            "method": "tau / cRT / balanced softmax",
            "best_delta_miou": 0.0002,
            "best_delta_picture_iou": -0.0006,
            "picture_to_wall_effect": "can reduce confusion but not IoU",
            "decision": "no-go",
            "source": "results_decoupled_classifier_readout.md",
        },
        {
            "family": "region readout",
            "method": "purity-aware hybrid region decoder",
            "best_delta_miou": 0.0002,
            "best_delta_picture_iou": 0.0,
            "picture_to_wall_effect": "no useful movement",
            "decision": "no-go",
            "source": "results_purity_aware_region_readout.md",
        },
        {
            "family": "proposal/readout",
            "method": "proposal-verify decoder",
            "best_delta_miou": 0.0,
            "best_delta_picture_iou": 0.0,
            "picture_to_wall_effect": "base remains best",
            "decision": "no-go",
            "source": "results_proposal_verify_decoder.md",
        },
        {
            "family": "encoder adaptation",
            "method": "linear-head LoRA",
            "best_delta_miou": 0.0132,
            "best_delta_picture_iou": 0.0225,
            "picture_to_wall_effect": "0.4151 -> 0.3867",
            "decision": "positive only in low-capacity linear-head family",
            "source": "results_scannet_lora_origin_perclass.md",
        },
        {
            "family": "decoder-capacity encoder adaptation",
            "method": "decoder + LoRA",
            "best_delta_miou": -0.0028,
            "best_delta_picture_iou": -0.0013,
            "picture_to_wall_effect": "0.4310 -> 0.4387",
            "decision": "no-go under decoder-capacity matching",
            "source": "results_scannet_dec_lora_origin_perclass.md",
        },
        {
            "family": "full fine-tuning",
            "method": "Concerto origin official-like full FT",
            "best_delta_miou": 0.0187,
            "best_delta_picture_iou": 0.0198,
            "picture_to_wall_effect": "residual 0.3956 in audit row",
            "decision": "improves aggregate but does not close oracle headroom",
            "source": "results_scannet_origin_fullft.md; results_scannet_origin_fullft_oracle_actionability/",
        },
    ]
    write_csv(OUT_DIR / "results_readout_fix_structural_test_battery.csv", rows)
    md = [
        "# Readout/Adaptation Structural Test Battery",
        "",
        "This table is not a method leaderboard. It is a structural test for the claim that the oracle/actionability gap is not a trivial readout artifact.",
        "",
        "| family | method | best delta mIoU | best delta picture | picture->wall effect | decision | source |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        md.append(
            f"| `{row['family']}` | `{row['method']}` | `{fmt(row['best_delta_miou'], 5)}` | "
            f"`{fmt(row['best_delta_picture_iou'], 5)}` | {row['picture_to_wall_effect']} | "
            f"{row['decision']} | `{row['source']}` |"
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Pairwise information and top-k candidate headroom exist, but fixed-logit, cached-feature, validation-aware rerank, nonparametric, decoupled-classifier, region, proposal, and simple adapter families recover almost none of the oracle headroom.",
            "- The linear-head LoRA row shows the target confusion can move when the low-capacity head family is changed, but the gain disappears under decoder-capacity matching.",
            "- Full fine-tuning improves aggregate mIoU and `picture`, but leaves large residual oracle headroom. This supports the phrasing `representation-readout actionability gap`, not `readout problem only`.",
        ]
    )
    (OUT_DIR / "results_readout_fix_structural_test_battery.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    build_cross_model_gap_table()
    build_readout_fix_battery()
    print("[write] tools/concerto_projection_shortcut/results_actionability_gap_cross_model_pairs.md")
    print("[write] tools/concerto_projection_shortcut/results_readout_fix_structural_test_battery.md")


if __name__ == "__main__":
    main()
