#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize raw-aligned fusion vs Pointcept precise/test protocol alignment.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument(
        "--fusion-csv",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_cross_model_fusion_scannet20_with_fullft_ptv3.csv"),
    )
    parser.add_argument(
        "--fullft-log",
        type=Path,
        default=Path("data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/train.log"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("tools/concerto_projection_shortcut/results_fusion_protocol_alignment"),
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_variant(rows: list[dict], variant: str) -> dict:
    for row in rows:
        if row.get("variant") == variant:
            return row
    raise KeyError(f"variant not found: {variant}")


def parse_precise_eval(log_path: Path) -> tuple[float, float, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Val result: mIoU/mAcc/allAcc ([0-9.]+)/([0-9.]+)/([0-9.]+)", text)
    if not matches:
        raise RuntimeError(f"no Pointcept test result found in {log_path}")
    miou, macc, allacc = matches[-1]
    return float(miou), float(macc), float(allacc)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    fusion_csv = resolve(root, args.fusion_csv)
    fullft_log = resolve(root, args.fullft_log)
    output_prefix = resolve(root, args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(fusion_csv)
    fullft_raw = find_variant(rows, "single::Concerto fullFT")
    avg5_raw = find_variant(rows, "avgprob::Concerto decoder+Sonata linear+Utonia+Concerto fullFT+PTv3_supervised")
    oracle5_raw = find_variant(rows, "oracle::Concerto decoder+Sonata linear+Utonia+Concerto fullFT+PTv3_supervised")
    precise_miou, precise_macc, precise_allacc = parse_precise_eval(fullft_log)
    raw_fullft_miou = float(fullft_raw["mIoU"])
    align_rows = [
        {
            "row": "Concerto fullFT raw-aligned single-pass",
            "protocol": "raw-point aligned cache; one pass; no Pointcept test fragments/voting",
            "mIoU": raw_fullft_miou,
            "mAcc": fullft_raw["mAcc"],
            "allAcc": fullft_raw["allAcc"],
            "delta_vs_raw_fullft": 0.0,
        },
        {
            "row": "Concerto fullFT Pointcept precise/test",
            "protocol": "Pointcept tester from training log; model_best test path",
            "mIoU": precise_miou,
            "mAcc": precise_macc,
            "allAcc": precise_allacc,
            "delta_vs_raw_fullft": precise_miou - raw_fullft_miou,
        },
        {
            "row": "5-expert avgprob raw-aligned",
            "protocol": "raw-point aligned cache fusion; diagnostic, not official SOTA protocol",
            "mIoU": float(avg5_raw["mIoU"]),
            "mAcc": avg5_raw["mAcc"],
            "allAcc": avg5_raw["allAcc"],
            "delta_vs_raw_fullft": float(avg5_raw["mIoU"]) - raw_fullft_miou,
        },
        {
            "row": "5-expert oracle raw-aligned",
            "protocol": "diagnostic oracle upper bound using labels",
            "mIoU": float(oracle5_raw["mIoU"]),
            "mAcc": oracle5_raw["mAcc"],
            "allAcc": oracle5_raw["allAcc"],
            "delta_vs_raw_fullft": float(oracle5_raw["mIoU"]) - raw_fullft_miou,
        },
    ]
    write_csv(output_prefix.with_suffix(".csv"), align_rows)
    md = [
        "# Fusion Protocol Alignment",
        "",
        "This table separates the raw-point aligned diagnostic fusion protocol from the Pointcept precise/test path. The fusion rows should not be reported as official SOTA numbers until the fusion output is evaluated under the same final protocol.",
        "",
        "| row | protocol | mIoU | mAcc | allAcc | delta vs raw fullFT |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in align_rows:
        md.append(
            f"| `{row['row']}` | {row['protocol']} | `{float(row['mIoU']):.4f}` | "
            f"`{float(row['mAcc']):.4f}` | `{float(row['allAcc']):.4f}` | `{float(row['delta_vs_raw_fullft']):+.4f}` |"
        )
    md.extend(
        [
            "",
            "Interpretation: the local full-FT reference is `0.8075` under Pointcept's test path, but `0.7969` in the raw-aligned single-pass cache protocol used by current fusion diagnostics. Current fusion gains should therefore be interpreted relative to the raw fullFT row until a final method is exported through a matched official evaluation path.",
        ]
    )
    output_prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[write] {output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
