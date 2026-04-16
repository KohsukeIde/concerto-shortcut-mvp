#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


METRIC_KEYS = [
    "loss",
    "enc2d_loss",
    "coord_residual_enc2d_loss",
    "coord_alignment_loss",
    "coord_target_energy",
    "coord_removed_energy",
    "coord_pred_energy",
    "coord_residual_norm",
    "coord_projection_loss_check",
]


def parse_metrics(log_path: Path) -> dict[str, list[float]]:
    values = {key: [] for key in METRIC_KEYS}
    pattern = re.compile(r"([A-Za-z0-9_]+):\s*([-+0-9.eE]+)")
    if not log_path.exists():
        return values
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Train:" not in line and "Train result:" not in line:
            continue
        for key, value in pattern.findall(line):
            if key not in values:
                continue
            try:
                values[key].append(float(value))
            except ValueError:
                pass
    return values


def finite(seq: list[float]) -> bool:
    return bool(seq) and all(math.isfinite(x) for x in seq)


def summarize_row(
    row: dict[str, str],
    exp_prefix: str,
    exp_tag: str,
    exp_root: Path,
    summary_root: Path,
    min_steps: int,
    min_residual_norm: float,
) -> dict:
    arm = row["arm"]
    exp = f"{exp_prefix}-{arm}{exp_tag}-smoke"
    log_path = exp_root / exp / "train.log"
    out_json = summary_root / f"{exp}.json"
    values = parse_metrics(log_path)
    payload = {
        "arm": arm,
        "exp": exp,
        "alpha": float(row["alpha"]),
        "beta": float(row["beta"]),
        "prior_name": row.get("prior", ""),
        "coord_prior_path": row.get("prior_path", ""),
        "log": str(log_path),
        "steps": len(values["loss"]),
        "partial": True,
        "pass": False,
        "reason": "missing_metrics",
        "first": {},
        "last": {},
    }
    for key, seq in values.items():
        if seq:
            payload["first"][key] = seq[0]
            payload["last"][key] = seq[-1]

    required = [
        "loss",
        "enc2d_loss",
        "coord_residual_enc2d_loss",
        "coord_alignment_loss",
        "coord_pred_energy",
        "coord_residual_norm",
    ]
    loss_check = values["coord_projection_loss_check"]
    if all(finite(values[key]) for key in required):
        enc_last = values["enc2d_loss"][-1]
        residual_last = values["coord_residual_norm"][-1]
        pred_last = values["coord_pred_energy"][-1]
        max_loss_check = max(loss_check) if loss_check else float("inf")
        payload["metrics_consistent"] = max_loss_check <= 1e-3
        payload["score"] = (
            enc_last
            + 10.0 * max(0.0, min_residual_norm - residual_last)
            + 25.0 * max(0.0, pred_last - 0.003)
        )
        if payload["steps"] < min_steps:
            payload["reason"] = "too_few_steps"
        elif max_loss_check > 1e-3:
            payload["reason"] = "projection_loss_check_failed"
        elif residual_last < min_residual_norm:
            payload["reason"] = "residual_norm_below_min"
        elif enc_last <= 0.01 or enc_last >= 50:
            payload["reason"] = "enc2d_loss_collapse_or_explode"
        else:
            payload["pass"] = True
            payload["reason"] = "pass"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["summary_json"] = str(out_json)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--exp-prefix", default="arkit-full-projres-v1c")
    parser.add_argument("--exp-tag", required=True)
    parser.add_argument("--min-steps", type=int, default=128)
    parser.add_argument("--min-residual-norm", type=float, default=0.80)
    args = parser.parse_args()

    rows = []
    with args.manifest.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("job_id") == "DRY_RUN":
                continue
            rows.append(
                summarize_row(
                    row,
                    args.exp_prefix,
                    args.exp_tag,
                    args.exp_root,
                    args.summary_root,
                    args.min_steps,
                    args.min_residual_norm,
                )
            )
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
