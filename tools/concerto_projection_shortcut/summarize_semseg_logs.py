#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path

VAL_RE = re.compile(
    r"Val result: mIoU/mAcc/allAcc\s+([-+0-9.eE]+)/([-+0-9.eE]+)/([-+0-9.eE]+)"
)
BEST_RE = re.compile(r"Best\s+([A-Za-z0-9_]+):\s+([-+0-9.eE]+)")


def summarize_log(path: Path) -> dict[str, str]:
    text = path.read_text(errors="ignore")
    val_matches = VAL_RE.findall(text)
    best_matches = BEST_RE.findall(text)

    row: dict[str, str] = {
        "log": str(path),
        "val_miou_last": "",
        "val_macc_last": "",
        "val_allacc_last": "",
        "best_metric_name": "",
        "best_metric_value": "",
        "val_eval_count": "0",
    }
    if val_matches:
        miou, macc, allacc = val_matches[-1]
        row["val_miou_last"] = miou
        row["val_macc_last"] = macc
        row["val_allacc_last"] = allacc
        row["val_eval_count"] = str(len(val_matches))
    if best_matches:
        best_name, best_value = best_matches[-1]
        row["best_metric_name"] = best_name
        row["best_metric_value"] = best_value
    return row


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: summarize_semseg_logs.py exp/concerto/*/train.log",
            file=sys.stderr,
        )
        return 2

    rows = [summarize_log(Path(path)) for path in argv[1:]]
    if not rows:
        print("no logs provided", file=sys.stderr)
        return 1

    fieldnames = [
        "log",
        "val_miou_last",
        "val_macc_last",
        "val_allacc_last",
        "best_metric_name",
        "best_metric_value",
        "val_eval_count",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
