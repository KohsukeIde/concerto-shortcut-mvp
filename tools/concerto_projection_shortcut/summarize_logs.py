#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path

PATTERNS = {
    "loss": re.compile(r"(?:^|\s)loss[:=]\s*([-+0-9.eE]+)"),
    "enc2d_loss": re.compile(r"enc2d_loss[:=]\s*([-+0-9.eE]+)"),
    "mask_loss": re.compile(r"mask_loss[:=]\s*([-+0-9.eE]+)"),
    "unmask_loss": re.compile(r"unmask_loss[:=]\s*([-+0-9.eE]+)"),
    "roll_mask_loss": re.compile(r"roll_mask_loss[:=]\s*([-+0-9.eE]+)"),
}


def metric_matches(text: str, pattern: re.Pattern[str]) -> list[float]:
    return [float(match.group(1)) for match in pattern.finditer(text)]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: summarize_logs.py exp/concerto/*/train.log", file=sys.stderr)
        return 2

    rows = []
    for pattern in argv[1:]:
        for path in sorted(Path().glob(pattern)):
            text = path.read_text(errors="ignore")
            row = {"log": str(path)}
            for key, regex in PATTERNS.items():
                values = metric_matches(text, regex)
                row[f"{key}_first"] = values[0] if values else ""
                row[f"{key}_last"] = values[-1] if values else ""
                row[f"{key}_min"] = min(values) if values else ""
                row[f"{key}_count"] = len(values)
            rows.append(row)

    if not rows:
        print("no logs matched", file=sys.stderr)
        return 1

    fieldnames = ["log"]
    for key in PATTERNS.keys():
        fieldnames.extend(
            [f"{key}_first", f"{key}_last", f"{key}_min", f"{key}_count"]
        )
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
