#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_root", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.summary_root.glob("arkit-full-projres-v1b-*-smoke.json")):
        payload = json.loads(path.read_text())
        payload["summary_json"] = str(path)
        rows.append(payload)

    passed = [row for row in rows if row.get("pass")]
    passed.sort(
        key=lambda row: (
            row.get("score", float("inf")),
            row.get("last", {}).get("enc2d_loss", float("inf")),
        )
    )
    selected = passed[: args.top_k]
    result = {
        "pass": bool(selected),
        "reason": "pass" if selected else "no_smoke_passed",
        "selected": selected[0] if selected else None,
        "top": selected,
        "candidates": rows,
    }
    out = args.out or args.summary_root / "selected_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
