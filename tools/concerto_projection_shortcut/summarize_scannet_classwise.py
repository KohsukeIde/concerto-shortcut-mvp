#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


NUMBER = r"[0-9]+(?:\.[0-9]+)?"
VAL_RE = re.compile(
    r"Val result: mIoU/mAcc/allAcc "
    rf"(?P<miou>{NUMBER})/(?P<macc>{NUMBER})/(?P<allacc>{NUMBER})"
)
CLASS_RE = re.compile(
    r"Class_(?P<idx>[0-9]+)-(?P<name>.+?) Result: "
    rf"iou/accuracy (?P<iou>{NUMBER})/(?P<acc>{NUMBER})"
)


@dataclass
class EvalBlock:
    index: int
    line_no: int
    miou: float
    macc: float
    allacc: float
    classes: list[dict]


def parse_log(path: Path) -> list[EvalBlock]:
    blocks: list[EvalBlock] = []
    current: EvalBlock | None = None
    for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        val_match = VAL_RE.search(line)
        if val_match:
            current = EvalBlock(
                index=len(blocks),
                line_no=line_no,
                miou=float(val_match.group("miou")),
                macc=float(val_match.group("macc")),
                allacc=float(val_match.group("allacc")),
                classes=[],
            )
            blocks.append(current)
            continue
        class_match = CLASS_RE.search(line)
        if class_match and current is not None:
            current.classes.append(
                {
                    "class_id": int(class_match.group("idx")),
                    "class_name": class_match.group("name"),
                    "iou": float(class_match.group("iou")),
                    "accuracy": float(class_match.group("acc")),
                }
            )
    return [block for block in blocks if block.classes]


def choose_block(blocks: list[EvalBlock], kind: str) -> EvalBlock:
    if not blocks:
        raise ValueError("no evaluation blocks found")
    if kind == "last":
        return blocks[-1]
    if kind == "best":
        return max(blocks, key=lambda block: block.miou)
    raise ValueError(f"unknown kind: {kind}")


def parse_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"expected LABEL=LOG_PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty label in spec {spec!r}")
    return label, Path(path).expanduser()


def fmt_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}"


def write_outputs(
    specs: list[tuple[str, Path]],
    baseline_label: str,
    output_csv: Path,
    output_md: Path,
) -> None:
    parsed: dict[str, dict] = {}
    for label, path in specs:
        blocks = parse_log(path)
        if not blocks:
            raise RuntimeError(f"no class-wise evaluation blocks in {path}")
        parsed[label] = {
            "path": path,
            "blocks": blocks,
            "last": choose_block(blocks, "last"),
            "best": choose_block(blocks, "best"),
        }

    if baseline_label not in parsed:
        raise RuntimeError(f"baseline label {baseline_label!r} not in specs")

    rows: list[dict] = []
    for kind in ("last", "best"):
        baseline_block: EvalBlock = parsed[baseline_label][kind]
        baseline_by_id = {row["class_id"]: row for row in baseline_block.classes}
        for label, _ in specs:
            block: EvalBlock = parsed[label][kind]
            for class_row in block.classes:
                base = baseline_by_id[class_row["class_id"]]
                rows.append(
                    {
                        "eval_kind": kind,
                        "experiment": label,
                        "eval_index": block.index,
                        "line_no": block.line_no,
                        "mIoU": f"{block.miou:.4f}",
                        "mAcc": f"{block.macc:.4f}",
                        "allAcc": f"{block.allacc:.4f}",
                        "class_id": class_row["class_id"],
                        "class_name": class_row["class_name"],
                        "iou": f"{class_row['iou']:.4f}",
                        "accuracy": f"{class_row['accuracy']:.4f}",
                        "delta_iou_vs_baseline": fmt_delta(
                            class_row["iou"] - base["iou"]
                            if label != baseline_label
                            else 0.0
                        ),
                        "delta_acc_vs_baseline": fmt_delta(
                            class_row["accuracy"] - base["accuracy"]
                            if label != baseline_label
                            else 0.0
                        ),
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# ScanNet Class-wise Diagnosis",
        "",
        "## Inputs",
        "",
        f"- Baseline: `{baseline_label}`.",
    ]
    for label, path in specs:
        lines.append(f"- `{label}`: `{path}`")

    lines += [
        "",
        "## Overall",
        "",
        "| experiment | last mIoU | best mIoU | last mAcc | last allAcc | eval count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _ in specs:
        last: EvalBlock = parsed[label]["last"]
        best: EvalBlock = parsed[label]["best"]
        lines.append(
            f"| {label} | {last.miou:.4f} | {best.miou:.4f} | "
            f"{last.macc:.4f} | {last.allacc:.4f} | {len(parsed[label]['blocks'])} |"
        )

    baseline_last = parsed[baseline_label]["last"]
    weak = sorted(baseline_last.classes, key=lambda row: row["iou"])[:8]
    lines += [
        "",
        "## Weakest Baseline Classes",
        "",
        "| rank | class | IoU | accuracy |",
        "| ---: | --- | ---: | ---: |",
    ]
    for rank, row in enumerate(weak, start=1):
        lines.append(
            f"| {rank} | {row['class_name']} | {row['iou']:.4f} | {row['accuracy']:.4f} |"
        )

    for label, _ in specs:
        if label == baseline_label:
            continue
        block = parsed[label]["last"]
        base_by_id = {row["class_id"]: row for row in baseline_last.classes}
        deltas = []
        for row in block.classes:
            base = base_by_id[row["class_id"]]
            deltas.append(
                {
                    "class_name": row["class_name"],
                    "iou": row["iou"],
                    "base_iou": base["iou"],
                    "delta": row["iou"] - base["iou"],
                    "accuracy": row["accuracy"],
                    "base_accuracy": base["accuracy"],
                    "delta_acc": row["accuracy"] - base["accuracy"],
                }
            )
        lines += [
            "",
            f"## Delta vs Baseline: {label}",
            "",
            "| class | baseline IoU | experiment IoU | delta IoU | baseline acc | experiment acc | delta acc |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(deltas, key=lambda item: item["delta"]):
            lines.append(
                f"| {row['class_name']} | {row['base_iou']:.4f} | {row['iou']:.4f} | "
                f"{row['delta']:+.4f} | {row['base_accuracy']:.4f} | "
                f"{row['accuracy']:.4f} | {row['delta_acc']:+.4f} |"
            )

    lines += [
        "",
        "## Immediate Readout",
        "",
        "- The lowest baseline class is the first target for failure analysis, not the overall mIoU.",
        "- SR-LoRA changes should be interpreted class-wise; tiny overall deltas can hide targeted gains or losses.",
        "- Confusion-matrix evaluation should focus on the weakest classes and on classes with the largest SR-LoRA deltas.",
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ScanNet class-wise IoU from Pointcept train logs.")
    parser.add_argument("--spec", action="append", required=True, help="LABEL=PATH_TO_TRAIN_LOG")
    parser.add_argument("--baseline", required=True, help="Baseline label from --spec")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    write_outputs(
        specs=[parse_spec(spec) for spec in args.spec],
        baseline_label=args.baseline,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print(f"[write] {args.output_csv}")
    print(f"[write] {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
