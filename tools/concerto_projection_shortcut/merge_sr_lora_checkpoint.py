#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge_lora_state_dict(state_dict: dict[str, torch.Tensor], scaling: float) -> tuple[dict[str, torch.Tensor], int]:
    merged: dict[str, torch.Tensor] = {}
    merged_count = 0
    lora_suffixes = (".lora_A.weight", ".lora_B.weight")

    for key, value in state_dict.items():
        if key.endswith(lora_suffixes):
            continue
        if key.endswith(".qkv.weight"):
            a_key = key[:-len(".weight")] + ".lora_A.weight"
            b_key = key[:-len(".weight")] + ".lora_B.weight"
            if a_key in state_dict and b_key in state_dict:
                a = state_dict[a_key].float()
                b = state_dict[b_key].float()
                delta = torch.matmul(b, a) * scaling
                merged[key] = (value.float() + delta).to(dtype=value.dtype)
                merged_count += 1
                continue
        merged[key] = value
    return merged, merged_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge SR-LoRA qkv deltas into a plain Concerto checkpoint.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scaling", type=float, default=2.0)
    parser.add_argument("--metadata-tag", default="sr_lora_merged")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    merged_state, merged_count = merge_lora_state_dict(state_dict, scaling=args.scaling)
    if merged_count == 0:
        raise SystemExit(f"[error] no LoRA qkv pairs found in {args.input}")

    out = dict(checkpoint) if isinstance(checkpoint, dict) else {}
    out["state_dict"] = merged_state
    out.setdefault("metadata", {})
    if isinstance(out["metadata"], dict):
        out["metadata"].update(
            {
                "source_checkpoint": str(args.input),
                "merge_type": args.metadata_tag,
                "merged_qkv_lora_modules": merged_count,
                "lora_scaling": args.scaling,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"[done] merged {merged_count} qkv LoRA modules")
    print(f"[write] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
