# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_linear_origin`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-lin-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_lora_origin/exp/scannet-lin-origin-e100/model/model_best.pth`
- Random keep ratios: `0.2`
- Structured keep ratios: ``
- Feature-zero ratios: ``
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.7362 | +0.0000 | 0.9197 | 0.7021 | 0.6169 | +0.0000 | 0.8634 | 0.9694 | 0.1079 |
| `random_keep0p2` | 0.1999 | 0.7282 | -0.0080 | 0.9179 | 0.6838 | 0.5603 | -0.0566 | 0.8575 | 0.9671 | 0.4221 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_lora_origin/masking_linear_smoke/masking_battery_summary.csv`
