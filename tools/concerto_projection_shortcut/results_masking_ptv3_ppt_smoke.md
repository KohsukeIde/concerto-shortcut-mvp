# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `ptv3_ppt_scannet`
- Config: `configs/scannet/semseg-pt-v3m1-1-ppt-extreme.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-1-ppt-extreme/model/model_best.pth`
- Random keep ratios: `0.2`
- Structured keep ratios: `0.2`
- Feature-zero ratios: `1.0`
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.0482 | +0.0000 | 0.5171 | 0.0007 | 0.0000 | +0.0000 | 0.4179 | 0.4920 | 0.2534 |
| `structured_b64_keep0p2` | 0.2136 | 0.0515 | +0.0033 | 0.5610 | 0.0014 | 0.0000 | +0.0000 | 0.4293 | 0.5393 | nan |
| `random_keep0p2` | 0.2003 | 0.0391 | -0.0091 | 0.4702 | 0.0002 | 0.0000 | +0.0000 | 0.3120 | 0.4547 | 0.1980 |
| `feature_zero1p0` | 1.0000 | 0.0175 | -0.0308 | 0.2809 | 0.0000 | 0.0000 | +0.0000 | 0.2690 | 0.0797 | 0.6029 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_ppt_smoke/masking_battery_summary.csv`
