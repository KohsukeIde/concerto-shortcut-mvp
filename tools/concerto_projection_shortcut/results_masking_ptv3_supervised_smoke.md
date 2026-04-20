# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `ptv3_supervised_scannet`
- Config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- Random keep ratios: `0.2`
- Structured keep ratios: `0.2`
- Feature-zero ratios: `1.0`
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.1042 | +0.0000 | 0.5384 | 0.0488 | 0.0000 | +0.0000 | 0.5213 | 0.7817 | 0.0564 |
| `structured_b64_keep0p2` | 0.1422 | 0.0827 | -0.0214 | 0.4935 | 0.0261 | 0.0000 | +0.0000 | 0.5005 | 0.8377 | nan |
| `random_keep0p2` | 0.1994 | 0.0737 | -0.0305 | 0.5150 | 0.0115 | 0.0000 | +0.0000 | 0.4431 | 0.8138 | 0.7321 |
| `feature_zero1p0` | 1.0000 | 0.0593 | -0.0449 | 0.4296 | 0.0018 | 0.0000 | +0.0000 | 0.4237 | 0.4832 | 0.0218 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_supervised_smoke/masking_battery_summary.csv`
