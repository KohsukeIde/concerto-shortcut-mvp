# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `ptv3_ppt_scannet`
- Config: `configs/scannet/semseg-pt-v3m1-1-ppt-extreme.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-1-ppt-extreme/model/model_best.pth`
- Random keep ratios: `0.5,0.3,0.2,0.1`
- Structured keep ratios: `0.5,0.2`
- Feature-zero ratios: `1.0`
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.0422 | +0.0000 | 0.4517 | 0.0008 | 0.0000 | +0.0000 | 0.3744 | 0.4309 | 0.4827 |
| `structured_b64_keep0p5` | 0.5032 | 0.0414 | -0.0008 | 0.4477 | 0.0008 | 0.0000 | -0.0000 | 0.3587 | 0.4350 | 0.4422 |
| `structured_b64_keep0p2` | 0.2038 | 0.0411 | -0.0011 | 0.4433 | 0.0006 | 0.0000 | -0.0000 | 0.3634 | 0.4226 | 0.4779 |
| `random_keep0p1` | 0.1000 | 0.0400 | -0.0022 | 0.4510 | 0.0000 | 0.0000 | -0.0000 | 0.3761 | 0.4216 | 0.5868 |
| `random_keep0p5` | 0.4999 | 0.0364 | -0.0058 | 0.4127 | 0.0007 | 0.0000 | -0.0000 | 0.3078 | 0.3955 | 0.3735 |
| `random_keep0p2` | 0.2000 | 0.0329 | -0.0093 | 0.3901 | 0.0002 | 0.0000 | -0.0000 | 0.2869 | 0.3587 | 0.3717 |
| `random_keep0p3` | 0.3001 | 0.0318 | -0.0104 | 0.3797 | 0.0004 | 0.0000 | -0.0000 | 0.2632 | 0.3556 | 0.3274 |
| `feature_zero1p0` | 1.0000 | 0.0181 | -0.0241 | 0.2928 | 0.0000 | 0.0000 | -0.0000 | 0.2868 | 0.0749 | 0.9140 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_ppt_full/masking_battery_summary.csv`
