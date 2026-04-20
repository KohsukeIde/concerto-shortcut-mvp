# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `sonata_linear_scannet_downloaded`
- Config: `configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/sonata/sonata_scannet_linear_merged.pth`
- Random keep ratios: `0.2`
- Structured keep ratios: `0.2`
- Feature-zero ratios: `1.0`
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.5155 | +0.0000 | 0.9134 | 0.4630 | 0.0000 | +0.0000 | 0.8567 | 0.9755 | 0.5221 |
| `random_keep0p2` | 0.2001 | 0.5054 | -0.0101 | 0.9132 | 0.4681 | 0.0000 | +0.0000 | 0.8563 | 0.9670 | 0.6138 |
| `structured_b64_keep0p2` | 0.1853 | 0.3654 | -0.1500 | 0.9093 | 0.1630 | 0.0000 | +0.0000 | 0.8727 | 0.9806 | nan |
| `feature_zero1p0` | 1.0000 | 0.0636 | -0.4518 | 0.4707 | 0.0000 | 0.0000 | +0.0000 | 0.4308 | 0.6463 | 0.9776 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/sonata_linear_smoke/masking_battery_summary.csv`
