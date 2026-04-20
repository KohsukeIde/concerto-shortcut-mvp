# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `sonata_linear_scannet_downloaded_classwise`
- Config: `configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/sonata/sonata_scannet_linear_merged.pth`
- Random keep ratios: `0.2`
- Class-wise keep ratios: `0.2`
- Structured keep ratios: ``
- Feature-zero ratios: ``
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.7169 | +0.0000 | 0.8993 | 0.6137 | 0.3711 | +0.0000 | 0.8581 | 0.9595 | 0.4523 |
| `classwise_keep0p2` | 0.2001 | 0.6951 | -0.0218 | 0.8910 | 0.5771 | 0.2248 | -0.1463 | 0.8486 | 0.9456 | 0.6911 |
| `random_keep0p2` | 0.2000 | 0.6951 | -0.0218 | 0.8911 | 0.5793 | 0.2285 | -0.1426 | 0.8483 | 0.9455 | 0.6900 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/sonata_linear_classwise/masking_battery_summary.csv`
