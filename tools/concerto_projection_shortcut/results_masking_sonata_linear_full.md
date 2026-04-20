# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `sonata_linear_scannet_downloaded`
- Config: `configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/sonata/sonata_scannet_linear_merged.pth`
- Random keep ratios: `0.5,0.3,0.2,0.1`
- Structured keep ratios: `0.5,0.2`
- Feature-zero ratios: `1.0`
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.7169 | +0.0000 | 0.8992 | 0.6136 | 0.3662 | +0.0000 | 0.8584 | 0.9597 | 0.4588 |
| `random_keep0p5` | 0.5001 | 0.7150 | -0.0019 | 0.8982 | 0.6085 | 0.3339 | -0.0323 | 0.8562 | 0.9572 | 0.5323 |
| `random_keep0p3` | 0.3000 | 0.7038 | -0.0131 | 0.8940 | 0.5910 | 0.2825 | -0.0838 | 0.8512 | 0.9519 | 0.6183 |
| `structured_b64_keep0p5` | 0.5045 | 0.6953 | -0.0216 | 0.8908 | 0.5963 | 0.3657 | -0.0005 | 0.8481 | 0.9565 | 0.4721 |
| `random_keep0p2` | 0.2001 | 0.6942 | -0.0227 | 0.8905 | 0.5769 | 0.2266 | -0.1396 | 0.8462 | 0.9452 | 0.6786 |
| `structured_b64_keep0p2` | 0.2003 | 0.6752 | -0.0417 | 0.8796 | 0.5684 | 0.3461 | -0.0201 | 0.8399 | 0.9551 | 0.4752 |
| `random_keep0p1` | 0.1001 | 0.6650 | -0.0519 | 0.8782 | 0.5425 | 0.1045 | -0.2617 | 0.8281 | 0.9289 | 0.8535 |
| `feature_zero1p0` | 1.0000 | 0.0607 | -0.6562 | 0.4633 | 0.0000 | 0.0000 | -0.3662 | 0.4703 | 0.6285 | 0.7292 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/sonata_linear_full/masking_battery_summary.csv`
