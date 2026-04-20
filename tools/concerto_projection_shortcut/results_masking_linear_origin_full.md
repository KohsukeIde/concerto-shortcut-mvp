# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_linear_origin`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-lin-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_lora_origin/exp/scannet-lin-origin-e100/model/model_best.pth`
- Random keep ratios: `0.5,0.3,0.2,0.1`
- Structured keep ratios: `0.5,0.2`
- Feature-zero ratios: `1.0`
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.7691 | +0.0000 | 0.9196 | 0.6639 | 0.4230 | +0.0000 | 0.8735 | 0.9577 | 0.3956 |
| `random_keep0p5` | 0.5000 | 0.7708 | +0.0017 | 0.9215 | 0.6612 | 0.3922 | -0.0308 | 0.8748 | 0.9573 | 0.4802 |
| `random_keep0p3` | 0.3001 | 0.7661 | -0.0030 | 0.9201 | 0.6526 | 0.3810 | -0.0420 | 0.8730 | 0.9555 | 0.5118 |
| `random_keep0p2` | 0.2001 | 0.7589 | -0.0103 | 0.9176 | 0.6424 | 0.3546 | -0.0684 | 0.8703 | 0.9534 | 0.5489 |
| `structured_b64_keep0p2` | 0.2065 | 0.7551 | -0.0140 | 0.9116 | 0.6605 | 0.4724 | +0.0494 | 0.8595 | 0.9573 | 0.4605 |
| `structured_b64_keep0p5` | 0.5057 | 0.7542 | -0.0149 | 0.9149 | 0.6455 | 0.3582 | -0.0648 | 0.8687 | 0.9563 | 0.5252 |
| `random_keep0p1` | 0.0999 | 0.7332 | -0.0359 | 0.9086 | 0.6133 | 0.2656 | -0.1574 | 0.8564 | 0.9482 | 0.6008 |
| `feature_zero1p0` | 1.0000 | 0.0390 | -0.7302 | 0.1903 | 0.0101 | 0.0000 | -0.4230 | 0.0160 | 0.4525 | 0.0120 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_lora_origin/masking_linear_full/masking_battery_summary.csv`
