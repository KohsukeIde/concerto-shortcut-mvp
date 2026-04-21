# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `sonata_linear_scannet_fullscene`
- Config: `configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/sonata/sonata_scannet_linear_merged.pth`
- Random keep ratios: `0.5,0.3,0.2,0.1`
- Class-wise keep ratios: `0.2`
- Structured keep ratios: `0.5,0.2`
- Feature-zero ratios: `1.0`
- Color feature space: `current_0_1`
- Repeats: `1`
- Full-scene scoring: `True`

## Results

| score | variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_nn` | `clean_voxel` | 1.0000 | 0.7170 | +0.0000 | 0.8991 | 0.6128 | 0.3627 | +0.0000 | 0.8577 | 0.9595 | 0.4642 |
| `full_nn` | `random_keep0p5` | 0.4999 | 0.7120 | -0.0049 | 0.8967 | 0.6047 | 0.3290 | -0.0337 | 0.8531 | 0.9527 | 0.5429 |
| `full_nn` | `random_keep0p3` | 0.2999 | 0.6970 | -0.0200 | 0.8906 | 0.5837 | 0.2796 | -0.0831 | 0.8462 | 0.9449 | 0.6244 |
| `full_nn` | `classwise_keep0p2` | 0.2000 | 0.6874 | -0.0296 | 0.8871 | 0.5696 | 0.2178 | -0.1449 | 0.8413 | 0.9362 | 0.7010 |
| `full_nn` | `random_keep0p2` | 0.2001 | 0.6865 | -0.0305 | 0.8865 | 0.5691 | 0.2239 | -0.1387 | 0.8409 | 0.9360 | 0.6972 |
| `full_nn` | `random_keep0p1` | 0.1000 | 0.6507 | -0.0663 | 0.8714 | 0.5212 | 0.0867 | -0.2759 | 0.8178 | 0.9155 | 0.8498 |
| `full_nn` | `structured_b64_keep0p5` | 0.4960 | 0.4938 | -0.2231 | 0.7755 | 0.3730 | 0.1788 | -0.1839 | 0.6962 | 0.8224 | 0.7019 |
| `full_nn` | `structured_b64_keep0p2` | 0.1957 | 0.2662 | -0.4508 | 0.5902 | 0.1850 | 0.0975 | -0.2652 | 0.4942 | 0.6124 | 0.6885 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0607 | -0.6563 | 0.4627 | 0.0000 | 0.0000 | -0.3627 | 0.4709 | 0.6272 | 0.7462 |
| `retained` | `clean_voxel` | 1.0000 | 0.7170 | +0.0000 | 0.8991 | 0.6128 | 0.3627 | +0.0000 | 0.8577 | 0.9595 | 0.4642 |
| `retained` | `random_keep0p5` | 0.4999 | 0.7154 | -0.0015 | 0.8986 | 0.6080 | 0.3332 | -0.0295 | 0.8561 | 0.9572 | 0.5389 |
| `retained` | `random_keep0p3` | 0.2999 | 0.7026 | -0.0144 | 0.8937 | 0.5898 | 0.2849 | -0.0778 | 0.8514 | 0.9520 | 0.6190 |
| `retained` | `structured_b64_keep0p5` | 0.4960 | 0.6966 | -0.0204 | 0.8873 | 0.5872 | 0.3122 | -0.0505 | 0.8432 | 0.9595 | 0.5781 |
| `retained` | `classwise_keep0p2` | 0.2000 | 0.6955 | -0.0215 | 0.8912 | 0.5782 | 0.2271 | -0.1356 | 0.8480 | 0.9455 | 0.6898 |
| `retained` | `random_keep0p2` | 0.2001 | 0.6951 | -0.0219 | 0.8909 | 0.5792 | 0.2365 | -0.1262 | 0.8482 | 0.9455 | 0.6858 |
| `retained` | `structured_b64_keep0p2` | 0.1957 | 0.6712 | -0.0457 | 0.8773 | 0.5933 | 0.3197 | -0.0429 | 0.8401 | 0.9545 | 0.5737 |
| `retained` | `random_keep0p1` | 0.1000 | 0.6632 | -0.0538 | 0.8781 | 0.5363 | 0.0945 | -0.2682 | 0.8298 | 0.9287 | 0.8417 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0607 | -0.6563 | 0.4627 | 0.0000 | 0.0000 | -0.3627 | 0.4709 | 0.6272 | 0.7462 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_fullscene/sonata_linear_full/masking_battery_summary.csv`
