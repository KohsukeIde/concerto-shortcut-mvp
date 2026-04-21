# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_linear_origin_fullscene`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-lin-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_lora_origin/exp/scannet-lin-origin-e100/model/model_best.pth`
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
| `full_nn` | `clean_voxel` | 1.0000 | 0.7696 | +0.0000 | 0.9198 | 0.6642 | 0.4194 | +0.0000 | 0.8732 | 0.9577 | 0.4063 |
| `full_nn` | `random_keep0p5` | 0.4999 | 0.7675 | -0.0021 | 0.9194 | 0.6580 | 0.3885 | -0.0309 | 0.8715 | 0.9532 | 0.4817 |
| `full_nn` | `random_keep0p3` | 0.3000 | 0.7605 | -0.0091 | 0.9170 | 0.6447 | 0.3709 | -0.0485 | 0.8676 | 0.9486 | 0.5172 |
| `full_nn` | `random_keep0p2` | 0.2001 | 0.7519 | -0.0177 | 0.9136 | 0.6326 | 0.3442 | -0.0752 | 0.8613 | 0.9439 | 0.5502 |
| `full_nn` | `classwise_keep0p2` | 0.2001 | 0.7511 | -0.0185 | 0.9134 | 0.6326 | 0.3467 | -0.0727 | 0.8616 | 0.9435 | 0.5549 |
| `full_nn` | `random_keep0p1` | 0.0999 | 0.7188 | -0.0508 | 0.9006 | 0.5984 | 0.2565 | -0.1629 | 0.8427 | 0.9310 | 0.5921 |
| `full_nn` | `structured_b64_keep0p5` | 0.4939 | 0.5416 | -0.2280 | 0.7987 | 0.4395 | 0.2442 | -0.1752 | 0.7120 | 0.8193 | 0.6087 |
| `full_nn` | `structured_b64_keep0p2` | 0.1928 | 0.2907 | -0.4789 | 0.6079 | 0.1991 | 0.1040 | -0.3154 | 0.4944 | 0.6122 | 0.6651 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0388 | -0.7308 | 0.1898 | 0.0099 | 0.0000 | -0.4194 | 0.0154 | 0.4514 | 0.0092 |
| `retained` | `clean_voxel` | 1.0000 | 0.7696 | +0.0000 | 0.9198 | 0.6642 | 0.4194 | +0.0000 | 0.8732 | 0.9577 | 0.4063 |
| `retained` | `random_keep0p5` | 0.4999 | 0.7704 | +0.0008 | 0.9210 | 0.6614 | 0.3907 | -0.0287 | 0.8740 | 0.9570 | 0.4805 |
| `retained` | `random_keep0p3` | 0.3000 | 0.7655 | -0.0041 | 0.9198 | 0.6496 | 0.3739 | -0.0455 | 0.8723 | 0.9554 | 0.5164 |
| `retained` | `random_keep0p2` | 0.2001 | 0.7597 | -0.0098 | 0.9178 | 0.6408 | 0.3503 | -0.0691 | 0.8687 | 0.9538 | 0.5458 |
| `retained` | `classwise_keep0p2` | 0.2001 | 0.7588 | -0.0108 | 0.9176 | 0.6418 | 0.3527 | -0.0667 | 0.8691 | 0.9531 | 0.5495 |
| `retained` | `structured_b64_keep0p5` | 0.4939 | 0.7523 | -0.0173 | 0.9122 | 0.6495 | 0.3823 | -0.0371 | 0.8644 | 0.9566 | 0.4833 |
| `retained` | `structured_b64_keep0p2` | 0.1928 | 0.7409 | -0.0287 | 0.9096 | 0.6275 | 0.3851 | -0.0343 | 0.8602 | 0.9535 | 0.4571 |
| `retained` | `random_keep0p1` | 0.0999 | 0.7321 | -0.0375 | 0.9082 | 0.6128 | 0.2710 | -0.1484 | 0.8560 | 0.9479 | 0.5812 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0388 | -0.7308 | 0.1898 | 0.0099 | 0.0000 | -0.4194 | 0.0154 | 0.4514 | 0.0092 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_fullscene/concerto_linear_full/masking_battery_summary.csv`
