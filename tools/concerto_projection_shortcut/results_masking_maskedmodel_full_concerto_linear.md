# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_linear_origin_maskedmodel_full`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-lin-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_lora_origin/exp/scannet-lin-origin-e100/model/model_best.pth`
- Random keep ratios: `0.2`
- Class-wise keep ratios: `0.2`
- Structured keep ratios: `0.2`
- Masked-model keep ratios: `0.2`
- Feature-zero ratios: `1.0`
- Color feature space: `current_0_1`
- Repeats: `1`
- Full-scene scoring: `True`

## Results

| score | variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_nn` | `clean_voxel` | 1.0000 | 0.7695 | +0.0000 | 0.9198 | 0.6636 | 0.4197 | +0.0000 | 0.8737 | 0.9578 | 0.4077 |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.7516 | -0.0179 | 0.9135 | 0.6340 | 0.3467 | -0.0730 | 0.8632 | 0.9437 | 0.5528 |
| `full_nn` | `classwise_keep0p2` | 0.2000 | 0.7512 | -0.0183 | 0.9133 | 0.6337 | 0.3443 | -0.0754 | 0.8620 | 0.9438 | 0.5647 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.4684 | -0.3012 | 0.7618 | 0.3630 | 0.0621 | -0.3576 | 0.6960 | 0.8420 | 0.7129 |
| `full_nn` | `structured_b64_keep0p2` | 0.2051 | 0.2925 | -0.4770 | 0.6103 | 0.2150 | 0.1108 | -0.3089 | 0.5020 | 0.6061 | 0.5912 |
| `full_nn` | `masked_model_keep0p2` | 0.2029 | 0.1990 | -0.5706 | 0.4632 | 0.1521 | 0.0821 | -0.3376 | 0.4035 | 0.3873 | 0.7047 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0389 | -0.7306 | 0.1895 | 0.0099 | 0.0000 | -0.4197 | 0.0160 | 0.4490 | 0.0125 |
| `retained` | `clean_voxel` | 1.0000 | 0.7695 | +0.0000 | 0.9198 | 0.6636 | 0.4197 | +0.0000 | 0.8737 | 0.9578 | 0.4077 |
| `retained` | `masked_model_keep0p2` | 0.2029 | 0.7834 | +0.0138 | 0.9277 | 0.6667 | 0.3773 | -0.0424 | 0.8879 | 0.9738 | 0.4735 |
| `retained` | `random_keep0p2` | 0.2000 | 0.7597 | -0.0098 | 0.9179 | 0.6433 | 0.3556 | -0.0641 | 0.8706 | 0.9537 | 0.5446 |
| `retained` | `classwise_keep0p2` | 0.2000 | 0.7589 | -0.0106 | 0.9176 | 0.6422 | 0.3520 | -0.0677 | 0.8693 | 0.9539 | 0.5597 |
| `retained` | `structured_b64_keep0p2` | 0.2051 | 0.7413 | -0.0283 | 0.9042 | 0.6746 | 0.4828 | +0.0631 | 0.8572 | 0.9545 | 0.4161 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.5689 | -0.2006 | 0.8231 | 0.4674 | 0.1222 | -0.2975 | 0.7677 | 0.9115 | 0.6879 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0389 | -0.7306 | 0.1895 | 0.0099 | 0.0000 | -0.4197 | 0.0160 | 0.4490 | 0.0125 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/concerto_linear/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/concerto_linear/masking_battery_perclass.csv`
