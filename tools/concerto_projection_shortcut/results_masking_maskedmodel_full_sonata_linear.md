# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `sonata_linear_maskedmodel_full`
- Config: `configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/sonata/sonata_scannet_linear_merged.pth`
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
| `full_nn` | `clean_voxel` | 1.0000 | 0.7167 | +0.0000 | 0.8991 | 0.6133 | 0.3699 | +0.0000 | 0.8576 | 0.9596 | 0.4566 |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.6862 | -0.0305 | 0.8864 | 0.5707 | 0.2225 | -0.1474 | 0.8405 | 0.9359 | 0.6913 |
| `full_nn` | `classwise_keep0p2` | 0.2000 | 0.6851 | -0.0316 | 0.8861 | 0.5693 | 0.2161 | -0.1538 | 0.8394 | 0.9357 | 0.7050 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.3850 | -0.3317 | 0.7235 | 0.2796 | 0.0141 | -0.3557 | 0.6730 | 0.8801 | 0.9108 |
| `full_nn` | `structured_b64_keep0p2` | 0.1981 | 0.2743 | -0.4424 | 0.5947 | 0.1873 | 0.0707 | -0.2992 | 0.4891 | 0.6176 | 0.6895 |
| `full_nn` | `masked_model_keep0p2` | 0.1938 | 0.1724 | -0.5444 | 0.4386 | 0.1223 | 0.0638 | -0.3061 | 0.3898 | 0.3697 | 0.6526 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0606 | -0.6561 | 0.4631 | 0.0000 | 0.0000 | -0.3699 | 0.4708 | 0.6278 | 0.7335 |
| `retained` | `clean_voxel` | 1.0000 | 0.7167 | +0.0000 | 0.8991 | 0.6133 | 0.3699 | +0.0000 | 0.8576 | 0.9596 | 0.4566 |
| `retained` | `random_keep0p2` | 0.2000 | 0.6946 | -0.0222 | 0.8906 | 0.5797 | 0.2303 | -0.1396 | 0.8480 | 0.9455 | 0.6821 |
| `retained` | `classwise_keep0p2` | 0.2000 | 0.6928 | -0.0239 | 0.8902 | 0.5783 | 0.2230 | -0.1469 | 0.8465 | 0.9450 | 0.7003 |
| `retained` | `masked_model_keep0p2` | 0.1938 | 0.6864 | -0.0303 | 0.8857 | 0.5698 | 0.3973 | +0.0274 | 0.8815 | 0.9699 | 0.2850 |
| `retained` | `structured_b64_keep0p2` | 0.1981 | 0.6710 | -0.0457 | 0.8801 | 0.5698 | 0.2615 | -0.1084 | 0.8317 | 0.9578 | 0.5651 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.4750 | -0.2417 | 0.7822 | 0.3837 | 0.0646 | -0.3053 | 0.7343 | 0.9231 | 0.8578 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0606 | -0.6561 | 0.4631 | 0.0000 | 0.0000 | -0.3699 | 0.4708 | 0.6278 | 0.7335 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/sonata_linear/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/sonata_linear/masking_battery_perclass.csv`
