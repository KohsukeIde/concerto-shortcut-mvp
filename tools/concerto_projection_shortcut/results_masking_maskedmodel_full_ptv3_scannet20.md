# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_scannet20_maskedmodel_full`
- Official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- Config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- Data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- Segment key: `segment20`
- Focus class: `picture`
- Confusion class: `wall`
- Full-scene scoring: `True`

## Results

| score | variant | keep | mIoU | ΔmIoU | allAcc | focus IoU | Δfocus | confusion IoU | focus->confusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `retained` | `clean_voxel` | 1.0000 | 0.7699 | +0.0000 | 0.9214 | 0.3881 | +0.0000 | 0.8725 | 0.4940 |
| `retained` | `random_keep0p2` | 0.2001 | 0.7093 | -0.0607 | 0.8978 | 0.2149 | -0.1733 | 0.8390 | 0.7562 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.3071 | -0.4629 | 0.6759 | 0.0394 | -0.3487 | 0.5594 | 0.9159 |
| `retained` | `classwise_keep0p2` | 0.2000 | 0.7127 | -0.0572 | 0.8981 | 0.2257 | -0.1624 | 0.8401 | 0.7459 |
| `retained` | `structured_b64_keep0p2` | 0.2037 | 0.6483 | -0.1217 | 0.8758 | 0.3453 | -0.0428 | 0.8209 | 0.5731 |
| `retained` | `masked_model_keep0p2` | 0.2028 | 0.6389 | -0.1310 | 0.8868 | 0.2676 | -0.1205 | 0.8320 | 0.6778 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0271 | -0.7429 | 0.3624 | 0.0000 | -0.3881 | 0.3228 | 1.0000 |
| `full_nn` | `clean_voxel` | 1.0000 | 0.7699 | +0.0000 | 0.9214 | 0.3881 | +0.0000 | 0.8725 | 0.4940 |
| `full_nn` | `random_keep0p2` | 0.2001 | 0.6969 | -0.0731 | 0.8913 | 0.2062 | -0.1819 | 0.8282 | 0.7638 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.2065 | -0.5635 | 0.6007 | 0.0068 | -0.3813 | 0.4901 | 0.9431 |
| `full_nn` | `classwise_keep0p2` | 0.2000 | 0.6994 | -0.0705 | 0.8915 | 0.2180 | -0.1701 | 0.8293 | 0.7525 |
| `full_nn` | `structured_b64_keep0p2` | 0.2037 | 0.2527 | -0.5172 | 0.5889 | 0.1238 | -0.2644 | 0.4892 | 0.6745 |
| `full_nn` | `masked_model_keep0p2` | 0.2028 | 0.1324 | -0.6375 | 0.4157 | 0.0524 | -0.3357 | 0.3616 | 0.6993 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0271 | -0.7429 | 0.3624 | 0.0000 | -0.3881 | 0.3228 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/ptv3_scannet20/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/ptv3_scannet20/masking_battery_perclass.csv`
