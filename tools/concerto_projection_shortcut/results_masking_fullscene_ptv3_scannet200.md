# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_scannet200_v151_fullscene`
- Official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- Config: `configs/scannet200/semseg-pt-v3m1-0-base.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet200-semseg-pt-v3m1-0-base/model/model_best.pth`
- Data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- Segment key: `segment200`
- Focus class: `picture`
- Confusion class: `wall`
- Full-scene scoring: `True`

## Results

| score | variant | keep | mIoU | ΔmIoU | allAcc | focus IoU | Δfocus | confusion IoU | focus->confusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `retained` | `clean_voxel` | 1.0000 | 0.3458 | +0.0000 | 0.8368 | 0.3771 | +0.0000 | 0.8061 | 0.4595 |
| `retained` | `random_keep0p5` | 0.5000 | 0.3323 | -0.0135 | 0.8359 | 0.3585 | -0.0186 | 0.8057 | 0.5177 |
| `retained` | `random_keep0p3` | 0.3001 | 0.3011 | -0.0448 | 0.8255 | 0.3160 | -0.0611 | 0.7912 | 0.5726 |
| `retained` | `random_keep0p2` | 0.2000 | 0.2669 | -0.0789 | 0.8118 | 0.2418 | -0.1353 | 0.7655 | 0.6540 |
| `retained` | `random_keep0p1` | 0.1000 | 0.1594 | -0.1865 | 0.7530 | 0.0814 | -0.2957 | 0.6906 | 0.8531 |
| `retained` | `classwise_keep0p2` | 0.2000 | 0.2646 | -0.0813 | 0.8112 | 0.2453 | -0.1318 | 0.7703 | 0.6493 |
| `retained` | `structured_b64_keep0p5` | 0.5032 | 0.2943 | -0.0515 | 0.8142 | 0.3996 | +0.0226 | 0.7752 | 0.4193 |
| `retained` | `structured_b64_keep0p2` | 0.1977 | 0.2425 | -0.1033 | 0.7870 | 0.3609 | -0.0161 | 0.7422 | 0.5433 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0019 | -0.3440 | 0.2761 | 0.0000 | -0.3771 | 0.2564 | 1.0000 |
| `full_nn` | `clean_voxel` | 1.0000 | 0.3458 | +0.0000 | 0.8368 | 0.3771 | +0.0000 | 0.8061 | 0.4595 |
| `full_nn` | `random_keep0p5` | 0.5000 | 0.3279 | -0.0180 | 0.8324 | 0.3505 | -0.0266 | 0.7995 | 0.5246 |
| `full_nn` | `random_keep0p3` | 0.3001 | 0.2943 | -0.0516 | 0.8197 | 0.3045 | -0.0725 | 0.7808 | 0.5830 |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.2579 | -0.0879 | 0.8039 | 0.2324 | -0.1447 | 0.7522 | 0.6648 |
| `full_nn` | `random_keep0p1` | 0.1000 | 0.1499 | -0.1959 | 0.7401 | 0.0719 | -0.3052 | 0.6715 | 0.8628 |
| `full_nn` | `classwise_keep0p2` | 0.2000 | 0.2561 | -0.0897 | 0.8033 | 0.2366 | -0.1405 | 0.7571 | 0.6560 |
| `full_nn` | `structured_b64_keep0p5` | 0.5032 | 0.1849 | -0.1610 | 0.6877 | 0.2157 | -0.1614 | 0.6043 | 0.5420 |
| `full_nn` | `structured_b64_keep0p2` | 0.1977 | 0.0772 | -0.2686 | 0.4930 | 0.0895 | -0.2876 | 0.3938 | 0.5036 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0019 | -0.3440 | 0.2761 | 0.0000 | -0.3771 | 0.2564 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_fullscene/ptv3_scannet200_full/masking_battery_summary.csv`
