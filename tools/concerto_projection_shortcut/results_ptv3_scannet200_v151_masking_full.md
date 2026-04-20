# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_supervised_scannet200_v151`
- Official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- Config: `configs/scannet200/semseg-pt-v3m1-0-base.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet200-semseg-pt-v3m1-0-base/model/model_best.pth`
- Data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- Segment key: `segment200`
- Focus class: `picture`
- Confusion class: `wall`

## Results

| variant | keep | mIoU | ΔmIoU | allAcc | focus IoU | Δfocus | confusion IoU | focus->confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.3458 | +0.0000 | 0.8368 | 0.3771 | +0.0000 | 0.8061 | 0.4595 |
| `random_keep0p5` | 0.5000 | 0.3334 | -0.0124 | 0.8353 | 0.3566 | -0.0205 | 0.8055 | 0.5159 |
| `random_keep0p3` | 0.3000 | 0.3085 | -0.0373 | 0.8269 | 0.3018 | -0.0752 | 0.7951 | 0.5785 |
| `random_keep0p2` | 0.2000 | 0.2618 | -0.0840 | 0.8114 | 0.2442 | -0.1328 | 0.7668 | 0.6536 |
| `random_keep0p1` | 0.1000 | 0.1640 | -0.1819 | 0.7535 | 0.1223 | -0.2547 | 0.6889 | 0.8115 |
| `classwise_keep0p2` | 0.2000 | 0.2640 | -0.0818 | 0.8095 | 0.2478 | -0.1293 | 0.7672 | 0.6599 |
| `structured_b64_keep0p5` | 0.5016 | 0.2993 | -0.0465 | 0.8099 | 0.3519 | -0.0252 | 0.7705 | 0.5020 |
| `structured_b64_keep0p2` | 0.2003 | 0.2691 | -0.0767 | 0.7794 | 0.2901 | -0.0870 | 0.7420 | 0.5325 |
| `feature_zero1p0` | 1.0000 | 0.0019 | -0.3440 | 0.2761 | 0.0000 | -0.3771 | 0.2564 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_scannet200_v151_full/masking_battery_summary.csv`
