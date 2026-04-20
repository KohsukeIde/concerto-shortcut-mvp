# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_supervised_s3dis_v151`
- Official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- Config: `configs/s3dis/semseg-pt-v3m1-0-rpe.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth`
- Data root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/concerto_s3dis_imagepoint/s3dis`
- Segment key: `segment`
- Focus class: `board`
- Confusion class: `wall`

## Results

| variant | keep | mIoU | ΔmIoU | allAcc | focus IoU | Δfocus | confusion IoU | focus->confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.7052 | +0.0000 | 0.9036 | 0.8264 | +0.0000 | 0.8419 | 0.0662 |
| `random_keep0p5` | 0.5000 | 0.6476 | -0.0576 | 0.8816 | 0.7417 | -0.0847 | 0.8083 | 0.1171 |
| `random_keep0p3` | 0.3002 | 0.5599 | -0.1452 | 0.8361 | 0.6488 | -0.1777 | 0.7421 | 0.2870 |
| `random_keep0p2` | 0.2000 | 0.4589 | -0.2463 | 0.7816 | 0.3296 | -0.4968 | 0.6818 | 0.5483 |
| `random_keep0p1` | 0.1001 | 0.2220 | -0.4831 | 0.4360 | 0.0008 | -0.8257 | 0.2831 | 0.2208 |
| `classwise_keep0p2` | 0.2000 | 0.4542 | -0.2509 | 0.7801 | 0.3599 | -0.4665 | 0.6813 | 0.5434 |
| `structured_b64_keep0p5` | 0.5074 | 0.6532 | -0.0520 | 0.8788 | 0.7352 | -0.0912 | 0.8078 | 0.1352 |
| `structured_b64_keep0p2` | 0.2011 | 0.6393 | -0.0659 | 0.8720 | 0.7777 | -0.0487 | 0.8039 | 0.1469 |
| `feature_zero1p0` | 1.0000 | 0.1138 | -0.5914 | 0.5401 | 0.0000 | -0.8264 | 0.4824 | 0.9979 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_s3dis_v151_full/masking_battery_summary.csv`
