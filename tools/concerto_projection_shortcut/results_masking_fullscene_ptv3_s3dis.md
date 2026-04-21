# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_s3dis_v151_fullscene`
- Official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- Config: `configs/s3dis/semseg-pt-v3m1-0-rpe.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth`
- Data root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/concerto_s3dis_imagepoint/s3dis`
- Segment key: `segment`
- Focus class: `board`
- Confusion class: `wall`
- Full-scene scoring: `True`

## Results

| score | variant | keep | mIoU | ΔmIoU | allAcc | focus IoU | Δfocus | confusion IoU | focus->confusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `retained` | `clean_voxel` | 1.0000 | 0.7052 | +0.0000 | 0.9036 | 0.8264 | +0.0000 | 0.8419 | 0.0662 |
| `retained` | `random_keep0p5` | 0.5001 | 0.6470 | -0.0582 | 0.8803 | 0.7399 | -0.0865 | 0.8067 | 0.1218 |
| `retained` | `random_keep0p3` | 0.3002 | 0.5549 | -0.1503 | 0.8337 | 0.6334 | -0.1930 | 0.7385 | 0.3101 |
| `retained` | `random_keep0p2` | 0.2000 | 0.4513 | -0.2539 | 0.7797 | 0.3333 | -0.4932 | 0.6812 | 0.5529 |
| `retained` | `random_keep0p1` | 0.1000 | 0.2221 | -0.4831 | 0.4386 | 0.0023 | -0.8241 | 0.2889 | 0.2213 |
| `retained` | `classwise_keep0p2` | 0.1998 | 0.4536 | -0.2516 | 0.7809 | 0.3485 | -0.4780 | 0.6829 | 0.5487 |
| `retained` | `structured_b64_keep0p5` | 0.4839 | 0.6616 | -0.0436 | 0.8873 | 0.7240 | -0.1025 | 0.8222 | 0.1172 |
| `retained` | `structured_b64_keep0p2` | 0.2038 | 0.6374 | -0.0678 | 0.8718 | 0.7733 | -0.0531 | 0.8007 | 0.1587 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.1138 | -0.5914 | 0.5401 | 0.0000 | -0.8264 | 0.4824 | 0.9979 |
| `full_nn` | `clean_voxel` | 1.0000 | 0.7052 | +0.0000 | 0.9036 | 0.8264 | +0.0000 | 0.8419 | 0.0662 |
| `full_nn` | `random_keep0p5` | 0.5001 | 0.6452 | -0.0600 | 0.8790 | 0.7355 | -0.0909 | 0.8044 | 0.1250 |
| `full_nn` | `random_keep0p3` | 0.3002 | 0.5515 | -0.1537 | 0.8311 | 0.6255 | -0.2010 | 0.7359 | 0.3156 |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.4445 | -0.2607 | 0.7731 | 0.3297 | -0.4967 | 0.6767 | 0.5451 |
| `full_nn` | `random_keep0p1` | 0.1000 | 0.2044 | -0.5008 | 0.4024 | 0.0018 | -0.8247 | 0.2554 | 0.1831 |
| `full_nn` | `classwise_keep0p2` | 0.1998 | 0.4468 | -0.2584 | 0.7746 | 0.3434 | -0.4830 | 0.6785 | 0.5429 |
| `full_nn` | `structured_b64_keep0p5` | 0.4839 | 0.4837 | -0.2214 | 0.7770 | 0.4691 | -0.3573 | 0.6704 | 0.2501 |
| `full_nn` | `structured_b64_keep0p2` | 0.2038 | 0.2881 | -0.4170 | 0.5955 | 0.2644 | -0.5621 | 0.4739 | 0.2423 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.1138 | -0.5914 | 0.5401 | 0.0000 | -0.8264 | 0.4824 | 0.9979 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_fullscene/ptv3_s3dis_full/masking_battery_summary.csv`
