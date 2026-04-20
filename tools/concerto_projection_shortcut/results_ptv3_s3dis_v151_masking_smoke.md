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
| `clean_voxel` | 1.0000 | 0.6155 | +0.0000 | 0.9047 | 0.4963 | +0.0000 | 0.8545 | 0.2573 |
| `random_keep0p2` | 0.2002 | 0.4183 | -0.1972 | 0.8264 | 0.0299 | -0.4664 | 0.7698 | 0.9216 |
| `structured_b64_keep0p2` | 0.1799 | 0.5151 | -0.1004 | 0.8380 | 0.4950 | -0.0013 | 0.7782 | 0.0666 |
| `feature_zero1p0` | 1.0000 | 0.1161 | -0.4994 | 0.6411 | 0.0000 | -0.4963 | 0.6718 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_s3dis_v151_smoke/masking_battery_summary.csv`
