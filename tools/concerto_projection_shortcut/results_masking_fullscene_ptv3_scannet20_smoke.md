# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_scannet20_fullscene_smoke`
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
| `retained` | `clean_voxel` | 1.0000 | 0.4252 | +0.0000 | 0.9215 | 0.0000 | +0.0000 | 0.7857 | nan |
| `retained` | `random_keep0p2` | 0.1993 | 0.3519 | -0.0733 | 0.8675 | 0.0000 | +0.0000 | 0.7494 | nan |
| `retained` | `classwise_keep0p2` | 0.2008 | 0.3878 | -0.0375 | 0.8826 | 0.0000 | +0.0000 | 0.7567 | nan |
| `full_nn` | `clean_voxel` | 1.0000 | 0.4252 | +0.0000 | 0.9215 | 0.0000 | +0.0000 | 0.7857 | nan |
| `full_nn` | `random_keep0p2` | 0.1993 | 0.3457 | -0.0795 | 0.8609 | 0.0000 | +0.0000 | 0.7391 | nan |
| `full_nn` | `classwise_keep0p2` | 0.2008 | 0.3769 | -0.0483 | 0.8752 | 0.0000 | +0.0000 | 0.7463 | nan |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_fullscene/ptv3_scannet20_smoke/masking_battery_summary.csv`
