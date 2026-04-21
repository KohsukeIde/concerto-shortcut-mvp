# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_scannet20_maskedmodel_smoke`
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
| `retained` | `clean_voxel` | 1.0000 | 0.4250 | +0.0000 | 0.9218 | 0.0000 | +0.0000 | 0.7849 | nan |
| `retained` | `random_keep0p2` | 0.2000 | 0.3951 | -0.0299 | 0.8956 | 0.0000 | +0.0000 | 0.7698 | nan |
| `retained` | `fixed_points_4000` | 0.0243 | 0.0544 | -0.3706 | 0.4886 | 0.0000 | +0.0000 | 0.3384 | nan |
| `retained` | `classwise_keep0p2` | 0.1978 | 0.3955 | -0.0295 | 0.8948 | 0.0000 | +0.0000 | 0.7697 | nan |
| `retained` | `structured_b64_keep0p2` | 0.2346 | 0.3672 | -0.0578 | 0.8524 | 0.0000 | +0.0000 | 0.5578 | nan |
| `retained` | `masked_model_keep0p2` | 0.1078 | 0.2212 | -0.2038 | 0.7721 | 0.0000 | +0.0000 | 0.4343 | nan |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0220 | -0.4030 | 0.2574 | 0.0000 | +0.0000 | 0.1928 | nan |
| `full_nn` | `clean_voxel` | 1.0000 | 0.4250 | +0.0000 | 0.9218 | 0.0000 | +0.0000 | 0.7849 | nan |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.3895 | -0.0355 | 0.8874 | 0.0000 | +0.0000 | 0.7595 | nan |
| `full_nn` | `fixed_points_4000` | 0.0243 | 0.0520 | -0.3730 | 0.4786 | 0.0000 | +0.0000 | 0.3288 | nan |
| `full_nn` | `classwise_keep0p2` | 0.1978 | 0.3870 | -0.0380 | 0.8882 | 0.0000 | +0.0000 | 0.7596 | nan |
| `full_nn` | `structured_b64_keep0p2` | 0.2346 | 0.1551 | -0.2699 | 0.5364 | 0.0000 | +0.0000 | 0.3285 | nan |
| `full_nn` | `masked_model_keep0p2` | 0.1078 | 0.0557 | -0.3693 | 0.2997 | 0.0000 | +0.0000 | 0.2062 | nan |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0220 | -0.4030 | 0.2574 | 0.0000 | +0.0000 | 0.1928 | nan |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_smoke/ptv3_scannet20/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_smoke/ptv3_scannet20/masking_battery_perclass.csv`
