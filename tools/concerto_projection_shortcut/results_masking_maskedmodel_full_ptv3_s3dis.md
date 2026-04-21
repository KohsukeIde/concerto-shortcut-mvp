# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_s3dis_maskedmodel_full`
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
| `retained` | `clean_voxel` | 1.0000 | 0.7040 | +0.0000 | 0.9033 | 0.8237 | +0.0000 | 0.8413 | 0.0678 |
| `retained` | `random_keep0p2` | 0.2003 | 0.4497 | -0.2543 | 0.7810 | 0.3112 | -0.5125 | 0.6806 | 0.5820 |
| `retained` | `fixed_points_4000` | 0.0175 | 0.0020 | -0.7021 | 0.0036 | 0.0000 | -0.8237 | 0.0016 | 0.0000 |
| `retained` | `classwise_keep0p2` | 0.2000 | 0.4556 | -0.2484 | 0.7803 | 0.3362 | -0.4875 | 0.6798 | 0.5657 |
| `retained` | `structured_b64_keep0p2` | 0.1886 | 0.5905 | -0.1135 | 0.8550 | 0.4203 | -0.4033 | 0.7836 | 0.3104 |
| `retained` | `masked_model_keep0p2` | 0.1875 | 0.6755 | -0.0285 | 0.8924 | 0.7330 | -0.0906 | 0.8522 | 0.1289 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.1150 | -0.5891 | 0.5423 | 0.0000 | -0.8237 | 0.4825 | 0.9978 |
| `full_nn` | `clean_voxel` | 1.0000 | 0.7040 | +0.0000 | 0.9033 | 0.8237 | +0.0000 | 0.8413 | 0.0678 |
| `full_nn` | `random_keep0p2` | 0.2003 | 0.4434 | -0.2606 | 0.7748 | 0.3057 | -0.5179 | 0.6765 | 0.5748 |
| `full_nn` | `fixed_points_4000` | 0.0175 | 0.0005 | -0.7035 | 0.0010 | 0.0000 | -0.8237 | 0.0003 | 0.0000 |
| `full_nn` | `classwise_keep0p2` | 0.2000 | 0.4489 | -0.2551 | 0.7741 | 0.3329 | -0.4907 | 0.6757 | 0.5572 |
| `full_nn` | `structured_b64_keep0p2` | 0.1886 | 0.2943 | -0.4098 | 0.6077 | 0.2039 | -0.6198 | 0.4872 | 0.3576 |
| `full_nn` | `masked_model_keep0p2` | 0.1875 | 0.1083 | -0.5957 | 0.2707 | 0.1048 | -0.7189 | 0.2339 | 0.2090 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.1150 | -0.5891 | 0.5423 | 0.0000 | -0.8237 | 0.4825 | 0.9978 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/ptv3_s3dis/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/ptv3_s3dis/masking_battery_perclass.csv`
