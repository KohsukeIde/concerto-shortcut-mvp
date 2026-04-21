# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_scannet20_v151_fullscene`
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
| `retained` | `clean_voxel` | 1.0000 | 0.7697 | +0.0000 | 0.9219 | 0.3917 | +0.0000 | 0.8744 | 0.4694 |
| `retained` | `random_keep0p5` | 0.5000 | 0.7669 | -0.0029 | 0.9187 | 0.3710 | -0.0206 | 0.8692 | 0.5450 |
| `retained` | `random_keep0p3` | 0.3000 | 0.7431 | -0.0266 | 0.9109 | 0.3063 | -0.0854 | 0.8602 | 0.6436 |
| `retained` | `random_keep0p2` | 0.2000 | 0.7131 | -0.0566 | 0.8978 | 0.2396 | -0.1520 | 0.8383 | 0.7308 |
| `retained` | `random_keep0p1` | 0.1000 | 0.5905 | -0.1792 | 0.8507 | 0.0376 | -0.3540 | 0.7617 | 0.9257 |
| `retained` | `classwise_keep0p2` | 0.1999 | 0.7108 | -0.0589 | 0.8991 | 0.2147 | -0.1770 | 0.8415 | 0.7576 |
| `retained` | `structured_b64_keep0p5` | 0.4965 | 0.7133 | -0.0565 | 0.9055 | 0.3851 | -0.0066 | 0.8559 | 0.4998 |
| `retained` | `structured_b64_keep0p2` | 0.2070 | 0.6573 | -0.1124 | 0.8779 | 0.4612 | +0.0696 | 0.8086 | 0.4413 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0269 | -0.7429 | 0.3612 | 0.0000 | -0.3917 | 0.3223 | 1.0000 |
| `full_nn` | `clean_voxel` | 1.0000 | 0.7697 | +0.0000 | 0.9219 | 0.3917 | +0.0000 | 0.8744 | 0.4694 |
| `full_nn` | `random_keep0p5` | 0.5000 | 0.7612 | -0.0085 | 0.9159 | 0.3647 | -0.0270 | 0.8644 | 0.5493 |
| `full_nn` | `random_keep0p3` | 0.3000 | 0.7329 | -0.0368 | 0.9061 | 0.2921 | -0.0995 | 0.8519 | 0.6560 |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.6995 | -0.0702 | 0.8911 | 0.2301 | -0.1615 | 0.8273 | 0.7368 |
| `full_nn` | `random_keep0p1` | 0.1000 | 0.5684 | -0.2013 | 0.8384 | 0.0318 | -0.3599 | 0.7452 | 0.9276 |
| `full_nn` | `classwise_keep0p2` | 0.1999 | 0.6977 | -0.0721 | 0.8924 | 0.2049 | -0.1868 | 0.8307 | 0.7661 |
| `full_nn` | `structured_b64_keep0p5` | 0.4965 | 0.4820 | -0.2877 | 0.7802 | 0.2211 | -0.1705 | 0.6913 | 0.6307 |
| `full_nn` | `structured_b64_keep0p2` | 0.2070 | 0.2491 | -0.5206 | 0.5917 | 0.0876 | -0.3041 | 0.4869 | 0.6981 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0269 | -0.7429 | 0.3612 | 0.0000 | -0.3917 | 0.3223 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_fullscene/ptv3_scannet20_full/masking_battery_summary.csv`
