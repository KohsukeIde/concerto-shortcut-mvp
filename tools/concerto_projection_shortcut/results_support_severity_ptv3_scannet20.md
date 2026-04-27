# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_scannet20_v151_severity`
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
| `retained` | `clean_voxel` | 1.0000 | 0.7713 | +0.0000 | 0.9222 | 0.3862 | +0.0000 | 0.8757 | 0.4918 |
| `retained` | `random_keep0p8` | 0.8001 | 0.7698 | -0.0015 | 0.9220 | 0.3930 | +0.0067 | 0.8784 | 0.4842 |
| `retained` | `random_keep0p5` | 0.5000 | 0.7630 | -0.0083 | 0.9188 | 0.3760 | -0.0102 | 0.8733 | 0.5440 |
| `retained` | `random_keep0p2` | 0.1998 | 0.7091 | -0.0622 | 0.8976 | 0.2286 | -0.1576 | 0.8367 | 0.7392 |
| `retained` | `random_keep0p1` | 0.1001 | 0.5795 | -0.1918 | 0.8498 | 0.0237 | -0.3625 | 0.7656 | 0.9495 |
| `retained` | `fixed_points_16000` | 0.1952 | 0.6795 | -0.0917 | 0.8848 | 0.1855 | -0.2007 | 0.8289 | 0.7998 |
| `retained` | `fixed_points_8000` | 0.0976 | 0.5313 | -0.2400 | 0.8060 | 0.0801 | -0.3061 | 0.7074 | 0.8899 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.3012 | -0.4701 | 0.6749 | 0.0268 | -0.3594 | 0.5569 | 0.9526 |
| `retained` | `structured_b64_keep0p8` | 0.8045 | 0.7465 | -0.0248 | 0.9128 | 0.3826 | -0.0036 | 0.8619 | 0.5067 |
| `retained` | `structured_b64_keep0p5` | 0.5078 | 0.7143 | -0.0569 | 0.9006 | 0.3829 | -0.0034 | 0.8421 | 0.5187 |
| `retained` | `structured_b64_keep0p2` | 0.1919 | 0.6714 | -0.0999 | 0.8792 | 0.3565 | -0.0297 | 0.8236 | 0.5284 |
| `retained` | `structured_b64_keep0p1` | 0.1049 | 0.6200 | -0.1513 | 0.8623 | 0.3002 | -0.0860 | 0.7930 | 0.4777 |
| `retained` | `masked_model_keep0p5` | 0.4927 | 0.7359 | -0.0354 | 0.9147 | 0.3760 | -0.0102 | 0.8617 | 0.5209 |
| `retained` | `masked_model_keep0p2` | 0.1970 | 0.6469 | -0.1244 | 0.8731 | 0.1596 | -0.2267 | 0.7964 | 0.7533 |
| `retained` | `masked_model_keep0p1` | 0.1052 | 0.6435 | -0.1278 | 0.8918 | 0.3615 | -0.0247 | 0.8219 | 0.4749 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0271 | -0.7442 | 0.3622 | 0.0000 | -0.3862 | 0.3228 | 1.0000 |
| `full_nn` | `clean_voxel` | 1.0000 | 0.7713 | +0.0000 | 0.9222 | 0.3862 | +0.0000 | 0.8757 | 0.4918 |
| `full_nn` | `random_keep0p8` | 0.8001 | 0.7676 | -0.0037 | 0.9210 | 0.3902 | +0.0040 | 0.8766 | 0.4859 |
| `full_nn` | `random_keep0p5` | 0.5000 | 0.7572 | -0.0141 | 0.9160 | 0.3715 | -0.0148 | 0.8684 | 0.5473 |
| `full_nn` | `random_keep0p2` | 0.1998 | 0.6963 | -0.0750 | 0.8912 | 0.2226 | -0.1636 | 0.8264 | 0.7434 |
| `full_nn` | `random_keep0p1` | 0.1001 | 0.5595 | -0.2118 | 0.8380 | 0.0223 | -0.3640 | 0.7491 | 0.9493 |
| `full_nn` | `fixed_points_16000` | 0.1952 | 0.6031 | -0.1682 | 0.8495 | 0.0878 | -0.2985 | 0.7692 | 0.8962 |
| `full_nn` | `fixed_points_8000` | 0.0976 | 0.4121 | -0.3592 | 0.7378 | 0.0204 | -0.3659 | 0.6172 | 0.9481 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.2035 | -0.5678 | 0.6006 | 0.0055 | -0.3807 | 0.4890 | 0.9659 |
| `full_nn` | `structured_b64_keep0p8` | 0.8045 | 0.6615 | -0.1098 | 0.8743 | 0.3226 | -0.0637 | 0.8100 | 0.5529 |
| `full_nn` | `structured_b64_keep0p5` | 0.5078 | 0.4928 | -0.2785 | 0.7803 | 0.2424 | -0.1438 | 0.6885 | 0.6279 |
| `full_nn` | `structured_b64_keep0p2` | 0.1919 | 0.2501 | -0.5212 | 0.5722 | 0.0903 | -0.2959 | 0.4723 | 0.6688 |
| `full_nn` | `structured_b64_keep0p1` | 0.1049 | 0.1564 | -0.6149 | 0.4703 | 0.0418 | -0.3444 | 0.3767 | 0.5734 |
| `full_nn` | `masked_model_keep0p5` | 0.4927 | 0.3138 | -0.4575 | 0.5896 | 0.1485 | -0.2377 | 0.5007 | 0.6778 |
| `full_nn` | `masked_model_keep0p2` | 0.1970 | 0.1388 | -0.6325 | 0.4112 | 0.0491 | -0.3371 | 0.3546 | 0.7255 |
| `full_nn` | `masked_model_keep0p1` | 0.1052 | 0.0894 | -0.6818 | 0.3553 | 0.0324 | -0.3538 | 0.3160 | 0.6850 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0271 | -0.7442 | 0.3622 | 0.0000 | -0.3862 | 0.3228 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/ptv3_scannet20/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/ptv3_scannet20/masking_battery_perclass.csv`
