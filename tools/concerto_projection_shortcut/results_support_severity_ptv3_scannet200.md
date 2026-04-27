# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_scannet200_v151_severity`
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
| `retained` | `clean_voxel` | 1.0000 | 0.3420 | +0.0000 | 0.8374 | 0.3641 | +0.0000 | 0.8083 | 0.4632 |
| `retained` | `random_keep0p8` | 0.8000 | 0.3446 | +0.0026 | 0.8366 | 0.3667 | +0.0026 | 0.8078 | 0.4795 |
| `retained` | `random_keep0p5` | 0.4998 | 0.3305 | -0.0115 | 0.8348 | 0.3488 | -0.0153 | 0.8067 | 0.5121 |
| `retained` | `random_keep0p2` | 0.2000 | 0.2690 | -0.0730 | 0.8131 | 0.2549 | -0.1092 | 0.7704 | 0.6174 |
| `retained` | `random_keep0p1` | 0.1000 | 0.1648 | -0.1772 | 0.7548 | 0.1239 | -0.2402 | 0.6914 | 0.8111 |
| `retained` | `fixed_points_16000` | 0.1952 | 0.2445 | -0.0975 | 0.7932 | 0.2473 | -0.1168 | 0.7469 | 0.6764 |
| `retained` | `fixed_points_8000` | 0.0976 | 0.1519 | -0.1901 | 0.7123 | 0.1184 | -0.2457 | 0.6374 | 0.8374 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.0618 | -0.2802 | 0.5948 | 0.0837 | -0.2804 | 0.5070 | 0.8523 |
| `retained` | `structured_b64_keep0p8` | 0.8125 | 0.3314 | -0.0106 | 0.8281 | 0.3486 | -0.0155 | 0.7999 | 0.5112 |
| `retained` | `structured_b64_keep0p5` | 0.4929 | 0.3029 | -0.0391 | 0.8087 | 0.2970 | -0.0671 | 0.7678 | 0.5890 |
| `retained` | `structured_b64_keep0p2` | 0.2066 | 0.2641 | -0.0779 | 0.7885 | 0.3890 | +0.0249 | 0.7530 | 0.4685 |
| `retained` | `structured_b64_keep0p1` | 0.1083 | 0.2196 | -0.1224 | 0.7724 | 0.2036 | -0.1605 | 0.7320 | 0.6406 |
| `retained` | `masked_model_keep0p5` | 0.4930 | 0.2753 | -0.0667 | 0.8095 | 0.2120 | -0.1521 | 0.7658 | 0.6172 |
| `retained` | `masked_model_keep0p2` | 0.2042 | 0.2184 | -0.1237 | 0.7903 | 0.1832 | -0.1809 | 0.7641 | 0.6454 |
| `retained` | `masked_model_keep0p1` | 0.0961 | 0.1740 | -0.1680 | 0.7452 | 0.3456 | -0.0185 | 0.6613 | 0.4719 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0019 | -0.3401 | 0.2760 | 0.0000 | -0.3641 | 0.2564 | 1.0000 |
| `full_nn` | `clean_voxel` | 1.0000 | 0.3420 | +0.0000 | 0.8374 | 0.3641 | +0.0000 | 0.8083 | 0.4632 |
| `full_nn` | `random_keep0p8` | 0.8000 | 0.3428 | +0.0008 | 0.8353 | 0.3639 | -0.0002 | 0.8056 | 0.4816 |
| `full_nn` | `random_keep0p5` | 0.4998 | 0.3261 | -0.0159 | 0.8314 | 0.3420 | -0.0221 | 0.8006 | 0.5193 |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.2591 | -0.0829 | 0.8052 | 0.2426 | -0.1215 | 0.7572 | 0.6305 |
| `full_nn` | `random_keep0p1` | 0.1000 | 0.1551 | -0.1869 | 0.7416 | 0.1140 | -0.2501 | 0.6718 | 0.8218 |
| `full_nn` | `fixed_points_16000` | 0.1952 | 0.2101 | -0.1319 | 0.7619 | 0.1481 | -0.2160 | 0.6979 | 0.7920 |
| `full_nn` | `fixed_points_8000` | 0.0976 | 0.1170 | -0.2250 | 0.6599 | 0.0370 | -0.3271 | 0.5703 | 0.9249 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.0416 | -0.3004 | 0.5419 | 0.0191 | -0.3450 | 0.4572 | 0.9228 |
| `full_nn` | `structured_b64_keep0p8` | 0.8125 | 0.2873 | -0.0547 | 0.7912 | 0.3238 | -0.0403 | 0.7440 | 0.5198 |
| `full_nn` | `structured_b64_keep0p5` | 0.4929 | 0.1858 | -0.1562 | 0.6808 | 0.2102 | -0.1539 | 0.5936 | 0.6376 |
| `full_nn` | `structured_b64_keep0p2` | 0.2066 | 0.0858 | -0.2562 | 0.5017 | 0.1179 | -0.2462 | 0.4086 | 0.5136 |
| `full_nn` | `structured_b64_keep0p1` | 0.1083 | 0.0498 | -0.2922 | 0.3909 | 0.0374 | -0.3267 | 0.3107 | 0.5799 |
| `full_nn` | `masked_model_keep0p5` | 0.4930 | 0.1094 | -0.2326 | 0.4875 | 0.1051 | -0.2590 | 0.4070 | 0.6000 |
| `full_nn` | `masked_model_keep0p2` | 0.2042 | 0.0425 | -0.2995 | 0.3159 | 0.0458 | -0.3183 | 0.2890 | 0.5350 |
| `full_nn` | `masked_model_keep0p1` | 0.0961 | 0.0231 | -0.3189 | 0.2519 | 0.0449 | -0.3192 | 0.2449 | 0.5306 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0019 | -0.3401 | 0.2760 | 0.0000 | -0.3641 | 0.2564 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/ptv3_scannet200/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/ptv3_scannet200/masking_battery_perclass.csv`
