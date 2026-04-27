# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_s3dis_v151_severity`
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
| `retained` | `clean_voxel` | 1.0000 | 0.7112 | +0.0000 | 0.9042 | 0.8451 | +0.0000 | 0.8441 | 0.0555 |
| `retained` | `random_keep0p8` | 0.7998 | 0.7020 | -0.0093 | 0.9031 | 0.8209 | -0.0243 | 0.8412 | 0.0562 |
| `retained` | `random_keep0p5` | 0.5000 | 0.6482 | -0.0630 | 0.8813 | 0.7474 | -0.0978 | 0.8075 | 0.1125 |
| `retained` | `random_keep0p2` | 0.1999 | 0.4517 | -0.2595 | 0.7797 | 0.3143 | -0.5309 | 0.6803 | 0.5815 |
| `retained` | `random_keep0p1` | 0.1000 | 0.2236 | -0.4877 | 0.4375 | 0.0021 | -0.8430 | 0.2866 | 0.2381 |
| `retained` | `fixed_points_16000` | 0.0701 | 0.1028 | -0.6084 | 0.1859 | 0.0000 | -0.8451 | 0.1113 | 0.0530 |
| `retained` | `fixed_points_8000` | 0.0350 | 0.0169 | -0.6944 | 0.0268 | 0.0000 | -0.8451 | 0.0194 | 0.0013 |
| `retained` | `fixed_points_4000` | 0.0175 | 0.0021 | -0.7092 | 0.0041 | 0.0000 | -0.8451 | 0.0028 | 0.0003 |
| `retained` | `structured_b64_keep0p8` | 0.7958 | 0.6998 | -0.0114 | 0.9001 | 0.8342 | -0.0109 | 0.8350 | 0.0534 |
| `retained` | `structured_b64_keep0p5` | 0.5024 | 0.6556 | -0.0557 | 0.8882 | 0.7550 | -0.0902 | 0.8297 | 0.1057 |
| `retained` | `structured_b64_keep0p2` | 0.1832 | 0.5875 | -0.1238 | 0.8765 | 0.6211 | -0.2241 | 0.8122 | 0.2233 |
| `retained` | `structured_b64_keep0p1` | 0.1032 | 0.6002 | -0.1111 | 0.8650 | 0.5684 | -0.2768 | 0.8040 | 0.1560 |
| `retained` | `masked_model_keep0p5` | 0.4894 | 0.6597 | -0.0515 | 0.8905 | 0.8036 | -0.0415 | 0.8149 | 0.0893 |
| `retained` | `masked_model_keep0p2` | 0.2159 | 0.6896 | -0.0216 | 0.9201 | 0.7998 | -0.0453 | 0.8527 | 0.0582 |
| `retained` | `masked_model_keep0p1` | 0.1113 | 0.6268 | -0.0844 | 0.8573 | 0.9341 | +0.0889 | 0.8119 | 0.0276 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.1147 | -0.5966 | 0.5417 | 0.0000 | -0.8451 | 0.4825 | 0.9978 |
| `full_nn` | `clean_voxel` | 1.0000 | 0.7112 | +0.0000 | 0.9042 | 0.8451 | +0.0000 | 0.8441 | 0.0555 |
| `full_nn` | `random_keep0p8` | 0.7998 | 0.7013 | -0.0099 | 0.9027 | 0.8192 | -0.0259 | 0.8404 | 0.0570 |
| `full_nn` | `random_keep0p5` | 0.5000 | 0.6463 | -0.0650 | 0.8799 | 0.7440 | -0.1011 | 0.8051 | 0.1145 |
| `full_nn` | `random_keep0p2` | 0.1999 | 0.4451 | -0.2661 | 0.7734 | 0.3124 | -0.5328 | 0.6765 | 0.5714 |
| `full_nn` | `random_keep0p1` | 0.1000 | 0.2056 | -0.5056 | 0.4011 | 0.0017 | -0.8434 | 0.2525 | 0.1986 |
| `full_nn` | `fixed_points_16000` | 0.0701 | 0.0600 | -0.6512 | 0.1056 | 0.0000 | -0.8451 | 0.0515 | 0.0259 |
| `full_nn` | `fixed_points_8000` | 0.0350 | 0.0066 | -0.7047 | 0.0096 | 0.0000 | -0.8451 | 0.0040 | 0.0004 |
| `full_nn` | `fixed_points_4000` | 0.0175 | 0.0006 | -0.7107 | 0.0011 | 0.0000 | -0.8451 | 0.0005 | 0.0001 |
| `full_nn` | `structured_b64_keep0p8` | 0.7958 | 0.6374 | -0.0738 | 0.8651 | 0.7759 | -0.0693 | 0.7846 | 0.0752 |
| `full_nn` | `structured_b64_keep0p5` | 0.5024 | 0.4895 | -0.2217 | 0.7847 | 0.5364 | -0.3088 | 0.6852 | 0.1848 |
| `full_nn` | `structured_b64_keep0p2` | 0.1832 | 0.2626 | -0.4486 | 0.5809 | 0.2933 | -0.5519 | 0.4673 | 0.3813 |
| `full_nn` | `structured_b64_keep0p1` | 0.1032 | 0.2141 | -0.4971 | 0.4870 | 0.1164 | -0.7287 | 0.3810 | 0.3629 |
| `full_nn` | `masked_model_keep0p5` | 0.4894 | 0.2647 | -0.4465 | 0.5173 | 0.3097 | -0.5354 | 0.3753 | 0.2066 |
| `full_nn` | `masked_model_keep0p2` | 0.2159 | 0.1211 | -0.5902 | 0.3059 | 0.1144 | -0.7308 | 0.2343 | 0.2942 |
| `full_nn` | `masked_model_keep0p1` | 0.1113 | 0.0744 | -0.6368 | 0.2407 | 0.0486 | -0.7966 | 0.2500 | 0.3619 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.1147 | -0.5966 | 0.5417 | 0.0000 | -0.8451 | 0.4825 | 0.9978 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/ptv3_s3dis/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/ptv3_s3dis/masking_battery_perclass.csv`
