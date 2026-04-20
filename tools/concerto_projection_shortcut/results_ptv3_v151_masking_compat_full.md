# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_supervised_v151_compat`
- Official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- Config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- Data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`

## Results

| variant | keep | mIoU | ΔmIoU | allAcc | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.7697 | +0.0000 | 0.9219 | 0.3917 | +0.0000 | 0.8744 | 0.9639 | 0.4694 |
| `random_keep0p5` | 0.5000 | 0.7653 | -0.0044 | 0.9189 | 0.3726 | -0.0190 | 0.8724 | 0.9612 | 0.5258 |
| `random_keep0p3` | 0.2999 | 0.7420 | -0.0278 | 0.9113 | 0.2962 | -0.0955 | 0.8606 | 0.9570 | 0.6579 |
| `random_keep0p2` | 0.2001 | 0.7143 | -0.0554 | 0.8997 | 0.2264 | -0.1653 | 0.8414 | 0.9528 | 0.7448 |
| `random_keep0p1` | 0.1000 | 0.5834 | -0.1863 | 0.8489 | 0.0438 | -0.3479 | 0.7647 | 0.9388 | 0.9352 |
| `classwise_keep0p2` | 0.2001 | 0.7107 | -0.0590 | 0.8977 | 0.2281 | -0.1636 | 0.8394 | 0.9514 | 0.7455 |
| `structured_b64_keep0p5` | 0.5009 | 0.7019 | -0.0678 | 0.8967 | 0.4273 | +0.0356 | 0.8403 | 0.9536 | 0.4616 |
| `structured_b64_keep0p2` | 0.1896 | 0.6521 | -0.1176 | 0.8782 | 0.3756 | -0.0161 | 0.8198 | 0.9446 | 0.5123 |
| `feature_zero1p0` | 1.0000 | 0.0269 | -0.7429 | 0.3612 | 0.0000 | -0.3917 | 0.3223 | 0.2148 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_v151_compat_full/masking_battery_summary.csv`
