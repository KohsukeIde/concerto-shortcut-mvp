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
| `clean_voxel` | 1.0000 | 0.7401 | +0.0000 | 0.9072 | 0.2397 | +0.0000 | 0.8751 | 0.9705 | 0.0020 |
| `random_keep0p2` | 0.2003 | 0.6865 | -0.0537 | 0.8827 | 0.0000 | -0.2397 | 0.8593 | 0.9640 | 1.0000 |
| `structured_b64_keep0p2` | 0.1767 | 0.5767 | -0.1634 | 0.8531 | 0.0000 | -0.2397 | 0.8109 | 0.9604 | nan |
| `feature_zero1p0` | 1.0000 | 0.0262 | -0.7139 | 0.3531 | 0.0000 | -0.2397 | 0.3213 | 0.2033 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_v151_compat_smoke/masking_battery_summary.csv`
