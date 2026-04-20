# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_supervised_scannet200_v151`
- Official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- Config: `configs/scannet200/semseg-pt-v3m1-0-base.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet200-semseg-pt-v3m1-0-base/model/model_best.pth`
- Data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- Segment key: `segment200`
- Focus class: `picture`
- Confusion class: `wall`

## Results

| variant | keep | mIoU | ΔmIoU | allAcc | focus IoU | Δfocus | confusion IoU | focus->confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.1623 | +0.0000 | 0.8256 | 0.8788 | +0.0000 | 0.8200 | 0.0295 |
| `random_keep0p2` | 0.1994 | 0.1242 | -0.0382 | 0.8023 | 0.0000 | -0.8788 | 0.7966 | 1.0000 |
| `structured_b64_keep0p2` | 0.2098 | 0.1219 | -0.0404 | 0.8071 | 0.0000 | -0.8788 | 0.7767 | nan |
| `feature_zero1p0` | 1.0000 | 0.0018 | -0.1606 | 0.2743 | 0.0000 | -0.8788 | 0.2614 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_scannet200_v151_smoke/masking_battery_summary.csv`
