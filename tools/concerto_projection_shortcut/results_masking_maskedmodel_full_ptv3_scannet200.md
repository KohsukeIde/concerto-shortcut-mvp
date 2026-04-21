# PTv3 v1.5.1 Compatibility Masking Eval

Uses the official Pointcept v1.5.1 model and transform implementation, while reading current `.npy` ScanNet scenes.

## Setup

- Method: `ptv3_scannet200_maskedmodel_full`
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
| `retained` | `clean_voxel` | 1.0000 | 0.3447 | +0.0000 | 0.8361 | 0.3596 | +0.0000 | 0.8065 | 0.4835 |
| `retained` | `random_keep0p2` | 0.2000 | 0.2641 | -0.0806 | 0.8104 | 0.2431 | -0.1166 | 0.7663 | 0.6443 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.0615 | -0.2832 | 0.5964 | 0.0679 | -0.2917 | 0.5111 | 0.8776 |
| `retained` | `classwise_keep0p2` | 0.2001 | 0.2569 | -0.0878 | 0.8092 | 0.2376 | -0.1220 | 0.7644 | 0.6488 |
| `retained` | `structured_b64_keep0p2` | 0.2015 | 0.2626 | -0.0821 | 0.7861 | 0.3065 | -0.0531 | 0.7236 | 0.5155 |
| `retained` | `masked_model_keep0p2` | 0.2035 | 0.2362 | -0.1084 | 0.8154 | 0.4028 | +0.0432 | 0.7734 | 0.2890 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0019 | -0.3428 | 0.2758 | 0.0000 | -0.3596 | 0.2563 | 1.0000 |
| `full_nn` | `clean_voxel` | 1.0000 | 0.3447 | +0.0000 | 0.8361 | 0.3596 | +0.0000 | 0.8065 | 0.4835 |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.2551 | -0.0896 | 0.8025 | 0.2345 | -0.1252 | 0.7533 | 0.6522 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.0419 | -0.3028 | 0.5453 | 0.0159 | -0.3437 | 0.4616 | 0.9356 |
| `full_nn` | `classwise_keep0p2` | 0.2001 | 0.2488 | -0.0959 | 0.8015 | 0.2238 | -0.1358 | 0.7517 | 0.6631 |
| `full_nn` | `structured_b64_keep0p2` | 0.2015 | 0.0836 | -0.2610 | 0.4952 | 0.0642 | -0.2955 | 0.3965 | 0.5938 |
| `full_nn` | `masked_model_keep0p2` | 0.2035 | 0.0441 | -0.3006 | 0.3212 | 0.0703 | -0.2893 | 0.2883 | 0.4897 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0019 | -0.3428 | 0.2758 | 0.0000 | -0.3596 | 0.2563 | 1.0000 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/ptv3_scannet200/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/ptv3_scannet200/masking_battery_perclass.csv`
