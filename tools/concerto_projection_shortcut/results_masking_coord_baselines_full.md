# Masking Coord/Majority Baselines

Coordinate-only and train-majority baselines for the masking battery.

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Train batches: `24`
- Train points stored: `800000`
- Majority class: `wall`
- Class-balanced loss: `False`

## Results

| method | variant | keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | wall | floor | p->wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `majority` | `clean_voxel` | 1.0000 | 0.0151 | +0.0000 | 0.3010 | 0.0000 | 0.0000 | 0.3010 | 0.0000 | 1.0000 |
| `majority` | `random_keep0p5` | 0.4999 | 0.0151 | +0.0000 | 0.3011 | 0.0000 | 0.0000 | 0.3011 | 0.0000 | 1.0000 |
| `majority` | `random_keep0p3` | 0.3000 | 0.0150 | -0.0000 | 0.3010 | 0.0000 | 0.0000 | 0.3010 | 0.0000 | 1.0000 |
| `majority` | `random_keep0p2` | 0.2000 | 0.0150 | -0.0000 | 0.3010 | 0.0000 | 0.0000 | 0.3010 | 0.0000 | 1.0000 |
| `majority` | `random_keep0p1` | 0.0999 | 0.0150 | -0.0000 | 0.3006 | 0.0000 | 0.0000 | 0.3006 | 0.0000 | 1.0000 |
| `majority` | `structured_b64_keep0p5` | 0.5022 | 0.0148 | -0.0003 | 0.2954 | 0.0000 | 0.0000 | 0.2954 | 0.0000 | 1.0000 |
| `majority` | `structured_b64_keep0p2` | 0.2055 | 0.0146 | -0.0005 | 0.2917 | 0.0000 | 0.0000 | 0.2917 | 0.0000 | 1.0000 |
| `majority` | `feature_zero1p0` | 1.0000 | 0.0151 | +0.0000 | 0.3010 | 0.0000 | 0.0000 | 0.3010 | 0.0000 | 1.0000 |
| `coord_mlp` | `clean_voxel` | 1.0000 | 0.0726 | +0.0000 | 0.3298 | 0.0189 | 0.0000 | 0.1994 | 0.6889 | 0.2162 |
| `coord_mlp` | `random_keep0p5` | 0.4999 | 0.0727 | +0.0001 | 0.3310 | 0.0189 | 0.0000 | 0.2003 | 0.6914 | 0.2176 |
| `coord_mlp` | `random_keep0p3` | 0.3000 | 0.0731 | +0.0005 | 0.3325 | 0.0190 | 0.0000 | 0.2014 | 0.6938 | 0.2184 |
| `coord_mlp` | `random_keep0p2` | 0.2000 | 0.0732 | +0.0006 | 0.3334 | 0.0189 | 0.0000 | 0.2012 | 0.6959 | 0.2215 |
| `coord_mlp` | `random_keep0p1` | 0.0999 | 0.0735 | +0.0009 | 0.3349 | 0.0192 | 0.0000 | 0.2027 | 0.6977 | 0.2263 |
| `coord_mlp` | `structured_b64_keep0p5` | 0.5022 | 0.0743 | +0.0017 | 0.3391 | 0.0187 | 0.0000 | 0.2156 | 0.6971 | 0.2656 |
| `coord_mlp` | `structured_b64_keep0p2` | 0.2055 | 0.0749 | +0.0023 | 0.3452 | 0.0186 | 0.0000 | 0.2289 | 0.6915 | 0.2264 |
| `coord_mlp` | `feature_zero1p0` | 1.0000 | 0.0726 | +0.0000 | 0.3298 | 0.0189 | 0.0000 | 0.1994 | 0.6889 | 0.2162 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_masking_baselines/coord_majority_full/masking_coord_baselines_summary.csv`
