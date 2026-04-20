# Masking Coord/Majority Baselines

Coordinate-only and train-majority baselines for the masking battery.

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Train batches: `16`
- Train points stored: `639332`
- Majority class: `wall`
- Class-balanced loss: `False`

## Results

| method | variant | keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | wall | floor | p->wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `majority` | `clean_voxel` | 1.0000 | 0.0143 | +0.0000 | 0.2862 | 0.0000 | 0.0000 | 0.2862 | 0.0000 | 1.0000 |
| `majority` | `random_keep0p2` | 0.1997 | 0.0143 | -0.0000 | 0.2854 | 0.0000 | 0.0000 | 0.2854 | 0.0000 | 1.0000 |
| `coord_mlp` | `clean_voxel` | 1.0000 | 0.0752 | +0.0000 | 0.4178 | 0.0039 | 0.0000 | 0.3179 | 0.7172 | 0.3415 |
| `coord_mlp` | `random_keep0p2` | 0.1997 | 0.0754 | +0.0001 | 0.4199 | 0.0036 | 0.0000 | 0.3193 | 0.7179 | 0.3586 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_masking_baselines/coord_majority_smoke/masking_coord_baselines_summary.csv`
