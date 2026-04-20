# Masking Coord/Majority Baselines

Coordinate-only and train-majority baselines for the masking battery.

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Train batches: `24`
- Train points stored: `800000`
- Majority class: `wall`
- Class-balanced loss: `True`

## Results

| method | variant | keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | wall | floor | p->wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `majority` | `clean_voxel` | 1.0000 | 0.0151 | +0.0000 | 0.3010 | 0.0000 | 0.0000 | 0.3010 | 0.0000 | 1.0000 |
| `majority` | `random_keep0p5` | 0.4999 | 0.0151 | +0.0000 | 0.3010 | 0.0000 | 0.0000 | 0.3010 | 0.0000 | 1.0000 |
| `majority` | `random_keep0p3` | 0.3000 | 0.0150 | -0.0000 | 0.3010 | 0.0000 | 0.0000 | 0.3010 | 0.0000 | 1.0000 |
| `majority` | `random_keep0p2` | 0.2000 | 0.0151 | +0.0000 | 0.3011 | 0.0000 | 0.0000 | 0.3011 | 0.0000 | 1.0000 |
| `majority` | `random_keep0p1` | 0.0999 | 0.0150 | -0.0000 | 0.3008 | 0.0000 | 0.0000 | 0.3008 | 0.0000 | 1.0000 |
| `majority` | `structured_b64_keep0p5` | 0.5012 | 0.0151 | +0.0000 | 0.3015 | 0.0000 | 0.0000 | 0.3015 | 0.0000 | 1.0000 |
| `majority` | `structured_b64_keep0p2` | 0.2013 | 0.0146 | -0.0004 | 0.2926 | 0.0000 | 0.0000 | 0.2926 | 0.0000 | 1.0000 |
| `majority` | `feature_zero1p0` | 1.0000 | 0.0151 | +0.0000 | 0.3010 | 0.0000 | 0.0000 | 0.3010 | 0.0000 | 1.0000 |
| `coord_mlp` | `clean_voxel` | 1.0000 | 0.0707 | +0.0000 | 0.3061 | 0.0220 | 0.0070 | 0.1778 | 0.6645 | 0.1954 |
| `coord_mlp` | `random_keep0p5` | 0.4999 | 0.0711 | +0.0003 | 0.3078 | 0.0221 | 0.0069 | 0.1789 | 0.6677 | 0.1951 |
| `coord_mlp` | `random_keep0p3` | 0.3000 | 0.0711 | +0.0004 | 0.3086 | 0.0219 | 0.0066 | 0.1797 | 0.6690 | 0.1941 |
| `coord_mlp` | `random_keep0p2` | 0.2000 | 0.0715 | +0.0008 | 0.3102 | 0.0222 | 0.0070 | 0.1805 | 0.6727 | 0.2012 |
| `coord_mlp` | `random_keep0p1` | 0.0999 | 0.0719 | +0.0012 | 0.3125 | 0.0223 | 0.0076 | 0.1815 | 0.6776 | 0.1904 |
| `coord_mlp` | `structured_b64_keep0p5` | 0.5012 | 0.0725 | +0.0017 | 0.3155 | 0.0238 | 0.0143 | 0.1909 | 0.6748 | 0.2295 |
| `coord_mlp` | `structured_b64_keep0p2` | 0.2013 | 0.0702 | -0.0005 | 0.3173 | 0.0237 | 0.0069 | 0.2078 | 0.6608 | 0.3026 |
| `coord_mlp` | `feature_zero1p0` | 1.0000 | 0.0707 | +0.0000 | 0.3061 | 0.0220 | 0.0070 | 0.1778 | 0.6645 | 0.1954 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_masking_baselines/coord_majority_balanced_full/masking_coord_baselines_summary.csv`
