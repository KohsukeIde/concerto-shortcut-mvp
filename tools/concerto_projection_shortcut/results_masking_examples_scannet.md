# Masking Example Export

- Dataset: `scannet`
- Config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- Data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- Split: `val`
- Segment key: `segment20`
- Scenes: `scene0011_00, scene0011_01, scene0015_00, scene0019_00, scene0019_01`

## Variants

- `clean_voxel`: kind=`clean`, keep_ratio=1.0, fixed_count=0
- `random_keep0p2`: kind=`random_drop`, keep_ratio=0.2, fixed_count=0
- `random_keep0p1`: kind=`random_drop`, keep_ratio=0.1, fixed_count=0
- `fixed_points_4000`: kind=`fixed_count_drop`, keep_ratio=1.0, fixed_count=4000
- `masked_model_keep0p2`: kind=`masked_model_drop_raw`, keep_ratio=0.2, fixed_count=0

## Summary

| scene | variant | base points | masked points | observed keep |
|---|---|---:|---:|---:|
| `scene0011_00` | `clean_voxel` | 164772 | 164772 | 1.0000 |
| `scene0011_00` | `random_keep0p2` | 164772 | 32995 | 0.2002 |
| `scene0011_00` | `random_keep0p1` | 164772 | 16447 | 0.0998 |
| `scene0011_00` | `fixed_points_4000` | 164772 | 4000 | 0.0243 |
| `scene0011_00` | `masked_model_keep0p2` | 164772 | 22486 | 0.1316 |
| `scene0011_01` | `clean_voxel` | 163792 | 163792 | 1.0000 |
| `scene0011_01` | `random_keep0p2` | 163792 | 33049 | 0.2018 |
| `scene0011_01` | `random_keep0p1` | 163792 | 16219 | 0.0990 |
| `scene0011_01` | `fixed_points_4000` | 163792 | 4000 | 0.0244 |
| `scene0011_01` | `masked_model_keep0p2` | 163792 | 58235 | 0.3085 |
| `scene0015_00` | `clean_voxel` | 147944 | 147944 | 1.0000 |
| `scene0015_00` | `random_keep0p2` | 147944 | 29762 | 0.2012 |
| `scene0015_00` | `random_keep0p1` | 147944 | 14983 | 0.1013 |
| `scene0015_00` | `fixed_points_4000` | 147944 | 4000 | 0.0270 |
| `scene0015_00` | `masked_model_keep0p2` | 147944 | 10639 | 0.0737 |
| `scene0019_00` | `clean_voxel` | 92713 | 92713 | 1.0000 |
| `scene0019_00` | `random_keep0p2` | 92713 | 18525 | 0.1998 |
| `scene0019_00` | `random_keep0p1` | 92713 | 9277 | 0.1001 |
| `scene0019_00` | `fixed_points_4000` | 92713 | 4000 | 0.0431 |
| `scene0019_00` | `masked_model_keep0p2` | 92713 | 45118 | 0.4701 |
| `scene0019_01` | `clean_voxel` | 113300 | 113300 | 1.0000 |
| `scene0019_01` | `random_keep0p2` | 113300 | 22647 | 0.1999 |
| `scene0019_01` | `random_keep0p1` | 113300 | 11227 | 0.0991 |
| `scene0019_01` | `fixed_points_4000` | 113300 | 4000 | 0.0353 |
| `scene0019_01` | `masked_model_keep0p2` | 113300 | 56842 | 0.4643 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_examples_scannet.csv`
- Example root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_examples/scannet`
