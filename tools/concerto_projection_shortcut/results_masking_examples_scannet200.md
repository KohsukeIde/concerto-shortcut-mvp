# Masking Example Export

- Dataset: `scannet200`
- Config: `configs/scannet200/semseg-pt-v3m1-0-base.py`
- Data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- Split: `val`
- Segment key: `segment200`
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
| `scene0011_00` | `random_keep0p2` | 164772 | 32977 | 0.2001 |
| `scene0011_00` | `random_keep0p1` | 164772 | 16404 | 0.0996 |
| `scene0011_00` | `fixed_points_4000` | 164772 | 4000 | 0.0243 |
| `scene0011_00` | `masked_model_keep0p2` | 164772 | 60842 | 0.3244 |
| `scene0011_01` | `clean_voxel` | 163792 | 163792 | 1.0000 |
| `scene0011_01` | `random_keep0p2` | 163792 | 32898 | 0.2009 |
| `scene0011_01` | `random_keep0p1` | 163792 | 16406 | 0.1002 |
| `scene0011_01` | `fixed_points_4000` | 163792 | 4000 | 0.0244 |
| `scene0011_01` | `masked_model_keep0p2` | 163792 | 20215 | 0.1100 |
| `scene0015_00` | `clean_voxel` | 147944 | 147944 | 1.0000 |
| `scene0015_00` | `random_keep0p2` | 147944 | 29807 | 0.2015 |
| `scene0015_00` | `random_keep0p1` | 147944 | 14872 | 0.1005 |
| `scene0015_00` | `fixed_points_4000` | 147944 | 4000 | 0.0270 |
| `scene0015_00` | `masked_model_keep0p2` | 147944 | 13577 | 0.0891 |
| `scene0019_00` | `clean_voxel` | 92713 | 92713 | 1.0000 |
| `scene0019_00` | `random_keep0p2` | 92713 | 18320 | 0.1976 |
| `scene0019_00` | `random_keep0p1` | 92713 | 9250 | 0.0998 |
| `scene0019_00` | `fixed_points_4000` | 92713 | 4000 | 0.0431 |
| `scene0019_00` | `masked_model_keep0p2` | 92713 | 16202 | 0.1724 |
| `scene0019_01` | `clean_voxel` | 113300 | 113300 | 1.0000 |
| `scene0019_01` | `random_keep0p2` | 113300 | 22634 | 0.1998 |
| `scene0019_01` | `random_keep0p1` | 113300 | 11252 | 0.0993 |
| `scene0019_01` | `fixed_points_4000` | 113300 | 4000 | 0.0353 |
| `scene0019_01` | `masked_model_keep0p2` | 113300 | 29652 | 0.2330 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_examples_scannet200.csv`
- Example root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_examples/scannet200`
