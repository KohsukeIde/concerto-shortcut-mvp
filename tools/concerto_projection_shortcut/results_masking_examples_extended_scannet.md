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
- `fixed_points_8000`: kind=`fixed_count_drop`, keep_ratio=1.0, fixed_count=8000
- `fixed_points_4000`: kind=`fixed_count_drop`, keep_ratio=1.0, fixed_count=4000
- `structured_b64_keep0p2`: kind=`structured_drop`, keep_ratio=0.2, fixed_count=0
- `masked_model_keep0p2`: kind=`masked_model_drop_raw`, keep_ratio=0.2, fixed_count=0

## Summary

| scene | variant | base points | masked points | observed keep |
|---|---|---:|---:|---:|
| `scene0011_00` | `clean_voxel` | 164772 | 164772 | 1.0000 |
| `scene0011_00` | `random_keep0p2` | 164772 | 32921 | 0.1998 |
| `scene0011_00` | `random_keep0p1` | 164772 | 16658 | 0.1011 |
| `scene0011_00` | `fixed_points_8000` | 164772 | 8000 | 0.0486 |
| `scene0011_00` | `fixed_points_4000` | 164772 | 4000 | 0.0243 |
| `scene0011_00` | `structured_b64_keep0p2` | 164772 | 24180 | 0.1467 |
| `scene0011_00` | `masked_model_keep0p2` | 164772 | 57317 | 0.3033 |
| `scene0011_01` | `clean_voxel` | 163792 | 163792 | 1.0000 |
| `scene0011_01` | `random_keep0p2` | 163792 | 32868 | 0.2007 |
| `scene0011_01` | `random_keep0p1` | 163792 | 16310 | 0.0996 |
| `scene0011_01` | `fixed_points_8000` | 163792 | 8000 | 0.0488 |
| `scene0011_01` | `fixed_points_4000` | 163792 | 4000 | 0.0244 |
| `scene0011_01` | `structured_b64_keep0p2` | 163792 | 23205 | 0.1417 |
| `scene0011_01` | `masked_model_keep0p2` | 163792 | 23897 | 0.1525 |
| `scene0015_00` | `clean_voxel` | 147944 | 147944 | 1.0000 |
| `scene0015_00` | `random_keep0p2` | 147944 | 29443 | 0.1990 |
| `scene0015_00` | `random_keep0p1` | 147944 | 14810 | 0.1001 |
| `scene0015_00` | `fixed_points_8000` | 147944 | 8000 | 0.0541 |
| `scene0015_00` | `fixed_points_4000` | 147944 | 4000 | 0.0270 |
| `scene0015_00` | `structured_b64_keep0p2` | 147944 | 19494 | 0.1318 |
| `scene0015_00` | `masked_model_keep0p2` | 147944 | 67055 | 0.4122 |
| `scene0019_00` | `clean_voxel` | 92713 | 92713 | 1.0000 |
| `scene0019_00` | `random_keep0p2` | 92713 | 18628 | 0.2009 |
| `scene0019_00` | `random_keep0p1` | 92713 | 9406 | 0.1015 |
| `scene0019_00` | `fixed_points_8000` | 92713 | 8000 | 0.0863 |
| `scene0019_00` | `fixed_points_4000` | 92713 | 4000 | 0.0431 |
| `scene0019_00` | `structured_b64_keep0p2` | 92713 | 20318 | 0.2191 |
| `scene0019_00` | `masked_model_keep0p2` | 92713 | 18791 | 0.1970 |
| `scene0019_01` | `clean_voxel` | 113300 | 113300 | 1.0000 |
| `scene0019_01` | `random_keep0p2` | 113300 | 22769 | 0.2010 |
| `scene0019_01` | `random_keep0p1` | 113300 | 11371 | 0.1004 |
| `scene0019_01` | `fixed_points_8000` | 113300 | 8000 | 0.0706 |
| `scene0019_01` | `fixed_points_4000` | 113300 | 4000 | 0.0353 |
| `scene0019_01` | `structured_b64_keep0p2` | 113300 | 1686 | 0.0149 |
| `scene0019_01` | `masked_model_keep0p2` | 113300 | 49035 | 0.4177 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_examples_extended_scannet.csv`
- Example root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_examples_extended/scannet`
