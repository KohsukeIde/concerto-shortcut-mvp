# Masking Example Export

- Dataset: `s3dis`
- Config: `configs/s3dis/semseg-pt-v3m1-0-rpe.py`
- Data root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/concerto_s3dis_imagepoint/s3dis`
- Split: `Area_5`
- Segment key: `segment`
- Scenes: `WC_1, WC_2, conferenceRoom_1, conferenceRoom_2, conferenceRoom_3`

## Variants

- `clean_voxel`: kind=`clean`, keep_ratio=1.0, fixed_count=0
- `random_keep0p2`: kind=`random_drop`, keep_ratio=0.2, fixed_count=0
- `random_keep0p1`: kind=`random_drop`, keep_ratio=0.1, fixed_count=0
- `fixed_points_4000`: kind=`fixed_count_drop`, keep_ratio=1.0, fixed_count=4000
- `masked_model_keep0p2`: kind=`masked_model_drop_raw`, keep_ratio=0.2, fixed_count=0

## Summary

| scene | variant | base points | masked points | observed keep |
|---|---|---:|---:|---:|
| `WC_1` | `clean_voxel` | 180672 | 180672 | 1.0000 |
| `WC_1` | `random_keep0p2` | 180672 | 36206 | 0.2004 |
| `WC_1` | `random_keep0p1` | 180672 | 18103 | 0.1002 |
| `WC_1` | `fixed_points_4000` | 180672 | 4000 | 0.0221 |
| `WC_1` | `masked_model_keep0p2` | 180672 | 66877 | 0.3722 |
| `WC_2` | `clean_voxel` | 161640 | 161640 | 1.0000 |
| `WC_2` | `random_keep0p2` | 161640 | 32544 | 0.2013 |
| `WC_2` | `random_keep0p1` | 161640 | 16428 | 0.1016 |
| `WC_2` | `fixed_points_4000` | 161640 | 4000 | 0.0247 |
| `WC_2` | `masked_model_keep0p2` | 161640 | 50091 | 0.3073 |
| `conferenceRoom_1` | `clean_voxel` | 263531 | 263531 | 1.0000 |
| `conferenceRoom_1` | `random_keep0p2` | 263531 | 52819 | 0.2004 |
| `conferenceRoom_1` | `random_keep0p1` | 263531 | 26304 | 0.0998 |
| `conferenceRoom_1` | `fixed_points_4000` | 263531 | 4000 | 0.0152 |
| `conferenceRoom_1` | `masked_model_keep0p2` | 263531 | 10743 | 0.0402 |
| `conferenceRoom_2` | `clean_voxel` | 489927 | 489927 | 1.0000 |
| `conferenceRoom_2` | `random_keep0p2` | 489927 | 98117 | 0.2003 |
| `conferenceRoom_2` | `random_keep0p1` | 489927 | 49124 | 0.1003 |
| `conferenceRoom_2` | `fixed_points_4000` | 489927 | 4000 | 0.0082 |
| `conferenceRoom_2` | `masked_model_keep0p2` | 489927 | 61438 | 0.1230 |
| `conferenceRoom_3` | `clean_voxel` | 383319 | 383319 | 1.0000 |
| `conferenceRoom_3` | `random_keep0p2` | 383319 | 76872 | 0.2005 |
| `conferenceRoom_3` | `random_keep0p1` | 383319 | 38293 | 0.0999 |
| `conferenceRoom_3` | `fixed_points_4000` | 383319 | 4000 | 0.0104 |
| `conferenceRoom_3` | `masked_model_keep0p2` | 383319 | 32968 | 0.0830 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_examples_s3dis.csv`
- Example root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_examples/s3dis`
