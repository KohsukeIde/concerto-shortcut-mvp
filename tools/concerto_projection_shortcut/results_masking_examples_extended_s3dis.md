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
- `fixed_points_8000`: kind=`fixed_count_drop`, keep_ratio=1.0, fixed_count=8000
- `fixed_points_4000`: kind=`fixed_count_drop`, keep_ratio=1.0, fixed_count=4000
- `structured_b64_keep0p2`: kind=`structured_drop`, keep_ratio=0.2, fixed_count=0
- `masked_model_keep0p2`: kind=`masked_model_drop_raw`, keep_ratio=0.2, fixed_count=0

## Summary

| scene | variant | base points | masked points | observed keep |
|---|---|---:|---:|---:|
| `WC_1` | `clean_voxel` | 180672 | 180672 | 1.0000 |
| `WC_1` | `random_keep0p2` | 180672 | 36231 | 0.2005 |
| `WC_1` | `random_keep0p1` | 180672 | 18066 | 0.1000 |
| `WC_1` | `fixed_points_8000` | 180672 | 8000 | 0.0443 |
| `WC_1` | `fixed_points_4000` | 180672 | 4000 | 0.0221 |
| `WC_1` | `structured_b64_keep0p2` | 180672 | 28281 | 0.1565 |
| `WC_1` | `masked_model_keep0p2` | 180672 | 21286 | 0.1135 |
| `WC_2` | `clean_voxel` | 161640 | 161640 | 1.0000 |
| `WC_2` | `random_keep0p2` | 161640 | 32183 | 0.1991 |
| `WC_2` | `random_keep0p1` | 161640 | 16286 | 0.1008 |
| `WC_2` | `fixed_points_8000` | 161640 | 8000 | 0.0495 |
| `WC_2` | `fixed_points_4000` | 161640 | 4000 | 0.0247 |
| `WC_2` | `structured_b64_keep0p2` | 161640 | 51477 | 0.3185 |
| `WC_2` | `masked_model_keep0p2` | 161640 | 8928 | 0.0495 |
| `conferenceRoom_1` | `clean_voxel` | 263531 | 263531 | 1.0000 |
| `conferenceRoom_1` | `random_keep0p2` | 263531 | 52796 | 0.2003 |
| `conferenceRoom_1` | `random_keep0p1` | 263531 | 26342 | 0.1000 |
| `conferenceRoom_1` | `fixed_points_8000` | 263531 | 8000 | 0.0304 |
| `conferenceRoom_1` | `fixed_points_4000` | 263531 | 4000 | 0.0152 |
| `conferenceRoom_1` | `structured_b64_keep0p2` | 263531 | 43027 | 0.1633 |
| `conferenceRoom_1` | `masked_model_keep0p2` | 263531 | 59120 | 0.2227 |
| `conferenceRoom_2` | `clean_voxel` | 489927 | 489927 | 1.0000 |
| `conferenceRoom_2` | `random_keep0p2` | 489927 | 98298 | 0.2006 |
| `conferenceRoom_2` | `random_keep0p1` | 489927 | 49141 | 0.1003 |
| `conferenceRoom_2` | `fixed_points_8000` | 489927 | 8000 | 0.0163 |
| `conferenceRoom_2` | `fixed_points_4000` | 489927 | 4000 | 0.0082 |
| `conferenceRoom_2` | `structured_b64_keep0p2` | 489927 | 90813 | 0.1854 |
| `conferenceRoom_2` | `masked_model_keep0p2` | 489927 | 31221 | 0.0617 |
| `conferenceRoom_3` | `clean_voxel` | 383319 | 383319 | 1.0000 |
| `conferenceRoom_3` | `random_keep0p2` | 383319 | 76528 | 0.1996 |
| `conferenceRoom_3` | `random_keep0p1` | 383319 | 38448 | 0.1003 |
| `conferenceRoom_3` | `fixed_points_8000` | 383319 | 8000 | 0.0209 |
| `conferenceRoom_3` | `fixed_points_4000` | 383319 | 4000 | 0.0104 |
| `conferenceRoom_3` | `structured_b64_keep0p2` | 383319 | 88087 | 0.2298 |
| `conferenceRoom_3` | `masked_model_keep0p2` | 383319 | 44157 | 0.1134 |

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_examples_extended_s3dis.csv`
- Example root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_examples_extended/s3dis`
