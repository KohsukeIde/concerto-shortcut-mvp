# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_decoder_origin_maskedmodel_full`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Random keep ratios: `0.2`
- Class-wise keep ratios: `0.2`
- Structured keep ratios: `0.2`
- Masked-model keep ratios: `0.2`
- Feature-zero ratios: `1.0`
- Color feature space: `current_0_1`
- Repeats: `1`
- Full-scene scoring: `True`

## Results

| score | variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_nn` | `clean_voxel` | 1.0000 | 0.7869 | +0.0000 | 0.9274 | 0.6878 | 0.4213 | +0.0000 | 0.8838 | 0.9667 | 0.4139 |
| `full_nn` | `random_keep0p2` | 0.1999 | 0.7522 | -0.0346 | 0.9152 | 0.6284 | 0.3037 | -0.1176 | 0.8667 | 0.9458 | 0.6342 |
| `full_nn` | `classwise_keep0p2` | 0.2001 | 0.7512 | -0.0356 | 0.9152 | 0.6266 | 0.3047 | -0.1166 | 0.8671 | 0.9458 | 0.6356 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.3930 | -0.3939 | 0.7274 | 0.2842 | 0.0195 | -0.4017 | 0.6791 | 0.7567 | 0.8232 |
| `full_nn` | `structured_b64_keep0p2` | 0.2068 | 0.3030 | -0.4838 | 0.6153 | 0.2189 | 0.1398 | -0.2815 | 0.5102 | 0.6236 | 0.6705 |
| `full_nn` | `masked_model_keep0p2` | 0.2030 | 0.1982 | -0.5887 | 0.4590 | 0.1423 | 0.0950 | -0.3263 | 0.4010 | 0.3915 | 0.6167 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0682 | -0.7187 | 0.5076 | 0.0015 | 0.0000 | -0.4213 | 0.5135 | 0.6488 | 0.9843 |
| `retained` | `clean_voxel` | 1.0000 | 0.7869 | +0.0000 | 0.9274 | 0.6878 | 0.4213 | +0.0000 | 0.8838 | 0.9667 | 0.4139 |
| `retained` | `masked_model_keep0p2` | 0.2030 | 0.7866 | -0.0003 | 0.9307 | 0.6803 | 0.4654 | +0.0441 | 0.8901 | 0.9790 | 0.4059 |
| `retained` | `random_keep0p2` | 0.1999 | 0.7636 | -0.0232 | 0.9208 | 0.6412 | 0.3173 | -0.1040 | 0.8761 | 0.9576 | 0.6186 |
| `retained` | `classwise_keep0p2` | 0.2001 | 0.7620 | -0.0249 | 0.9207 | 0.6388 | 0.3170 | -0.1042 | 0.8766 | 0.9574 | 0.6217 |
| `retained` | `structured_b64_keep0p2` | 0.2068 | 0.7505 | -0.0364 | 0.9157 | 0.6366 | 0.4206 | -0.0007 | 0.8674 | 0.9658 | 0.4940 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.5148 | -0.2721 | 0.7989 | 0.4046 | 0.0745 | -0.3467 | 0.7527 | 0.8387 | 0.7953 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0682 | -0.7187 | 0.5076 | 0.0015 | 0.0000 | -0.4213 | 0.5135 | 0.6488 | 0.9843 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/concerto_decoder/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_full/concerto_decoder/masking_battery_perclass.csv`
