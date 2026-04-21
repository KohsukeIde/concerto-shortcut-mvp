# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_decoder_fullscene_smoke`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Random keep ratios: `0.2`
- Class-wise keep ratios: `0.2`
- Structured keep ratios: ``
- Feature-zero ratios: ``
- Color feature space: `current_0_1`
- Repeats: `1`
- Full-scene scoring: `True`

## Results

| score | variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_nn` | `clean_voxel` | 1.0000 | 0.3850 | +0.0000 | 0.9359 | 0.3748 | 0.6412 | +0.0000 | 0.9535 | 0.9502 | 0.0251 |
| `full_nn` | `random_keep0p2` | 0.2001 | 0.3609 | -0.0241 | 0.9220 | 0.3437 | 0.8182 | +0.1770 | 0.9424 | 0.9074 | 0.0482 |
| `full_nn` | `classwise_keep0p2` | 0.1987 | 0.3447 | -0.0403 | 0.9102 | 0.3255 | 0.8060 | +0.1648 | 0.9324 | 0.8844 | 0.0401 |
| `retained` | `clean_voxel` | 1.0000 | 0.3850 | +0.0000 | 0.9359 | 0.3748 | 0.6412 | +0.0000 | 0.9535 | 0.9502 | 0.0251 |
| `retained` | `random_keep0p2` | 0.2001 | 0.3664 | -0.0186 | 0.9287 | 0.3517 | 0.8442 | +0.2030 | 0.9517 | 0.9260 | 0.0451 |
| `retained` | `classwise_keep0p2` | 0.1987 | 0.3556 | -0.0294 | 0.9225 | 0.3347 | 0.8135 | +0.1723 | 0.9467 | 0.9129 | 0.0392 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_fullscene/concerto_decoder_smoke/masking_battery_summary.csv`
