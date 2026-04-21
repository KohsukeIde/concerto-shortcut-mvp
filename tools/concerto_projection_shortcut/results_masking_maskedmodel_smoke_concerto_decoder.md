# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_decoder_origin_maskedmodel_smoke`
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
| `full_nn` | `clean_voxel` | 1.0000 | 0.3877 | +0.0000 | 0.9371 | 0.3801 | 0.6406 | +0.0000 | 0.9555 | 0.9506 | 0.0245 |
| `full_nn` | `random_keep0p2` | 0.1999 | 0.3628 | -0.0249 | 0.9175 | 0.3582 | 0.8280 | +0.1874 | 0.9377 | 0.8969 | 0.0537 |
| `full_nn` | `classwise_keep0p2` | 0.2009 | 0.3574 | -0.0303 | 0.9165 | 0.3515 | 0.7698 | +0.1292 | 0.9364 | 0.9041 | 0.0482 |
| `full_nn` | `fixed_points_4000` | 0.0636 | 0.2082 | -0.1795 | 0.8157 | 0.1399 | 0.0000 | -0.6406 | 0.8258 | 0.7479 | 0.9490 |
| `full_nn` | `structured_b64_keep0p2` | 0.1811 | 0.0850 | -0.3027 | 0.6526 | 0.0200 | 0.0000 | -0.6406 | 0.6597 | 0.5565 | 1.0000 |
| `full_nn` | `masked_model_keep0p2` | 0.1320 | 0.0768 | -0.3108 | 0.4770 | 0.0320 | 0.0000 | -0.6406 | 0.5914 | 0.2491 | 0.0054 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0720 | -0.3156 | 0.7000 | 0.0002 | 0.0000 | -0.6406 | 0.6890 | 0.7290 | 0.9993 |
| `retained` | `clean_voxel` | 1.0000 | 0.3877 | +0.0000 | 0.9371 | 0.3801 | 0.6406 | +0.0000 | 0.9555 | 0.9506 | 0.0245 |
| `retained` | `random_keep0p2` | 0.1999 | 0.3726 | -0.0151 | 0.9291 | 0.3686 | 0.8344 | +0.1939 | 0.9513 | 0.9205 | 0.0366 |
| `retained` | `classwise_keep0p2` | 0.2009 | 0.3668 | -0.0208 | 0.9267 | 0.3682 | 0.7697 | +0.1292 | 0.9485 | 0.9273 | 0.0348 |
| `retained` | `fixed_points_4000` | 0.0636 | 0.2103 | -0.1774 | 0.8087 | 0.1389 | 0.0000 | -0.6406 | 0.8402 | 0.7426 | 0.9455 |
| `retained` | `structured_b64_keep0p2` | 0.1811 | 0.1726 | -0.2151 | 0.9491 | 0.0224 | 0.0000 | -0.6406 | 0.9643 | 0.9805 | nan |
| `retained` | `masked_model_keep0p2` | 0.1320 | 0.1225 | -0.2652 | 0.9703 | 0.1383 | 0.0000 | -0.6406 | 0.9690 | 0.0000 | nan |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0720 | -0.3156 | 0.7000 | 0.0002 | 0.0000 | -0.6406 | 0.6890 | 0.7290 | 0.9993 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_smoke/concerto_decoder/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_maskedmodel_smoke/concerto_decoder/masking_battery_perclass.csv`
