# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_decoder_origin_classwise`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Random keep ratios: `0.2`
- Class-wise keep ratios: `0.2`
- Structured keep ratios: ``
- Feature-zero ratios: ``
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.7865 | +0.0000 | 0.9275 | 0.6899 | 0.4246 | +0.0000 | 0.8840 | 0.9666 | 0.4105 |
| `classwise_keep0p2` | 0.2000 | 0.7629 | -0.0236 | 0.9209 | 0.6380 | 0.2916 | -0.1330 | 0.8762 | 0.9575 | 0.6327 |
| `random_keep0p2` | 0.2000 | 0.7626 | -0.0239 | 0.9207 | 0.6375 | 0.3137 | -0.1110 | 0.8765 | 0.9574 | 0.6098 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/concerto_decoder_classwise/masking_battery_summary.csv`
