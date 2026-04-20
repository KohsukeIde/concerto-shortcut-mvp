# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_linear_origin_classwise`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-lin-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_lora_origin/exp/scannet-lin-origin-e100/model/model_best.pth`
- Random keep ratios: `0.2`
- Class-wise keep ratios: `0.2`
- Structured keep ratios: ``
- Feature-zero ratios: ``
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.7689 | +0.0000 | 0.9197 | 0.6634 | 0.4212 | +0.0000 | 0.8738 | 0.9577 | 0.4082 |
| `classwise_keep0p2` | 0.2000 | 0.7603 | -0.0086 | 0.9180 | 0.6431 | 0.3569 | -0.0643 | 0.8694 | 0.9536 | 0.5408 |
| `random_keep0p2` | 0.1999 | 0.7602 | -0.0087 | 0.9178 | 0.6427 | 0.3498 | -0.0714 | 0.8694 | 0.9538 | 0.5603 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/concerto_linear_classwise/masking_battery_summary.csv`
