# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `ptv3_supervised_legacy_color`
- Config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- Random keep ratios: `0.2`
- Class-wise keep ratios: ``
- Structured keep ratios: `0.2`
- Feature-zero ratios: `1.0`
- Color feature space: `legacy_minus1_1`
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.1780 | +0.0000 | 0.5819 | 0.1280 | 0.0306 | +0.0000 | 0.5776 | 0.8659 | 0.9249 |
| `structured_b64_keep0p2` | 0.2685 | 0.1586 | -0.0194 | 0.5799 | 0.1509 | 0.0000 | -0.0306 | 0.6293 | 0.8174 | nan |
| `random_keep0p2` | 0.2002 | 0.1221 | -0.0559 | 0.4730 | 0.0856 | 0.0000 | -0.0306 | 0.3729 | 0.8272 | 0.9562 |
| `feature_zero1p0` | 1.0000 | 0.0610 | -0.1170 | 0.4276 | 0.0010 | 0.0000 | -0.0306 | 0.3995 | 0.4959 | 0.4647 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_supervised_legacy_color_smoke/masking_battery_summary.csv`
