# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `ptv3_supervised_scannet`
- Config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- Random keep ratios: `0.5,0.3,0.2,0.1`
- Structured keep ratios: `0.5,0.2`
- Feature-zero ratios: `1.0`
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.1496 | +0.0000 | 0.4936 | 0.0706 | 0.0937 | +0.0000 | 0.5428 | 0.7366 | 0.5541 |
| `random_keep0p5` | 0.4998 | 0.1455 | -0.0040 | 0.4563 | 0.0669 | 0.0766 | -0.0172 | 0.4615 | 0.7381 | 0.5907 |
| `structured_b64_keep0p5` | 0.4971 | 0.1334 | -0.0162 | 0.4637 | 0.0537 | 0.0766 | -0.0171 | 0.4943 | 0.7325 | 0.4728 |
| `random_keep0p3` | 0.3000 | 0.1201 | -0.0295 | 0.4460 | 0.0373 | 0.0527 | -0.0410 | 0.4215 | 0.7547 | 0.6197 |
| `structured_b64_keep0p2` | 0.1933 | 0.1091 | -0.0404 | 0.4157 | 0.0375 | 0.0197 | -0.0741 | 0.4144 | 0.7000 | 0.4195 |
| `random_keep0p2` | 0.2000 | 0.0900 | -0.0595 | 0.4335 | 0.0204 | 0.0069 | -0.0868 | 0.3859 | 0.7751 | 0.7043 |
| `feature_zero1p0` | 1.0000 | 0.0627 | -0.0869 | 0.4152 | 0.0056 | 0.0000 | -0.0937 | 0.3952 | 0.5113 | 0.8394 |
| `random_keep0p1` | 0.0999 | 0.0496 | -0.0999 | 0.3260 | 0.0060 | 0.0000 | -0.0937 | 0.2685 | 0.6551 | 0.4478 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_ranking/ptv3_supervised_full/masking_battery_summary.csv`
