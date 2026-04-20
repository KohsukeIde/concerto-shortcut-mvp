# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_decoder_origin`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Random keep ratios: `0.5,0.3,0.2,0.1`
- Structured keep ratios: `0.5,0.2`
- Feature-zero ratios: `1.0`
- Repeats: `1`

## Results

| variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_voxel` | 1.0000 | 0.7874 | +0.0000 | 0.9277 | 0.6886 | 0.4153 | +0.0000 | 0.8839 | 0.9666 | 0.4220 |
| `random_keep0p5` | 0.5000 | 0.7839 | -0.0035 | 0.9269 | 0.6804 | 0.4127 | -0.0026 | 0.8849 | 0.9652 | 0.4583 |
| `random_keep0p3` | 0.3001 | 0.7773 | -0.0101 | 0.9249 | 0.6642 | 0.3882 | -0.0271 | 0.8814 | 0.9620 | 0.5172 |
| `structured_b64_keep0p5` | 0.5028 | 0.7639 | -0.0235 | 0.9199 | 0.6714 | 0.4060 | -0.0093 | 0.8735 | 0.9666 | 0.4672 |
| `random_keep0p2` | 0.2000 | 0.7636 | -0.0238 | 0.9207 | 0.6404 | 0.3215 | -0.0938 | 0.8752 | 0.9574 | 0.6126 |
| `structured_b64_keep0p2` | 0.2068 | 0.7569 | -0.0305 | 0.9134 | 0.6614 | 0.4623 | +0.0470 | 0.8718 | 0.9640 | 0.3949 |
| `random_keep0p1` | 0.1001 | 0.7121 | -0.0753 | 0.9027 | 0.5767 | 0.1887 | -0.2266 | 0.8522 | 0.9423 | 0.7384 |
| `feature_zero1p0` | 1.0000 | 0.0680 | -0.7194 | 0.5076 | 0.0015 | 0.0000 | -0.4153 | 0.5130 | 0.6497 | 0.9827 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/masking_battery_full/masking_battery_summary.csv`
