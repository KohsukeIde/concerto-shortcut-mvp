# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_decoder_origin_fullscene`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Random keep ratios: `0.5,0.3,0.2,0.1`
- Class-wise keep ratios: `0.2`
- Structured keep ratios: `0.5,0.2`
- Feature-zero ratios: `1.0`
- Color feature space: `current_0_1`
- Repeats: `1`
- Full-scene scoring: `True`

## Results

| score | variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_nn` | `clean_voxel` | 1.0000 | 0.7863 | +0.0000 | 0.9274 | 0.6884 | 0.4202 | +0.0000 | 0.8834 | 0.9667 | 0.4177 |
| `full_nn` | `random_keep0p5` | 0.5000 | 0.7800 | -0.0063 | 0.9252 | 0.6763 | 0.4105 | -0.0097 | 0.8820 | 0.9604 | 0.4551 |
| `full_nn` | `random_keep0p3` | 0.3000 | 0.7687 | -0.0176 | 0.9207 | 0.6562 | 0.3749 | -0.0453 | 0.8742 | 0.9537 | 0.5243 |
| `full_nn` | `random_keep0p2` | 0.1999 | 0.7527 | -0.0336 | 0.9149 | 0.6275 | 0.2948 | -0.1254 | 0.8662 | 0.9455 | 0.6478 |
| `full_nn` | `classwise_keep0p2` | 0.1999 | 0.7506 | -0.0357 | 0.9148 | 0.6239 | 0.2988 | -0.1214 | 0.8656 | 0.9453 | 0.6394 |
| `full_nn` | `random_keep0p1` | 0.1001 | 0.6916 | -0.0947 | 0.8933 | 0.5539 | 0.1502 | -0.2700 | 0.8376 | 0.9233 | 0.7597 |
| `full_nn` | `structured_b64_keep0p5` | 0.5013 | 0.5386 | -0.2477 | 0.8054 | 0.4236 | 0.2595 | -0.1607 | 0.7250 | 0.8277 | 0.5801 |
| `full_nn` | `structured_b64_keep0p2` | 0.2087 | 0.3012 | -0.4851 | 0.6104 | 0.2206 | 0.1387 | -0.2815 | 0.5037 | 0.6180 | 0.6178 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0683 | -0.7180 | 0.5080 | 0.0014 | 0.0000 | -0.4202 | 0.5128 | 0.6502 | 0.9799 |
| `retained` | `clean_voxel` | 1.0000 | 0.7863 | +0.0000 | 0.9274 | 0.6884 | 0.4202 | +0.0000 | 0.8834 | 0.9667 | 0.4177 |
| `retained` | `random_keep0p5` | 0.5000 | 0.7843 | -0.0020 | 0.9273 | 0.6810 | 0.4150 | -0.0052 | 0.8857 | 0.9654 | 0.4522 |
| `retained` | `random_keep0p3` | 0.3000 | 0.7764 | -0.0099 | 0.9246 | 0.6647 | 0.3855 | -0.0347 | 0.8810 | 0.9624 | 0.5150 |
| `retained` | `structured_b64_keep0p5` | 0.5013 | 0.7697 | -0.0166 | 0.9202 | 0.6707 | 0.4548 | +0.0346 | 0.8749 | 0.9642 | 0.4163 |
| `retained` | `random_keep0p2` | 0.1999 | 0.7632 | -0.0231 | 0.9203 | 0.6389 | 0.3091 | -0.1111 | 0.8755 | 0.9572 | 0.6323 |
| `retained` | `classwise_keep0p2` | 0.1999 | 0.7615 | -0.0248 | 0.9202 | 0.6363 | 0.3152 | -0.1050 | 0.8752 | 0.9573 | 0.6203 |
| `retained` | `structured_b64_keep0p2` | 0.2087 | 0.7388 | -0.0475 | 0.9101 | 0.6434 | 0.4380 | +0.0178 | 0.8678 | 0.9690 | 0.4155 |
| `retained` | `random_keep0p1` | 0.1001 | 0.7103 | -0.0760 | 0.9029 | 0.5729 | 0.1731 | -0.2472 | 0.8540 | 0.9421 | 0.7336 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0683 | -0.7180 | 0.5080 | 0.0014 | 0.0000 | -0.4202 | 0.5128 | 0.6502 | 0.9799 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_fullscene/concerto_decoder_full/masking_battery_summary.csv`
