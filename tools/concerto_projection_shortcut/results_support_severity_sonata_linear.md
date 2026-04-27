# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `sonata_linear_scannet_severity`
- Config: `configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/sonata/sonata_scannet_linear_merged.pth`
- Random keep ratios: `0.8,0.5,0.2,0.1`
- Class-wise keep ratios: ``
- Structured keep ratios: `0.8,0.5,0.2,0.1`
- Masked-model keep ratios: `0.5,0.2,0.1`
- Feature-zero ratios: `1.0`
- Color feature space: `current_0_1`
- Repeats: `1`
- Full-scene scoring: `True`

## Results

| score | variant | observed keep | mIoU | ΔmIoU | allAcc | weak mIoU | picture | Δpicture | wall | floor | p->wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_nn` | `clean_voxel` | 1.0000 | 0.7164 | +0.0000 | 0.8991 | 0.6122 | 0.3692 | +0.0000 | 0.8579 | 0.9596 | 0.4542 |
| `full_nn` | `random_keep0p8` | 0.8000 | 0.7172 | +0.0008 | 0.8993 | 0.6138 | 0.3583 | -0.0109 | 0.8577 | 0.9577 | 0.4802 |
| `full_nn` | `random_keep0p5` | 0.5000 | 0.7119 | -0.0045 | 0.8964 | 0.6064 | 0.3279 | -0.0412 | 0.8530 | 0.9527 | 0.5394 |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.6867 | -0.0297 | 0.8868 | 0.5705 | 0.2197 | -0.1494 | 0.8409 | 0.9360 | 0.7036 |
| `full_nn` | `random_keep0p1` | 0.0999 | 0.6502 | -0.0662 | 0.8705 | 0.5257 | 0.0864 | -0.2828 | 0.8155 | 0.9152 | 0.8606 |
| `full_nn` | `fixed_points_16000` | 0.1952 | 0.6485 | -0.0679 | 0.8677 | 0.5245 | 0.1167 | -0.2524 | 0.8236 | 0.9233 | 0.8207 |
| `full_nn` | `structured_b64_keep0p8` | 0.8024 | 0.6482 | -0.0682 | 0.8654 | 0.5393 | 0.2863 | -0.0828 | 0.8107 | 0.9209 | 0.5802 |
| `full_nn` | `fixed_points_8000` | 0.0976 | 0.5278 | -0.1886 | 0.8068 | 0.4262 | 0.0316 | -0.3376 | 0.7618 | 0.9036 | 0.9244 |
| `full_nn` | `structured_b64_keep0p5` | 0.4967 | 0.4870 | -0.2294 | 0.7773 | 0.3877 | 0.1865 | -0.1826 | 0.6965 | 0.8241 | 0.6832 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.3845 | -0.3319 | 0.7257 | 0.2820 | 0.0158 | -0.3534 | 0.6780 | 0.8818 | 0.9224 |
| `full_nn` | `masked_model_keep0p5` | 0.4969 | 0.3450 | -0.3714 | 0.6176 | 0.2830 | 0.1773 | -0.1918 | 0.5367 | 0.5571 | 0.6465 |
| `full_nn` | `structured_b64_keep0p2` | 0.1984 | 0.2761 | -0.4403 | 0.6008 | 0.1836 | 0.0691 | -0.3000 | 0.4925 | 0.6121 | 0.6946 |
| `full_nn` | `masked_model_keep0p2` | 0.2196 | 0.1789 | -0.5376 | 0.4564 | 0.1339 | 0.0545 | -0.3146 | 0.3988 | 0.3976 | 0.6845 |
| `full_nn` | `structured_b64_keep0p1` | 0.1029 | 0.1622 | -0.5542 | 0.4718 | 0.0930 | 0.0271 | -0.3421 | 0.3850 | 0.4709 | 0.6927 |
| `full_nn` | `masked_model_keep0p1` | 0.1023 | 0.1087 | -0.6077 | 0.3799 | 0.0631 | 0.0263 | -0.3428 | 0.3518 | 0.3274 | 0.7501 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0606 | -0.6558 | 0.4623 | 0.0000 | 0.0000 | -0.3692 | 0.4700 | 0.6279 | 0.7334 |
| `retained` | `clean_voxel` | 1.0000 | 0.7164 | +0.0000 | 0.8991 | 0.6122 | 0.3692 | +0.0000 | 0.8579 | 0.9596 | 0.4542 |
| `retained` | `masked_model_keep0p5` | 0.4969 | 0.7305 | +0.0141 | 0.9041 | 0.6226 | 0.3528 | -0.0163 | 0.8668 | 0.9674 | 0.5100 |
| `retained` | `random_keep0p8` | 0.8000 | 0.7185 | +0.0020 | 0.8999 | 0.6151 | 0.3602 | -0.0090 | 0.8587 | 0.9594 | 0.4786 |
| `retained` | `random_keep0p5` | 0.5000 | 0.7150 | -0.0014 | 0.8982 | 0.6094 | 0.3301 | -0.0390 | 0.8559 | 0.9573 | 0.5373 |
| `retained` | `structured_b64_keep0p8` | 0.8024 | 0.7144 | -0.0020 | 0.8982 | 0.6062 | 0.3364 | -0.0328 | 0.8576 | 0.9592 | 0.5149 |
| `retained` | `masked_model_keep0p2` | 0.2196 | 0.7043 | -0.0122 | 0.9148 | 0.5714 | 0.3556 | -0.0136 | 0.8835 | 0.9727 | 0.3154 |
| `retained` | `random_keep0p2` | 0.2000 | 0.6945 | -0.0219 | 0.8911 | 0.5794 | 0.2294 | -0.1397 | 0.8488 | 0.9454 | 0.6921 |
| `retained` | `structured_b64_keep0p5` | 0.4967 | 0.6915 | -0.0249 | 0.8853 | 0.5961 | 0.3764 | +0.0072 | 0.8397 | 0.9566 | 0.5066 |
| `retained` | `fixed_points_16000` | 0.1952 | 0.6900 | -0.0264 | 0.8889 | 0.5787 | 0.1922 | -0.1770 | 0.8498 | 0.9423 | 0.7286 |
| `retained` | `structured_b64_keep0p2` | 0.1984 | 0.6877 | -0.0287 | 0.8904 | 0.5730 | 0.3502 | -0.0190 | 0.8541 | 0.9550 | 0.4680 |
| `retained` | `random_keep0p1` | 0.0999 | 0.6619 | -0.0545 | 0.8774 | 0.5382 | 0.0961 | -0.2730 | 0.8277 | 0.9288 | 0.8514 |
| `retained` | `structured_b64_keep0p1` | 0.1029 | 0.6431 | -0.0734 | 0.8655 | 0.5107 | 0.2095 | -0.1596 | 0.8110 | 0.9481 | 0.6822 |
| `retained` | `masked_model_keep0p1` | 0.1023 | 0.6291 | -0.0873 | 0.8635 | 0.4203 | 0.1383 | -0.2308 | 0.8421 | 0.9707 | 0.7315 |
| `retained` | `fixed_points_8000` | 0.0976 | 0.6048 | -0.1116 | 0.8495 | 0.5101 | 0.0994 | -0.2697 | 0.8068 | 0.9320 | 0.8603 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.4732 | -0.2432 | 0.7829 | 0.3860 | 0.0736 | -0.2955 | 0.7387 | 0.9222 | 0.8786 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0606 | -0.6558 | 0.4623 | 0.0000 | 0.0000 | -0.3692 | 0.4700 | 0.6279 | 0.7334 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/sonata_linear/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/sonata_linear/masking_battery_perclass.csv`
