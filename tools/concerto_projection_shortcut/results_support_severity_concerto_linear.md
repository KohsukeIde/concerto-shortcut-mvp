# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_linear_origin_severity`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-lin-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_lora_origin/exp/scannet-lin-origin-e100/model/model_best.pth`
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
| `full_nn` | `clean_voxel` | 1.0000 | 0.7696 | +0.0000 | 0.9199 | 0.6643 | 0.4221 | +0.0000 | 0.8738 | 0.9576 | 0.4027 |
| `full_nn` | `random_keep0p8` | 0.8001 | 0.7705 | +0.0010 | 0.9204 | 0.6645 | 0.4156 | -0.0066 | 0.8735 | 0.9564 | 0.4231 |
| `full_nn` | `random_keep0p5` | 0.5002 | 0.7682 | -0.0014 | 0.9197 | 0.6587 | 0.3929 | -0.0292 | 0.8722 | 0.9529 | 0.4753 |
| `full_nn` | `random_keep0p2` | 0.1999 | 0.7516 | -0.0180 | 0.9135 | 0.6334 | 0.3435 | -0.0787 | 0.8621 | 0.9439 | 0.5595 |
| `full_nn` | `fixed_points_16000` | 0.1952 | 0.7264 | -0.0432 | 0.9038 | 0.6086 | 0.2799 | -0.1422 | 0.8503 | 0.9352 | 0.5908 |
| `full_nn` | `random_keep0p1` | 0.1001 | 0.7209 | -0.0486 | 0.9011 | 0.6020 | 0.2714 | -0.1507 | 0.8423 | 0.9305 | 0.5928 |
| `full_nn` | `structured_b64_keep0p8` | 0.8046 | 0.6998 | -0.0698 | 0.8858 | 0.5963 | 0.3758 | -0.0463 | 0.8262 | 0.9206 | 0.4586 |
| `full_nn` | `fixed_points_8000` | 0.0976 | 0.6314 | -0.1382 | 0.8606 | 0.5056 | 0.1652 | -0.2569 | 0.7969 | 0.9098 | 0.6305 |
| `full_nn` | `structured_b64_keep0p5` | 0.5016 | 0.5350 | -0.2346 | 0.8004 | 0.4306 | 0.2642 | -0.1580 | 0.7081 | 0.8231 | 0.5707 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.4675 | -0.3020 | 0.7609 | 0.3612 | 0.0798 | -0.3424 | 0.6945 | 0.8416 | 0.6455 |
| `full_nn` | `masked_model_keep0p5` | 0.5047 | 0.3761 | -0.3935 | 0.6434 | 0.3209 | 0.2085 | -0.2137 | 0.5665 | 0.5617 | 0.5350 |
| `full_nn` | `structured_b64_keep0p2` | 0.1993 | 0.3008 | -0.4688 | 0.6151 | 0.2023 | 0.1183 | -0.3039 | 0.5014 | 0.6124 | 0.6263 |
| `full_nn` | `structured_b64_keep0p1` | 0.1108 | 0.1966 | -0.5729 | 0.4960 | 0.1252 | 0.0411 | -0.3810 | 0.3838 | 0.4976 | 0.5306 |
| `full_nn` | `masked_model_keep0p2` | 0.2031 | 0.1961 | -0.5735 | 0.4691 | 0.1443 | 0.0895 | -0.3327 | 0.4047 | 0.4063 | 0.6300 |
| `full_nn` | `masked_model_keep0p1` | 0.0960 | 0.1249 | -0.6447 | 0.3552 | 0.0865 | 0.0444 | -0.3777 | 0.3042 | 0.3093 | 0.4931 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0391 | -0.7305 | 0.1906 | 0.0103 | 0.0000 | -0.4221 | 0.0155 | 0.4526 | 0.0086 |
| `retained` | `clean_voxel` | 1.0000 | 0.7696 | +0.0000 | 0.9199 | 0.6643 | 0.4221 | +0.0000 | 0.8738 | 0.9576 | 0.4027 |
| `retained` | `masked_model_keep0p2` | 0.2031 | 0.7972 | +0.0276 | 0.9388 | 0.6863 | 0.4524 | +0.0303 | 0.9015 | 0.9746 | 0.4414 |
| `retained` | `masked_model_keep0p5` | 0.5047 | 0.7811 | +0.0115 | 0.9243 | 0.6808 | 0.4053 | -0.0168 | 0.8839 | 0.9651 | 0.4240 |
| `retained` | `random_keep0p8` | 0.8001 | 0.7715 | +0.0019 | 0.9210 | 0.6655 | 0.4160 | -0.0061 | 0.8744 | 0.9578 | 0.4227 |
| `retained` | `random_keep0p5` | 0.5002 | 0.7713 | +0.0017 | 0.9214 | 0.6621 | 0.3948 | -0.0273 | 0.8749 | 0.9569 | 0.4744 |
| `retained` | `structured_b64_keep0p8` | 0.8046 | 0.7698 | +0.0002 | 0.9189 | 0.6642 | 0.4246 | +0.0025 | 0.8722 | 0.9567 | 0.4239 |
| `retained` | `masked_model_keep0p1` | 0.0960 | 0.7641 | -0.0055 | 0.9201 | 0.6543 | 0.2394 | -0.1827 | 0.8668 | 0.9820 | 0.6373 |
| `retained` | `random_keep0p2` | 0.1999 | 0.7597 | -0.0099 | 0.9178 | 0.6432 | 0.3528 | -0.0693 | 0.8693 | 0.9538 | 0.5481 |
| `retained` | `structured_b64_keep0p5` | 0.5016 | 0.7567 | -0.0129 | 0.9125 | 0.6609 | 0.3708 | -0.0513 | 0.8643 | 0.9560 | 0.5138 |
| `retained` | `fixed_points_16000` | 0.1952 | 0.7565 | -0.0131 | 0.9167 | 0.6473 | 0.3546 | -0.0675 | 0.8742 | 0.9525 | 0.5108 |
| `retained` | `structured_b64_keep0p2` | 0.1993 | 0.7448 | -0.0248 | 0.9073 | 0.6478 | 0.4404 | +0.0183 | 0.8541 | 0.9534 | 0.4347 |
| `retained` | `random_keep0p1` | 0.1001 | 0.7348 | -0.0348 | 0.9088 | 0.6177 | 0.2883 | -0.1338 | 0.8549 | 0.9479 | 0.5792 |
| `retained` | `structured_b64_keep0p1` | 0.1108 | 0.6994 | -0.0702 | 0.9002 | 0.5838 | 0.3266 | -0.0955 | 0.8507 | 0.9519 | 0.5292 |
| `retained` | `fixed_points_8000` | 0.0976 | 0.6976 | -0.0720 | 0.8915 | 0.5844 | 0.2401 | -0.1820 | 0.8427 | 0.9430 | 0.5863 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.5677 | -0.2019 | 0.8221 | 0.4717 | 0.1401 | -0.2821 | 0.7672 | 0.9108 | 0.6221 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0391 | -0.7305 | 0.1906 | 0.0103 | 0.0000 | -0.4221 | 0.0155 | 0.4526 | 0.0086 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/concerto_linear/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/concerto_linear/masking_battery_perclass.csv`
