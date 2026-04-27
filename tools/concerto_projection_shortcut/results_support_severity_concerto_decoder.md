# Masking Battery Pilot

Voxel-level clean/masked evaluation for shortcut-compatible sparsity. Clean and masked rows use the same input-point evaluation space.

## Setup

- Method: `concerto_decoder_origin_severity`
- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
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
| `full_nn` | `clean_voxel` | 1.0000 | 0.7864 | +0.0000 | 0.9273 | 0.6887 | 0.4216 | +0.0000 | 0.8837 | 0.9665 | 0.4142 |
| `full_nn` | `random_keep0p8` | 0.7998 | 0.7857 | -0.0007 | 0.9273 | 0.6858 | 0.4135 | -0.0080 | 0.8838 | 0.9648 | 0.4328 |
| `full_nn` | `random_keep0p5` | 0.5000 | 0.7789 | -0.0075 | 0.9248 | 0.6741 | 0.4083 | -0.0132 | 0.8807 | 0.9603 | 0.4624 |
| `full_nn` | `random_keep0p2` | 0.2000 | 0.7522 | -0.0342 | 0.9150 | 0.6288 | 0.3089 | -0.1126 | 0.8661 | 0.9454 | 0.6314 |
| `full_nn` | `structured_b64_keep0p8` | 0.8044 | 0.7122 | -0.0742 | 0.8912 | 0.6285 | 0.3692 | -0.0523 | 0.8341 | 0.9230 | 0.4569 |
| `full_nn` | `fixed_points_16000` | 0.1952 | 0.7076 | -0.0789 | 0.8976 | 0.5779 | 0.1926 | -0.2289 | 0.8452 | 0.9296 | 0.6970 |
| `full_nn` | `random_keep0p1` | 0.1000 | 0.6909 | -0.0955 | 0.8930 | 0.5532 | 0.1351 | -0.2864 | 0.8361 | 0.9238 | 0.7780 |
| `full_nn` | `fixed_points_8000` | 0.0976 | 0.5857 | -0.2008 | 0.8417 | 0.4495 | 0.0745 | -0.3470 | 0.7829 | 0.8720 | 0.8315 |
| `full_nn` | `structured_b64_keep0p5` | 0.4887 | 0.5245 | -0.2619 | 0.7926 | 0.4183 | 0.2444 | -0.1771 | 0.7047 | 0.8135 | 0.5924 |
| `full_nn` | `fixed_points_4000` | 0.0488 | 0.4011 | -0.3853 | 0.7293 | 0.2913 | 0.0216 | -0.3999 | 0.6789 | 0.7587 | 0.8220 |
| `full_nn` | `masked_model_keep0p5` | 0.5105 | 0.3736 | -0.4128 | 0.6323 | 0.3152 | 0.1828 | -0.2388 | 0.5546 | 0.5443 | 0.5749 |
| `full_nn` | `structured_b64_keep0p2` | 0.1984 | 0.2945 | -0.4920 | 0.6053 | 0.2043 | 0.1059 | -0.3157 | 0.5062 | 0.6086 | 0.7064 |
| `full_nn` | `structured_b64_keep0p1` | 0.1059 | 0.1974 | -0.5891 | 0.4855 | 0.1195 | 0.0395 | -0.3820 | 0.3934 | 0.4706 | 0.5916 |
| `full_nn` | `masked_model_keep0p2` | 0.2046 | 0.1892 | -0.5973 | 0.4504 | 0.1431 | 0.0657 | -0.3559 | 0.4025 | 0.3835 | 0.5791 |
| `full_nn` | `masked_model_keep0p1` | 0.1027 | 0.1126 | -0.6739 | 0.3867 | 0.0669 | 0.0367 | -0.3848 | 0.3560 | 0.3459 | 0.5539 |
| `full_nn` | `feature_zero1p0` | 1.0000 | 0.0682 | -0.7183 | 0.5077 | 0.0015 | 0.0000 | -0.4216 | 0.5131 | 0.6499 | 0.9835 |
| `retained` | `clean_voxel` | 1.0000 | 0.7864 | +0.0000 | 0.9273 | 0.6887 | 0.4216 | +0.0000 | 0.8837 | 0.9665 | 0.4142 |
| `retained` | `random_keep0p8` | 0.7998 | 0.7874 | +0.0009 | 0.9281 | 0.6875 | 0.4158 | -0.0057 | 0.8852 | 0.9666 | 0.4313 |
| `retained` | `structured_b64_keep0p8` | 0.8044 | 0.7863 | -0.0001 | 0.9267 | 0.6931 | 0.4222 | +0.0006 | 0.8846 | 0.9662 | 0.4239 |
| `retained` | `masked_model_keep0p5` | 0.5105 | 0.7847 | -0.0017 | 0.9294 | 0.6846 | 0.3926 | -0.0290 | 0.8873 | 0.9750 | 0.4121 |
| `retained` | `random_keep0p5` | 0.5000 | 0.7835 | -0.0030 | 0.9271 | 0.6787 | 0.4136 | -0.0080 | 0.8847 | 0.9653 | 0.4584 |
| `retained` | `masked_model_keep0p2` | 0.2046 | 0.7797 | -0.0067 | 0.9300 | 0.6890 | 0.3722 | -0.0494 | 0.9074 | 0.9790 | 0.4012 |
| `retained` | `structured_b64_keep0p5` | 0.4887 | 0.7720 | -0.0145 | 0.9200 | 0.6643 | 0.4072 | -0.0143 | 0.8762 | 0.9653 | 0.4119 |
| `retained` | `masked_model_keep0p1` | 0.1027 | 0.7667 | -0.0198 | 0.9361 | 0.6500 | 0.3989 | -0.0227 | 0.9085 | 0.9861 | 0.4228 |
| `retained` | `random_keep0p2` | 0.2000 | 0.7629 | -0.0236 | 0.9204 | 0.6411 | 0.3243 | -0.0972 | 0.8753 | 0.9573 | 0.6145 |
| `retained` | `structured_b64_keep0p2` | 0.1984 | 0.7568 | -0.0297 | 0.9141 | 0.6534 | 0.5144 | +0.0929 | 0.8675 | 0.9644 | 0.3577 |
| `retained` | `fixed_points_16000` | 0.1952 | 0.7513 | -0.0351 | 0.9155 | 0.6363 | 0.3037 | -0.1179 | 0.8754 | 0.9525 | 0.5750 |
| `retained` | `structured_b64_keep0p1` | 0.1059 | 0.7207 | -0.0657 | 0.9001 | 0.6158 | 0.3706 | -0.0510 | 0.8394 | 0.9640 | 0.4649 |
| `retained` | `random_keep0p1` | 0.1000 | 0.7090 | -0.0774 | 0.9026 | 0.5723 | 0.1526 | -0.2689 | 0.8521 | 0.9429 | 0.7592 |
| `retained` | `fixed_points_8000` | 0.0976 | 0.6736 | -0.1128 | 0.8818 | 0.5505 | 0.1679 | -0.2537 | 0.8350 | 0.9212 | 0.7374 |
| `retained` | `fixed_points_4000` | 0.0488 | 0.5226 | -0.2638 | 0.8009 | 0.4133 | 0.0801 | -0.3415 | 0.7538 | 0.8404 | 0.7854 |
| `retained` | `feature_zero1p0` | 1.0000 | 0.0682 | -0.7183 | 0.5077 | 0.0015 | 0.0000 | -0.4216 | 0.5131 | 0.6499 | 0.9835 |

## Interpretation Gate

- Strong smoke signal: random keep 0.2 retains unexpectedly high mIoU/per-class IoU relative to clean, motivating supervised/coord-only/ranking battery.
- Weak smoke signal: random keep 0.2 collapses similarly to ordinary sparsity, so masking is a support experiment rather than the main axis.
- This pilot alone is not shortcut proof; majority/coord-only/supervised baselines and structured/feature-only masks are required for the full claim.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/concerto_decoder/masking_battery_summary.csv`
- Per-class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/support_severity/concerto_decoder/masking_battery_perclass.csv`
