# Latent Subgroup Decoder

Latent subgroup diagnostic and targeted sub-center readout pilot for the `concerto_base_origin` decoder-probe checkpoint.

## Setup

- Config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Train points: `27231`
- Heldout points: `28011`
- Seen val batches: `4`

## Headline

- Base: `base` mIoU=0.4124 (Δ+0.0000), picture=0.6612 (Δ+0.0000), p->wall=0.0277 (Δ+0.0000)
- Best mIoU: `subcenter_tau0p1_lam0p4` mIoU=0.4220 (Δ+0.0096), picture=0.7013 (Δ+0.0400), p->wall=0.0207 (Δ-0.0071)
- Best picture: `subcenter_tau0p2_lam0p4` mIoU=0.4196 (Δ+0.0072), picture=0.7040 (Δ+0.0428), p->wall=0.0236 (Δ-0.0041)
- Best safe picture (mIoU >= base - 0.002): `subcenter_tau0p2_lam0p4` mIoU=0.4196 (Δ+0.0072), picture=0.7040 (Δ+0.0428), p->wall=0.0236 (Δ-0.0041)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `subcenter_tau0p1_lam0p4` | 0.4220 | +0.0096 | 0.3063 | +0.0125 | 0.7013 | +0.0400 | 0.0207 | -0.0071 |
| 2 | `subcenter_tau0p05_lam0p4` | 0.4204 | +0.0079 | 0.3039 | +0.0102 | 0.6893 | +0.0281 | 0.0171 | -0.0106 |
| 3 | `subcenter_tau0p2_lam0p4` | 0.4196 | +0.0072 | 0.3046 | +0.0109 | 0.7040 | +0.0428 | 0.0236 | -0.0041 |
| 4 | `subcenter_tau0p05_lam0p2` | 0.4164 | +0.0040 | 0.3002 | +0.0065 | 0.6899 | +0.0286 | 0.0218 | -0.0059 |
| 5 | `subcenter_tau0p1_lam0p2` | 0.4159 | +0.0035 | 0.2993 | +0.0056 | 0.6846 | +0.0234 | 0.0230 | -0.0047 |
| 6 | `subcenter_tau0p2_lam0p2` | 0.4150 | +0.0026 | 0.2978 | +0.0040 | 0.6768 | +0.0156 | 0.0248 | -0.0030 |
| 7 | `subcenter_adapt_tau0p05_lam0p4` | 0.4147 | +0.0023 | 0.2976 | +0.0039 | 0.6808 | +0.0196 | 0.0242 | -0.0035 |
| 8 | `subcenter_adapt_tau0p1_lam0p4` | 0.4145 | +0.0021 | 0.2972 | +0.0035 | 0.6781 | +0.0169 | 0.0248 | -0.0030 |
| 9 | `subcenter_tau0p05_lam0p1` | 0.4141 | +0.0017 | 0.2965 | +0.0028 | 0.6733 | +0.0121 | 0.0242 | -0.0035 |
| 10 | `subcenter_tau0p1_lam0p1` | 0.4141 | +0.0016 | 0.2963 | +0.0025 | 0.6720 | +0.0108 | 0.0248 | -0.0030 |
| 11 | `subcenter_adapt_tau0p2_lam0p4` | 0.4140 | +0.0015 | 0.2963 | +0.0025 | 0.6737 | +0.0125 | 0.0260 | -0.0018 |
| 12 | `subcenter_adapt_tau0p05_lam0p2` | 0.4135 | +0.0011 | 0.2955 | +0.0018 | 0.6696 | +0.0084 | 0.0260 | -0.0018 |
| 13 | `subcenter_tau0p2_lam0p1` | 0.4135 | +0.0011 | 0.2955 | +0.0018 | 0.6689 | +0.0077 | 0.0260 | -0.0018 |
| 14 | `subcenter_adapt_tau0p1_lam0p2` | 0.4133 | +0.0009 | 0.2953 | +0.0015 | 0.6680 | +0.0067 | 0.0260 | -0.0018 |
| 15 | `subcenter_tau0p05_lam0p05` | 0.4133 | +0.0009 | 0.2950 | +0.0013 | 0.6661 | +0.0049 | 0.0260 | -0.0018 |
| 16 | `subcenter_tau0p1_lam0p05` | 0.4131 | +0.0007 | 0.2947 | +0.0010 | 0.6641 | +0.0028 | 0.0260 | -0.0018 |
| 17 | `subcenter_adapt_tau0p2_lam0p2` | 0.4130 | +0.0006 | 0.2947 | +0.0010 | 0.6655 | +0.0043 | 0.0266 | -0.0012 |
| 18 | `subcenter_adapt_tau0p05_lam0p1` | 0.4130 | +0.0006 | 0.2944 | +0.0007 | 0.6633 | +0.0021 | 0.0266 | -0.0012 |
| 19 | `subcenter_adapt_tau0p1_lam0p1` | 0.4130 | +0.0006 | 0.2945 | +0.0008 | 0.6641 | +0.0028 | 0.0266 | -0.0012 |
| 20 | `subcenter_tau0p2_lam0p05` | 0.4129 | +0.0005 | 0.2943 | +0.0006 | 0.6627 | +0.0015 | 0.0277 | +0.0000 |

## Val Cluster Diagnostic

| domain | class | counterpart | cluster | count | target top1 | counterpart top1 | target-counterpart margin |
|---|---|---|---:|---:|---:|---:|---:|
| val | picture | wall | 1 | 1357 | 0.7708 | 0.0000 | 3.8887 |
| val | picture | wall | 0 | 218 | 0.1422 | 0.2156 | -2.0996 |
| val | picture | wall | 3 | 61 | 0.9508 | 0.0000 | 1.7617 |
| val | picture | wall | 2 | 58 | 0.7241 | 0.0000 | 3.0714 |
| val | door | wall | 2 | 4539 | 1.0000 | 0.0000 | 4.6040 |
| val | door | wall | 0 | 4342 | 0.3819 | 0.2692 | -0.5649 |
| val | door | wall | 2 | 2852 | 1.0000 | 0.0000 | 4.6213 |
| val | door | wall | 2 | 2709 | 1.0000 | 0.0000 | 4.5211 |
| val | door | wall | 0 | 1386 | 0.8759 | 0.1219 | 1.8101 |
| val | door | wall | 1 | 919 | 1.0000 | 0.0000 | 4.9504 |
| val | door | wall | 2 | 643 | 1.0000 | 0.0000 | 5.1656 |
| val | door | wall | 1 | 573 | 0.9546 | 0.0000 | 4.5436 |
| val | door | wall | 3 | 542 | 0.6697 | 0.0683 | 2.2450 |
| val | door | wall | 3 | 521 | 0.8752 | 0.0403 | 3.9992 |
| val | door | wall | 1 | 488 | 1.0000 | 0.0000 | 5.0151 |
| val | door | wall | 3 | 411 | 0.9221 | 0.0000 | 3.7829 |

## Interpretation Gate

- Promising if a variant improves mIoU by >= +0.003, or improves `picture` by >= +0.02 while keeping mIoU within -0.002.
- If no variant passes, targeted latent-subgroup readout is not recovering the oracle/actionability headroom under this offline protocol.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/latent_subgroup_smoke/latent_subgroup_variants.csv`
- Class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/latent_subgroup_smoke/latent_subgroup_class_metrics.csv`
- Cluster diagnostic CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/latent_subgroup_smoke/latent_subgroup_cluster_diagnostic.csv`
- Metadata: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/latent_subgroup_smoke/metadata.json`
