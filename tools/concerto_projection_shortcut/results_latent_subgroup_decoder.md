# Latent Subgroup Decoder

Latent subgroup diagnostic and targeted sub-center readout pilot for the `concerto_base_origin` decoder-probe checkpoint.

## Setup

- Config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Train points: `1200000`
- Heldout points: `968943`
- Seen val batches: `312`

## Headline

- Base: `base` mIoU=0.7777 (Δ+0.0000), picture=0.4043 (Δ+0.0000), p->wall=0.4348 (Δ+0.0000)
- Best mIoU: `subcenter_tau0p1_lam0p4` mIoU=0.7788 (Δ+0.0011), picture=0.4060 (Δ+0.0017), p->wall=0.4096 (Δ-0.0252)
- Best picture: `subcenter_tau0p1_lam0p4` mIoU=0.7788 (Δ+0.0011), picture=0.4060 (Δ+0.0017), p->wall=0.4096 (Δ-0.0252)
- Best safe picture (mIoU >= base - 0.002): `subcenter_tau0p1_lam0p4` mIoU=0.7788 (Δ+0.0011), picture=0.4060 (Δ+0.0017), p->wall=0.4096 (Δ-0.0252)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `subcenter_tau0p1_lam0p4` | 0.7788 | +0.0011 | 0.6809 | +0.0017 | 0.4060 | +0.0017 | 0.4096 | -0.0252 |
| 2 | `subcenter_tau0p1_lam0p2` | 0.7785 | +0.0009 | 0.6805 | +0.0013 | 0.4055 | +0.0011 | 0.4239 | -0.0109 |
| 3 | `subcenter_tau0p05_lam0p2` | 0.7785 | +0.0008 | 0.6806 | +0.0014 | 0.4052 | +0.0009 | 0.4142 | -0.0206 |
| 4 | `subcenter_tau0p2_lam0p4` | 0.7784 | +0.0008 | 0.6801 | +0.0009 | 0.4050 | +0.0006 | 0.4312 | -0.0036 |
| 5 | `subcenter_adapt_tau0p1_lam0p4` | 0.7783 | +0.0006 | 0.6799 | +0.0007 | 0.4044 | +0.0001 | 0.4298 | -0.0050 |
| 6 | `subcenter_adapt_tau0p05_lam0p4` | 0.7782 | +0.0006 | 0.6799 | +0.0007 | 0.4034 | -0.0010 | 0.4249 | -0.0098 |
| 7 | `subcenter_tau0p05_lam0p1` | 0.7782 | +0.0006 | 0.6802 | +0.0009 | 0.4050 | +0.0006 | 0.4254 | -0.0094 |
| 8 | `subcenter_tau0p2_lam0p2` | 0.7781 | +0.0005 | 0.6798 | +0.0005 | 0.4046 | +0.0003 | 0.4334 | -0.0014 |
| 9 | `subcenter_tau0p1_lam0p1` | 0.7781 | +0.0005 | 0.6799 | +0.0007 | 0.4049 | +0.0005 | 0.4299 | -0.0049 |
| 10 | `subcenter_adapt_tau0p2_lam0p4` | 0.7780 | +0.0004 | 0.6795 | +0.0003 | 0.4042 | -0.0001 | 0.4343 | -0.0005 |
| 11 | `subcenter_adapt_tau0p05_lam0p2` | 0.7780 | +0.0004 | 0.6797 | +0.0005 | 0.4043 | -0.0000 | 0.4302 | -0.0046 |
| 12 | `subcenter_adapt_tau0p1_lam0p2` | 0.7780 | +0.0003 | 0.6796 | +0.0004 | 0.4044 | +0.0000 | 0.4324 | -0.0024 |
| 13 | `subcenter_tau0p05_lam0p05` | 0.7779 | +0.0003 | 0.6797 | +0.0005 | 0.4048 | +0.0004 | 0.4304 | -0.0043 |
| 14 | `subcenter_tau0p2_lam0p1` | 0.7779 | +0.0002 | 0.6795 | +0.0002 | 0.4044 | +0.0001 | 0.4341 | -0.0007 |
| 15 | `subcenter_tau0p1_lam0p05` | 0.7779 | +0.0002 | 0.6796 | +0.0003 | 0.4046 | +0.0002 | 0.4325 | -0.0023 |
| 16 | `subcenter_adapt_tau0p2_lam0p2` | 0.7779 | +0.0002 | 0.6794 | +0.0001 | 0.4043 | -0.0000 | 0.4345 | -0.0003 |
| 17 | `subcenter_adapt_tau0p05_lam0p1` | 0.7778 | +0.0002 | 0.6795 | +0.0003 | 0.4043 | -0.0000 | 0.4326 | -0.0022 |
| 18 | `subcenter_adapt_tau0p1_lam0p1` | 0.7778 | +0.0002 | 0.6794 | +0.0002 | 0.4043 | -0.0000 | 0.4336 | -0.0012 |
| 19 | `subcenter_tau0p2_lam0p05` | 0.7778 | +0.0001 | 0.6793 | +0.0001 | 0.4044 | +0.0000 | 0.4345 | -0.0003 |
| 20 | `subcenter_adapt_tau0p2_lam0p1` | 0.7778 | +0.0001 | 0.6793 | +0.0001 | 0.4043 | +0.0000 | 0.4346 | -0.0002 |

## Val Cluster Diagnostic

| domain | class | counterpart | cluster | count | target top1 | counterpart top1 | target-counterpart margin |
|---|---|---|---:|---:|---:|---:|---:|
| val | picture | wall | 3 | 23151 | 0.0606 | 0.9360 | -5.9442 |
| val | picture | wall | 3 | 10992 | 0.0129 | 0.9817 | -6.5944 |
| val | picture | wall | 3 | 9208 | 0.3020 | 0.6975 | -1.2218 |
| val | picture | wall | 3 | 8958 | 0.0000 | 1.0000 | -7.5938 |
| val | picture | wall | 2 | 7253 | 0.9695 | 0.0019 | 4.2052 |
| val | picture | wall | 3 | 6212 | 0.1648 | 0.8352 | -3.4074 |
| val | picture | wall | 3 | 5480 | 0.0000 | 0.9469 | -7.9454 |
| val | picture | wall | 3 | 5435 | 0.1489 | 0.8511 | -3.4504 |
| val | picture | wall | 3 | 4077 | 0.2737 | 0.7263 | -3.1685 |
| val | picture | wall | 3 | 3906 | 0.4519 | 0.5481 | -0.2462 |
| val | picture | wall | 3 | 3187 | 0.5089 | 0.4911 | -0.4691 |
| val | picture | wall | 0 | 2714 | 0.9996 | 0.0000 | 4.7257 |
| val | picture | wall | 3 | 2510 | 0.2530 | 0.7470 | -1.2650 |
| val | picture | wall | 2 | 2270 | 1.0000 | 0.0000 | 4.3247 |
| val | picture | wall | 3 | 1999 | 0.6848 | 0.3087 | 0.4071 |
| val | picture | wall | 3 | 1963 | 0.7392 | 0.2608 | 0.7633 |

## Interpretation Gate

- Promising if a variant improves mIoU by >= +0.003, or improves `picture` by >= +0.02 while keeping mIoU within -0.002.
- If no variant passes, targeted latent-subgroup readout is not recovering the oracle/actionability headroom under this offline protocol.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/latent_subgroup_decoder/latent_subgroup_variants.csv`
- Class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/latent_subgroup_decoder/latent_subgroup_class_metrics.csv`
- Cluster diagnostic CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/latent_subgroup_decoder/latent_subgroup_cluster_diagnostic.csv`
- Metadata: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/latent_subgroup_decoder/metadata.json`
