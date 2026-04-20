# Purity-Aware Hybrid Region Decoder Gate

Zero-train gate for PHRD: use label-free region purity proxies to decide when to mix region logits with point logits.

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Region sizes: `4,8`
- Proxies: `pred_agreement,mean_conf,region_conf,mean_top_gap,region_entropy_score`
- Thresholds: `0.6,0.7,0.8,0.9,0.95`
- Alphas: `0.1,0.25,0.5,0.75,1.0`
- Directions: `low`

## Headline

- Base: mIoU=0.7788, picture=0.4076, picture->wall=0.4378
- Best mIoU: `phrd_s8_mean_conf_low_thr0p7_a0p5` mIoU=0.7796 (Δ+0.0008), picture=0.4084 (Δ+0.0008), p->wall=0.4392 (Δ+0.0013)
- Best safe picture: `phrd_s8_mean_top_gap_low_thr0p6_a0p5` mIoU=0.7793 (Δ+0.0005), picture=0.4095 (Δ+0.0019), p->wall=0.4411 (Δ+0.0033)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | p->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `phrd_s8_mean_conf_low_thr0p7_a0p5` | 0.7796 | +0.0008 | 0.6831 | +0.0013 | 0.4084 | +0.0008 | 0.4392 | +0.0013 |
| 2 | `phrd_s8_mean_conf_low_thr0p7_a0p75` | 0.7795 | +0.0007 | 0.6831 | +0.0012 | 0.4081 | +0.0005 | 0.4398 | +0.0020 |
| 3 | `phrd_s8_mean_top_gap_low_thr0p6_a0p25` | 0.7794 | +0.0006 | 0.6828 | +0.0009 | 0.4088 | +0.0013 | 0.4394 | +0.0015 |
| 4 | `phrd_s8_region_entropy_score_low_thr0p8_a0p25` | 0.7794 | +0.0006 | 0.6828 | +0.0009 | 0.4086 | +0.0010 | 0.4389 | +0.0011 |
| 5 | `phrd_s8_region_conf_low_thr0p8_a0p25` | 0.7794 | +0.0006 | 0.6827 | +0.0009 | 0.4084 | +0.0008 | 0.4393 | +0.0015 |
| 6 | `phrd_s4_mean_conf_low_thr0p7_a0p5` | 0.7794 | +0.0005 | 0.6827 | +0.0008 | 0.4083 | +0.0007 | 0.4387 | +0.0009 |
| 7 | `phrd_s8_mean_conf_low_thr0p7_a0p25` | 0.7793 | +0.0005 | 0.6828 | +0.0009 | 0.4081 | +0.0005 | 0.4385 | +0.0007 |
| 8 | `phrd_s8_mean_top_gap_low_thr0p6_a0p5` | 0.7793 | +0.0005 | 0.6826 | +0.0007 | 0.4095 | +0.0019 | 0.4411 | +0.0033 |
| 9 | `phrd_s8_mean_conf_low_thr0p8_a0p25` | 0.7793 | +0.0005 | 0.6826 | +0.0008 | 0.4089 | +0.0013 | 0.4394 | +0.0016 |
| 10 | `phrd_s8_region_conf_low_thr0p7_a0p25` | 0.7793 | +0.0005 | 0.6825 | +0.0007 | 0.4080 | +0.0004 | 0.4384 | +0.0005 |
| 11 | `phrd_s4_mean_conf_low_thr0p7_a0p75` | 0.7793 | +0.0004 | 0.6825 | +0.0006 | 0.4084 | +0.0008 | 0.4390 | +0.0012 |
| 12 | `phrd_s8_mean_conf_low_thr0p6_a0p75` | 0.7793 | +0.0004 | 0.6827 | +0.0008 | 0.4076 | +0.0000 | 0.4383 | +0.0005 |
| 13 | `phrd_s4_mean_conf_low_thr0p6_a0p75` | 0.7792 | +0.0004 | 0.6824 | +0.0006 | 0.4084 | +0.0008 | 0.4379 | +0.0000 |
| 14 | `phrd_s8_mean_conf_low_thr0p6_a1p0` | 0.7792 | +0.0004 | 0.6827 | +0.0008 | 0.4070 | -0.0006 | 0.4382 | +0.0004 |
| 15 | `phrd_s8_region_conf_low_thr0p7_a0p5` | 0.7792 | +0.0004 | 0.6824 | +0.0006 | 0.4083 | +0.0008 | 0.4390 | +0.0012 |
| 16 | `phrd_s8_mean_top_gap_low_thr0p7_a0p25` | 0.7792 | +0.0004 | 0.6823 | +0.0005 | 0.4085 | +0.0009 | 0.4407 | +0.0029 |
| 17 | `phrd_s8_region_conf_low_thr0p9_a0p25` | 0.7792 | +0.0004 | 0.6825 | +0.0006 | 0.4080 | +0.0004 | 0.4407 | +0.0028 |
| 18 | `phrd_s4_mean_conf_low_thr0p7_a0p25` | 0.7792 | +0.0004 | 0.6824 | +0.0006 | 0.4081 | +0.0005 | 0.4382 | +0.0004 |
| 19 | `phrd_s8_region_entropy_score_low_thr0p7_a0p25` | 0.7792 | +0.0004 | 0.6825 | +0.0007 | 0.4080 | +0.0004 | 0.4382 | +0.0003 |
| 20 | `phrd_s8_mean_conf_low_thr0p6_a0p5` | 0.7792 | +0.0004 | 0.6824 | +0.0006 | 0.4076 | +0.0000 | 0.4384 | +0.0006 |

## Proxy Diagnostic

| rank | region | proxy | rho purity | AUC purity>=0.9 | rho picture purity | AUC hard picture-wall | hard rate |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 4 | `region_entropy_score` | 0.2802 | 0.8830 | -0.0696 | 0.9306 | 0.8611 |
| 2 | 4 | `region_conf` | 0.2823 | 0.8857 | -0.0807 | 0.9242 | 0.8611 |
| 3 | 4 | `mean_conf` | 0.2855 | 0.8860 | -0.0793 | 0.9097 | 0.8611 |
| 4 | 4 | `mean_top_gap` | 0.2866 | 0.8873 | -0.0752 | 0.9048 | 0.8611 |
| 5 | 4 | `pred_agreement` | 0.5148 | 0.7812 | 0.0431 | 0.8871 | 0.8611 |
| 6 | 8 | `region_entropy_score` | 0.3816 | 0.8783 | 0.2634 | 0.7091 | 0.4040 |
| 7 | 8 | `region_conf` | 0.3837 | 0.8802 | 0.2693 | 0.7041 | 0.4040 |
| 8 | 8 | `mean_conf` | 0.3881 | 0.8702 | 0.2787 | 0.6987 | 0.4040 |
| 9 | 8 | `mean_top_gap` | 0.3906 | 0.8722 | 0.2847 | 0.6938 | 0.4040 |
| 10 | 8 | `pred_agreement` | 0.6780 | 0.8856 | 0.3686 | 0.5961 | 0.4040 |

## Interpretation Gate

- Strong PHRD go: mIoU >= base +0.003, or picture >= base +0.02 while mIoU >= base -0.002.
- Proxy go: label-free proxy should have useful hard-picture-wall AUC and should not merely select wall-dominated coarse regions.
- If variants are near-tie/no-go, purity-aware coarse region mixing is diagnostic only; next method needs better object proposals or learned masks.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/purity_aware_region_readout_lowgate/purity_aware_region_variants.csv`
- Proxy CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/purity_aware_region_readout_lowgate/purity_proxy_diagnostic.csv`
