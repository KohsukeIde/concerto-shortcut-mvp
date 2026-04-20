# Purity-Aware Hybrid Region Decoder Gate

Zero-train gate for PHRD: use label-free region purity proxies to decide when to mix region logits with point logits.

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Region sizes: `4,8`
- Proxies: `pred_agreement,mean_conf,region_conf,mean_top_gap,region_entropy_score`
- Thresholds: `0.6,0.7,0.8,0.9,0.95`
- Alphas: `0.1,0.25,0.5,0.75,1.0`

## Headline

- Base: mIoU=0.4091, picture=0.6286, picture->wall=0.0260
- Best mIoU: `phrd_s8_mean_conf_thr0p6_a0p25` mIoU=0.4107 (Δ+0.0016), picture=0.6644 (Δ+0.0358), p->wall=0.0230 (Δ-0.0030)
- Best safe picture: `phrd_s8_mean_conf_thr0p6_a0p5` mIoU=0.4086 (Δ-0.0005), picture=0.6718 (Δ+0.0432), p->wall=0.0283 (Δ+0.0024)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | p->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `phrd_s8_mean_conf_thr0p6_a0p25` | 0.4107 | +0.0016 | 0.2934 | +0.0037 | 0.6644 | +0.0358 | 0.0230 | -0.0030 |
| 2 | `phrd_s8_region_entropy_score_thr0p6_a0p25` | 0.4103 | +0.0012 | 0.2923 | +0.0026 | 0.6559 | +0.0273 | 0.0230 | -0.0030 |
| 3 | `phrd_s8_pred_agreement_thr0p6_a0p25` | 0.4101 | +0.0010 | 0.2925 | +0.0028 | 0.6578 | +0.0292 | 0.0242 | -0.0018 |
| 4 | `phrd_s8_region_conf_thr0p6_a0p25` | 0.4101 | +0.0010 | 0.2923 | +0.0026 | 0.6565 | +0.0279 | 0.0230 | -0.0030 |
| 5 | `phrd_s8_pred_agreement_thr0p8_a0p25` | 0.4100 | +0.0009 | 0.2919 | +0.0022 | 0.6495 | +0.0209 | 0.0236 | -0.0024 |
| 6 | `phrd_s8_pred_agreement_thr0p7_a0p25` | 0.4099 | +0.0009 | 0.2921 | +0.0024 | 0.6530 | +0.0244 | 0.0236 | -0.0024 |
| 7 | `phrd_s8_mean_conf_thr0p6_a0p1` | 0.4099 | +0.0008 | 0.2915 | +0.0017 | 0.6447 | +0.0161 | 0.0248 | -0.0012 |
| 8 | `phrd_s4_pred_agreement_thr0p6_a0p5` | 0.4098 | +0.0008 | 0.2928 | +0.0030 | 0.6542 | +0.0256 | 0.0230 | -0.0030 |
| 9 | `phrd_s8_mean_conf_thr0p7_a0p25` | 0.4098 | +0.0008 | 0.2912 | +0.0015 | 0.6476 | +0.0191 | 0.0230 | -0.0030 |
| 10 | `phrd_s8_region_entropy_score_thr0p7_a0p25` | 0.4098 | +0.0007 | 0.2913 | +0.0016 | 0.6499 | +0.0213 | 0.0230 | -0.0030 |
| 11 | `phrd_s8_region_entropy_score_thr0p6_a0p1` | 0.4098 | +0.0007 | 0.2912 | +0.0015 | 0.6430 | +0.0144 | 0.0248 | -0.0012 |
| 12 | `phrd_s8_region_conf_thr0p7_a0p25` | 0.4098 | +0.0007 | 0.2917 | +0.0019 | 0.6508 | +0.0223 | 0.0230 | -0.0030 |
| 13 | `phrd_s4_pred_agreement_thr0p6_a0p25` | 0.4097 | +0.0007 | 0.2916 | +0.0019 | 0.6445 | +0.0159 | 0.0242 | -0.0018 |
| 14 | `phrd_s8_region_conf_thr0p8_a0p25` | 0.4097 | +0.0006 | 0.2903 | +0.0005 | 0.6408 | +0.0123 | 0.0230 | -0.0030 |
| 15 | `phrd_s8_pred_agreement_thr0p6_a0p1` | 0.4097 | +0.0006 | 0.2913 | +0.0015 | 0.6432 | +0.0146 | 0.0248 | -0.0012 |
| 16 | `phrd_s8_region_conf_thr0p6_a0p1` | 0.4096 | +0.0006 | 0.2911 | +0.0014 | 0.6424 | +0.0138 | 0.0248 | -0.0012 |
| 17 | `phrd_s8_pred_agreement_thr0p8_a0p5` | 0.4096 | +0.0006 | 0.2927 | +0.0030 | 0.6560 | +0.0275 | 0.0236 | -0.0024 |
| 18 | `phrd_s8_region_entropy_score_thr0p8_a0p25` | 0.4096 | +0.0005 | 0.2900 | +0.0003 | 0.6392 | +0.0106 | 0.0230 | -0.0030 |
| 19 | `phrd_s8_region_conf_thr0p8_a0p1` | 0.4096 | +0.0005 | 0.2906 | +0.0008 | 0.6375 | +0.0090 | 0.0242 | -0.0018 |
| 20 | `phrd_s8_region_entropy_score_thr0p7_a0p1` | 0.4096 | +0.0005 | 0.2908 | +0.0011 | 0.6404 | +0.0118 | 0.0248 | -0.0012 |

## Proxy Diagnostic

| rank | region | proxy | rho purity | AUC purity>=0.9 | rho picture purity | AUC hard picture-wall | hard rate |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 4 | `mean_conf` | 0.2675 | 0.8756 | nan | nan | nan |
| 2 | 4 | `mean_top_gap` | 0.2691 | 0.8774 | nan | nan | nan |
| 3 | 4 | `pred_agreement` | 0.4732 | 0.7737 | nan | nan | nan |
| 4 | 4 | `region_conf` | 0.2681 | 0.8788 | nan | nan | nan |
| 5 | 4 | `region_entropy_score` | 0.2652 | 0.8749 | nan | nan | nan |
| 6 | 8 | `mean_conf` | 0.3685 | 0.8612 | nan | nan | 0.0000 |
| 7 | 8 | `mean_top_gap` | 0.3713 | 0.8635 | nan | nan | 0.0000 |
| 8 | 8 | `pred_agreement` | 0.6357 | 0.8805 | nan | nan | 0.0000 |
| 9 | 8 | `region_conf` | 0.3698 | 0.8703 | nan | nan | 0.0000 |
| 10 | 8 | `region_entropy_score` | 0.3669 | 0.8672 | nan | nan | 0.0000 |

## Interpretation Gate

- Strong PHRD go: mIoU >= base +0.003, or picture >= base +0.02 while mIoU >= base -0.002.
- Proxy go: label-free proxy should have useful hard-picture-wall AUC and should not merely select wall-dominated coarse regions.
- If variants are near-tie/no-go, purity-aware coarse region mixing is diagnostic only; next method needs better object proposals or learned masks.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/purity_aware_region_smoke/purity_aware_region_variants.csv`
- Proxy CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/purity_aware_region_smoke/purity_proxy_diagnostic.csv`
