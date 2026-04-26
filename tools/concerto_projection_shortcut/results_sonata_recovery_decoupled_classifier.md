# Decoupled Classifier Readout

Frozen decoder-feature classifier-family pilot for the `concerto_base_origin` decoder probe. This tests whether long-tail / decoupled classifier learning can recover the actionability headroom left by fixed-logit, pair-emphasis, retrieval, and LoRA variants.

## Setup

- Config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/sonata/sonata_scannet_linear_merged.pth`
- Bank points: `200000`
- Seen val batches: `312`
- Train steps: `2000`, train batch size `4096`, lr `0.001`
- Balanced sampler: `True`

## Headline

- Base: `base_head` mIoU=0.7090 (Δ+0.0000), picture=0.3580 (Δ+0.0000), p->wall=0.4770 (Δ+0.0000)
- Best mIoU: `tau0p25_bias` mIoU=0.7107 (Δ+0.0017), picture=0.3506 (Δ-0.0073), p->wall=0.5184 (Δ+0.0414)
- Best picture: `tau0p25_nobias` mIoU=0.7105 (Δ+0.0015), picture=0.3579 (Δ-0.0000), p->wall=0.4642 (Δ-0.0129)
- Best safe picture (mIoU >= base - 0.002): `tau0p25_nobias` mIoU=0.7105 (Δ+0.0015), picture=0.3579 (Δ-0.0000), p->wall=0.4642 (Δ-0.0129)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tau0p25_bias` | 0.7107 | +0.0017 | 0.6028 | -0.0012 | 0.3506 | -0.0073 | 0.5184 | +0.0414 |
| 2 | `tau0p25_nobias` | 0.7105 | +0.0015 | 0.6037 | -0.0003 | 0.3579 | -0.0000 | 0.4642 | -0.0129 |
| 3 | `base_head` | 0.7090 | +0.0000 | 0.6040 | +0.0000 | 0.3580 | +0.0000 | 0.4770 | +0.0000 |
| 4 | `tau0p5_nobias` | 0.7075 | -0.0014 | 0.5972 | -0.0067 | 0.3554 | -0.0025 | 0.4832 | +0.0062 |
| 5 | `mix_crt_ce_lam0p05` | 0.7057 | -0.0033 | 0.6006 | -0.0034 | 0.3547 | -0.0032 | 0.4387 | -0.0384 |
| 6 | `mix_crt_balanced_softmax_lam0p05` | 0.7057 | -0.0033 | 0.6006 | -0.0034 | 0.3548 | -0.0032 | 0.4391 | -0.0380 |
| 7 | `tau0p5_bias` | 0.7025 | -0.0065 | 0.5888 | -0.0152 | 0.3204 | -0.0376 | 0.5882 | +0.1112 |
| 8 | `base_logit_adjust_alpha0p25` | 0.7021 | -0.0069 | 0.5963 | -0.0076 | 0.3444 | -0.0136 | 0.4062 | -0.0708 |
| 9 | `mix_crt_balanced_softmax_lam0p1` | 0.6995 | -0.0094 | 0.5930 | -0.0109 | 0.3419 | -0.0161 | 0.4078 | -0.0692 |
| 10 | `mix_crt_ce_lam0p1` | 0.6995 | -0.0095 | 0.5930 | -0.0109 | 0.3417 | -0.0162 | 0.4071 | -0.0699 |
| 11 | `tau0p75_nobias` | 0.6975 | -0.0115 | 0.5830 | -0.0209 | 0.3493 | -0.0087 | 0.5058 | +0.0288 |
| 12 | `base_logit_adjust_alpha0p5` | 0.6866 | -0.0223 | 0.5764 | -0.0275 | 0.3018 | -0.0561 | 0.3524 | -0.1247 |
| 13 | `mix_crt_balanced_softmax_lam0p2` | 0.6799 | -0.0291 | 0.5676 | -0.0364 | 0.2961 | -0.0618 | 0.3553 | -0.1218 |
| 14 | `mix_crt_ce_lam0p2` | 0.6796 | -0.0293 | 0.5674 | -0.0365 | 0.2957 | -0.0622 | 0.3543 | -0.1227 |
| 15 | `tau1_nobias` | 0.6768 | -0.0321 | 0.5567 | -0.0473 | 0.3378 | -0.0201 | 0.5332 | +0.0562 |
| 16 | `tau0p75_bias` | 0.6767 | -0.0322 | 0.5528 | -0.0511 | 0.2370 | -0.1210 | 0.7091 | +0.2321 |
| 17 | `base_logit_adjust_alpha1` | 0.6310 | -0.0780 | 0.5104 | -0.0936 | 0.1920 | -0.1659 | 0.2272 | -0.2499 |
| 18 | `mix_crt_balanced_softmax_lam0p4` | 0.6247 | -0.0843 | 0.4949 | -0.1091 | 0.1988 | -0.1592 | 0.2435 | -0.2335 |
| 19 | `mix_crt_ce_lam0p4` | 0.6237 | -0.0853 | 0.4945 | -0.1094 | 0.1982 | -0.1597 | 0.2418 | -0.2353 |
| 20 | `tau1_bias` | 0.6156 | -0.0933 | 0.4768 | -0.1272 | 0.0952 | -0.2628 | 0.8737 | +0.3966 |
| 21 | `crt_balanced_softmax` | 0.4364 | -0.2725 | 0.3101 | -0.2939 | 0.0636 | -0.2943 | 0.0301 | -0.4470 |
| 22 | `crt_ce` | 0.4334 | -0.2756 | 0.3104 | -0.2935 | 0.0632 | -0.2947 | 0.0273 | -0.4498 |

## Interpretation Gate

- Promising if a variant improves mIoU by >= +0.003, or improves `picture` by >= +0.02 while keeping mIoU within -0.002.
- If no variant passes, decoupled classifier learning is not recovering the oracle/actionability headroom under this offline protocol.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/sonata_recovery_decoupled/decoupled_classifier_variants.csv`
- Class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/sonata_recovery_decoupled/decoupled_classifier_class_metrics.csv`
- Metadata: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/sonata_recovery_decoupled/metadata.json`
