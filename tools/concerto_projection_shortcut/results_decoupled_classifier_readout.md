# Decoupled Classifier Readout

Frozen decoder-feature classifier-family pilot for the `concerto_base_origin` decoder probe. This tests whether long-tail / decoupled classifier learning can recover the actionability headroom left by fixed-logit, pair-emphasis, retrieval, and LoRA variants.

## Setup

- Config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Bank points: `200000`
- Seen val batches: `312`
- Train steps: `2000`, train batch size `4096`, lr `0.001`
- Balanced sampler: `True`

## Headline

- Base: `base_head` mIoU=0.7776 (Δ+0.0000), picture=0.4051 (Δ+0.0000), p->wall=0.4407 (Δ+0.0000)
- Best mIoU: `tau0p25_nobias` mIoU=0.7778 (Δ+0.0002), picture=0.4025 (Δ-0.0025), p->wall=0.4516 (Δ+0.0108)
- Best picture: `mix_crt_balanced_softmax_lam0p05` mIoU=0.7777 (Δ+0.0001), picture=0.4045 (Δ-0.0006), p->wall=0.4146 (Δ-0.0261)
- Best safe picture (mIoU >= base - 0.002): `mix_crt_balanced_softmax_lam0p05` mIoU=0.7777 (Δ+0.0001), picture=0.4045 (Δ-0.0006), p->wall=0.4146 (Δ-0.0261)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tau0p25_nobias` | 0.7778 | +0.0002 | 0.6792 | -0.0004 | 0.4025 | -0.0025 | 0.4516 | +0.0108 |
| 2 | `tau0p25_bias` | 0.7778 | +0.0001 | 0.6791 | -0.0005 | 0.4023 | -0.0028 | 0.4531 | +0.0124 |
| 3 | `mix_crt_balanced_softmax_lam0p05` | 0.7777 | +0.0001 | 0.6796 | -0.0000 | 0.4045 | -0.0006 | 0.4146 | -0.0261 |
| 4 | `tau0p5_bias` | 0.7777 | +0.0000 | 0.6784 | -0.0012 | 0.3989 | -0.0062 | 0.4659 | +0.0252 |
| 5 | `mix_crt_ce_lam0p05` | 0.7777 | +0.0000 | 0.6795 | -0.0001 | 0.4044 | -0.0007 | 0.4155 | -0.0252 |
| 6 | `tau0p5_nobias` | 0.7777 | +0.0000 | 0.6784 | -0.0012 | 0.3994 | -0.0056 | 0.4641 | +0.0234 |
| 7 | `base_head` | 0.7776 | +0.0000 | 0.6796 | +0.0000 | 0.4051 | +0.0000 | 0.4407 | +0.0000 |
| 8 | `tau0p75_bias` | 0.7772 | -0.0004 | 0.6769 | -0.0027 | 0.3953 | -0.0098 | 0.4787 | +0.0380 |
| 9 | `tau0p75_nobias` | 0.7771 | -0.0005 | 0.6768 | -0.0028 | 0.3956 | -0.0094 | 0.4772 | +0.0365 |
| 10 | `tau1_bias` | 0.7766 | -0.0011 | 0.6752 | -0.0044 | 0.3908 | -0.0142 | 0.4918 | +0.0510 |
| 11 | `tau1_nobias` | 0.7764 | -0.0012 | 0.6750 | -0.0046 | 0.3913 | -0.0138 | 0.4901 | +0.0494 |
| 12 | `mix_crt_ce_lam0p1` | 0.7763 | -0.0013 | 0.6778 | -0.0018 | 0.3985 | -0.0066 | 0.3923 | -0.0484 |
| 13 | `mix_crt_balanced_softmax_lam0p1` | 0.7763 | -0.0013 | 0.6778 | -0.0018 | 0.3979 | -0.0071 | 0.3905 | -0.0502 |
| 14 | `base_logit_adjust_alpha0p25` | 0.7752 | -0.0024 | 0.6779 | -0.0017 | 0.4035 | -0.0016 | 0.3891 | -0.0516 |
| 15 | `mix_crt_ce_lam0p2` | 0.7707 | -0.0069 | 0.6700 | -0.0096 | 0.3692 | -0.0358 | 0.3527 | -0.0881 |
| 16 | `mix_crt_balanced_softmax_lam0p2` | 0.7706 | -0.0070 | 0.6700 | -0.0096 | 0.3671 | -0.0379 | 0.3500 | -0.0907 |
| 17 | `base_logit_adjust_alpha0p5` | 0.7678 | -0.0098 | 0.6676 | -0.0120 | 0.3777 | -0.0274 | 0.3440 | -0.0967 |
| 18 | `mix_crt_ce_lam0p4` | 0.7516 | -0.0260 | 0.6436 | -0.0360 | 0.2704 | -0.1346 | 0.3030 | -0.1377 |
| 19 | `mix_crt_balanced_softmax_lam0p4` | 0.7513 | -0.0264 | 0.6430 | -0.0366 | 0.2652 | -0.1398 | 0.2999 | -0.1408 |
| 20 | `base_logit_adjust_alpha1` | 0.7373 | -0.0403 | 0.6191 | -0.0605 | 0.2490 | -0.1561 | 0.2756 | -0.1651 |
| 21 | `crt_ce` | 0.6670 | -0.1107 | 0.5425 | -0.1371 | 0.1055 | -0.2996 | 0.1905 | -0.2503 |
| 22 | `crt_balanced_softmax` | 0.6657 | -0.1119 | 0.5407 | -0.1389 | 0.1034 | -0.3017 | 0.1822 | -0.2586 |

## Interpretation Gate

- Promising if a variant improves mIoU by >= +0.003, or improves `picture` by >= +0.02 while keeping mIoU within -0.002.
- If no variant passes, decoupled classifier learning is not recovering the oracle/actionability headroom under this offline protocol.

## Decision

- No variant passes the gate.
- `tau` normalization gives at most a tiny mIoU movement (`+0.0002`) and hurts
  `picture` / `picture -> wall`.
- Direct cRT and Balanced Softmax classifier replacement severely overcorrect
  the multiclass prior (`mIoU` around `0.666-0.667`, `picture` around
  `0.103-0.105`).
- Small trust-region mixing with cRT / Balanced Softmax reduces
  `picture -> wall` (`-0.026` at `lambda=0.05`) but does not improve
  `picture` IoU or weak-class mean.
- Treat offline decoupled classifier retraining as no-go under this protocol.
  The raw-prior correction is included in this result; the training bank is
  balanced for cRT, while class-prior terms use raw train counts from the first
  256 ScanNet train batches.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/decoupled_classifier_readout_prior/decoupled_classifier_variants.csv`
- Class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/decoupled_classifier_readout_prior/decoupled_classifier_class_metrics.csv`
- Metadata: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/decoupled_classifier_readout_prior/metadata.json`
