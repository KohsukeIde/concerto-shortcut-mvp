# Decoupled Classifier Readout

Frozen decoder-feature classifier-family pilot for the `concerto_base_origin` decoder probe. This tests whether long-tail / decoupled classifier learning can recover the actionability headroom left by fixed-logit, pair-emphasis, retrieval, and LoRA variants.

## Setup

- Config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Bank points: `13298`
- Seen val batches: `4`
- Train steps: `10`, train batch size `512`, lr `0.001`
- Balanced sampler: `True`

## Headline

- Base: `base_head` mIoU=0.4110 (Δ+0.0000), picture=0.6716 (Δ+0.0000), p->wall=0.0230 (Δ+0.0000)
- Best mIoU: `base_head` mIoU=0.4110 (Δ+0.0000), picture=0.6716 (Δ+0.0000), p->wall=0.0230 (Δ+0.0000)
- Best picture: `base_logit_adjust_alpha0p25` mIoU=0.4064 (Δ-0.0046), picture=0.7307 (Δ+0.0590), p->wall=0.0130 (Δ-0.0100)
- Best safe picture (mIoU >= base - 0.002): `tau0p25_bias` mIoU=0.4091 (Δ-0.0019), picture=0.6533 (Δ-0.0184), p->wall=0.0260 (Δ+0.0030)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `base_head` | 0.4110 | +0.0000 | 0.2964 | +0.0000 | 0.6716 | +0.0000 | 0.0230 | +0.0000 |
| 2 | `crt_balanced_softmax` | 0.4106 | -0.0004 | 0.2885 | -0.0079 | 0.5661 | -0.1055 | 0.0213 | -0.0018 |
| 3 | `tau0p25_bias` | 0.4091 | -0.0019 | 0.2939 | -0.0025 | 0.6533 | -0.0184 | 0.0260 | +0.0030 |
| 4 | `tau0p25_nobias` | 0.4090 | -0.0021 | 0.2958 | -0.0006 | 0.6657 | -0.0059 | 0.0248 | +0.0018 |
| 5 | `crt_ce` | 0.4076 | -0.0034 | 0.2835 | -0.0129 | 0.5346 | -0.1371 | 0.0236 | +0.0006 |
| 6 | `tau0p5_bias` | 0.4075 | -0.0036 | 0.2919 | -0.0044 | 0.6408 | -0.0309 | 0.0301 | +0.0071 |
| 7 | `tau0p5_nobias` | 0.4071 | -0.0039 | 0.2929 | -0.0035 | 0.6470 | -0.0246 | 0.0289 | +0.0059 |
| 8 | `base_logit_adjust_alpha0p25` | 0.4064 | -0.0046 | 0.2986 | +0.0022 | 0.7307 | +0.0590 | 0.0130 | -0.0100 |
| 9 | `tau0p75_bias` | 0.4054 | -0.0056 | 0.2884 | -0.0080 | 0.6171 | -0.0545 | 0.0336 | +0.0106 |
| 10 | `tau0p75_nobias` | 0.4050 | -0.0060 | 0.2899 | -0.0065 | 0.6264 | -0.0452 | 0.0336 | +0.0106 |
| 11 | `tau1_nobias` | 0.4029 | -0.0082 | 0.2861 | -0.0103 | 0.6022 | -0.0695 | 0.0390 | +0.0159 |
| 12 | `tau1_bias` | 0.4028 | -0.0082 | 0.2842 | -0.0122 | 0.5896 | -0.0820 | 0.0384 | +0.0153 |
| 13 | `base_logit_adjust_alpha0p5` | 0.3819 | -0.0292 | 0.2743 | -0.0220 | 0.6514 | -0.0202 | 0.0089 | -0.0142 |
| 14 | `base_logit_adjust_alpha1` | 0.2689 | -0.1421 | 0.1508 | -0.1456 | 0.1137 | -0.5580 | 0.0030 | -0.0201 |

## Interpretation Gate

- Promising if a variant improves mIoU by >= +0.003, or improves `picture` by >= +0.02 while keeping mIoU within -0.002.
- If no variant passes, decoupled classifier learning is not recovering the oracle/actionability headroom under this offline protocol.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/decoupled_classifier_smoke/decoupled_classifier_variants.csv`
- Class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/decoupled_classifier_smoke/decoupled_classifier_class_metrics.csv`
- Metadata: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/decoupled_classifier_smoke/metadata.json`
