# Retrieval / Prototype Readout

Eval-only frozen decoder-feature baselines for the `concerto_base_origin` decoder probe. This tests whether local nonparametric evidence can recover the oracle/actionability headroom that fixed-logit rerankers and pair-emphasis adapters failed to recover.

## Setup

- Config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Bank points: `40000`
- Seen val batches: `312`
- Bank caps: max_train_batches `128`, max_bank_points `40000`, max_per_class `2000`

## Headline

- Base: `base` mIoU=0.7783 (Δ+0.0000), picture=0.4097 (Δ+0.0000), p->wall=0.4359 (Δ+0.0000)
- Best mIoU: `knn_k5_tau0p05_lam0p05` mIoU=0.7785 (Δ+0.0002), picture=0.4099 (Δ+0.0002), p->wall=0.4307 (Δ-0.0052)
- Best picture: `knn_k50_tau0p1_lam0p05` mIoU=0.7785 (Δ+0.0002), picture=0.4100 (Δ+0.0003), p->wall=0.4307 (Δ-0.0052)
- Best safe picture (mIoU >= base - 0.002): `knn_k50_tau0p1_lam0p05` mIoU=0.7785 (Δ+0.0002), picture=0.4100 (Δ+0.0003), p->wall=0.4307 (Δ-0.0052)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `knn_k5_tau0p05_lam0p05` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4307 | -0.0052 |
| 2 | `knn_k5_tau0p1_lam0p05` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4307 | -0.0052 |
| 3 | `knn_k10_tau0p1_lam0p05` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4307 | -0.0052 |
| 4 | `knn_k10_tau0p05_lam0p05` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4307 | -0.0052 |
| 5 | `knn_k50_tau0p1_lam0p05` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4100 | +0.0003 | 0.4307 | -0.0052 |
| 6 | `knn_k20_tau0p1_lam0p05` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4307 | -0.0052 |
| 7 | `knn_k50_tau0p05_lam0p05` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4307 | -0.0052 |
| 8 | `knn_k20_tau0p05_lam0p05` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4307 | -0.0052 |
| 9 | `knn_k5_tau0p05_lam0p1` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4248 | -0.0111 |
| 10 | `knn_k5_tau0p1_lam0p1` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4248 | -0.0111 |
| 11 | `knn_k50_tau0p1_lam0p1` | 0.7785 | +0.0002 | 0.6813 | +0.0005 | 0.4099 | +0.0002 | 0.4248 | -0.0111 |
| 12 | `knn_k50_tau0p05_lam0p1` | 0.7785 | +0.0002 | 0.6813 | +0.0004 | 0.4099 | +0.0002 | 0.4248 | -0.0111 |
| 13 | `knn_k10_tau0p05_lam0p1` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4249 | -0.0111 |
| 14 | `knn_k20_tau0p05_lam0p1` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4099 | +0.0002 | 0.4248 | -0.0111 |
| 15 | `knn_k10_tau0p1_lam0p1` | 0.7785 | +0.0002 | 0.6812 | +0.0004 | 0.4098 | +0.0001 | 0.4249 | -0.0110 |
| 16 | `knn_k20_tau0p1_lam0p1` | 0.7785 | +0.0001 | 0.6812 | +0.0004 | 0.4098 | +0.0001 | 0.4248 | -0.0111 |
| 17 | `knn_adapt_k5_tau0p1_lam0p1` | 0.7784 | +0.0001 | 0.6810 | +0.0002 | 0.4098 | +0.0001 | 0.4330 | -0.0029 |
| 18 | `knn_adapt_k5_tau0p05_lam0p1` | 0.7784 | +0.0001 | 0.6810 | +0.0002 | 0.4098 | +0.0001 | 0.4330 | -0.0029 |
| 19 | `knn_adapt_k50_tau0p1_lam0p2` | 0.7784 | +0.0001 | 0.6810 | +0.0002 | 0.4097 | -0.0000 | 0.4303 | -0.0056 |
| 20 | `knn_adapt_k10_tau0p05_lam0p1` | 0.7784 | +0.0001 | 0.6810 | +0.0002 | 0.4098 | +0.0001 | 0.4330 | -0.0029 |

## Interpretation Gate

- Promising if a variant improves mIoU by >= +0.003, or improves picture by >= +0.02 while keeping mIoU within -0.002.
- If no variant passes that gate, retrieval/prototype readout is treated as no-go under this protocol and the next line should be LP-FT / class-safe LoRA.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/knn_readout_small/retrieval_variants.csv`
- Class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/knn_readout_small/retrieval_class_metrics.csv`
- Metadata: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/knn_readout_small/metadata.json`
