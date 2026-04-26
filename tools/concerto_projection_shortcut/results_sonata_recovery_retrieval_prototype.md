# Retrieval / Prototype Readout

Eval-only frozen decoder-feature baselines for the `concerto_base_origin` decoder probe. This tests whether local nonparametric evidence can recover the oracle/actionability headroom that fixed-logit rerankers and pair-emphasis adapters failed to recover.

## Setup

- Config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/sonata/sonata_scannet_linear_merged.pth`
- Bank points: `200000`
- Seen val batches: `312`
- Bank caps: max_train_batches `256`, max_bank_points `200000`, max_per_class `10000`

## Headline

- Base: `base` mIoU=0.7095 (Δ+0.0000), picture=0.3619 (Δ+0.0000), p->wall=0.4759 (Δ+0.0000)
- Best mIoU: `base` mIoU=0.7095 (Δ+0.0000), picture=0.3619 (Δ+0.0000), p->wall=0.4759 (Δ+0.0000)
- Best picture: `proto_tau0p1_lam0p05` mIoU=0.7094 (Δ-0.0001), picture=0.3620 (Δ+0.0001), p->wall=0.4735 (Δ-0.0025)
- Best safe picture (mIoU >= base - 0.002): `proto_tau0p1_lam0p05` mIoU=0.7094 (Δ-0.0001), picture=0.3620 (Δ+0.0001), p->wall=0.4735 (Δ-0.0025)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `base` | 0.7095 | +0.0000 | 0.6049 | +0.0000 | 0.3619 | +0.0000 | 0.4759 | +0.0000 |
| 2 | `multiproto8_adapt_tau0p2_lam0p05` | 0.7095 | -0.0000 | 0.6049 | +0.0000 | 0.3619 | +0.0000 | 0.4756 | -0.0004 |
| 3 | `multiproto4_adapt_tau0p2_lam0p05` | 0.7095 | -0.0000 | 0.6049 | +0.0000 | 0.3619 | +0.0000 | 0.4756 | -0.0004 |
| 4 | `proto_adapt_tau0p2_lam0p05` | 0.7095 | -0.0000 | 0.6049 | -0.0000 | 0.3619 | +0.0000 | 0.4757 | -0.0003 |
| 5 | `proto_adapt_tau0p2_lam0p1` | 0.7095 | -0.0000 | 0.6049 | -0.0000 | 0.3619 | -0.0000 | 0.4755 | -0.0004 |
| 6 | `proto_tau0p2_lam0p05` | 0.7095 | -0.0000 | 0.6049 | -0.0000 | 0.3619 | -0.0000 | 0.4752 | -0.0007 |
| 7 | `multiproto8_adapt_tau0p2_lam0p1` | 0.7095 | -0.0000 | 0.6049 | -0.0000 | 0.3618 | -0.0000 | 0.4753 | -0.0006 |
| 8 | `multiproto8_tau0p2_lam0p05` | 0.7095 | -0.0000 | 0.6049 | -0.0000 | 0.3619 | +0.0000 | 0.4748 | -0.0012 |
| 9 | `proto_adapt_tau0p1_lam0p05` | 0.7095 | -0.0000 | 0.6049 | -0.0000 | 0.3619 | -0.0000 | 0.4753 | -0.0006 |
| 10 | `multiproto4_adapt_tau0p2_lam0p1` | 0.7095 | -0.0000 | 0.6049 | -0.0000 | 0.3618 | -0.0000 | 0.4753 | -0.0006 |
| 11 | `multiproto8_adapt_tau0p1_lam0p05` | 0.7095 | -0.0000 | 0.6049 | -0.0000 | 0.3619 | -0.0000 | 0.4749 | -0.0010 |
| 12 | `multiproto4_tau0p2_lam0p05` | 0.7095 | -0.0001 | 0.6049 | -0.0000 | 0.3619 | +0.0000 | 0.4747 | -0.0012 |
| 13 | `multiproto4_adapt_tau0p1_lam0p05` | 0.7094 | -0.0001 | 0.6049 | -0.0000 | 0.3619 | +0.0000 | 0.4749 | -0.0010 |
| 14 | `proto_adapt_tau0p2_lam0p2` | 0.7094 | -0.0001 | 0.6049 | -0.0001 | 0.3619 | -0.0000 | 0.4751 | -0.0009 |
| 15 | `multiproto8_adapt_tau0p2_lam0p2` | 0.7094 | -0.0001 | 0.6049 | -0.0001 | 0.3618 | -0.0001 | 0.4746 | -0.0013 |
| 16 | `proto_tau0p2_lam0p1` | 0.7094 | -0.0001 | 0.6049 | -0.0001 | 0.3619 | +0.0000 | 0.4743 | -0.0016 |
| 17 | `multiproto8_tau0p2_lam0p1` | 0.7094 | -0.0001 | 0.6049 | -0.0000 | 0.3619 | +0.0000 | 0.4735 | -0.0024 |
| 18 | `multiproto4_adapt_tau0p2_lam0p2` | 0.7094 | -0.0001 | 0.6048 | -0.0001 | 0.3618 | -0.0000 | 0.4746 | -0.0014 |
| 19 | `multiproto8_adapt_tau0p1_lam0p1` | 0.7094 | -0.0001 | 0.6049 | -0.0001 | 0.3618 | -0.0001 | 0.4740 | -0.0020 |
| 20 | `proto_adapt_tau0p1_lam0p1` | 0.7094 | -0.0001 | 0.6048 | -0.0001 | 0.3619 | -0.0000 | 0.4746 | -0.0014 |

## Interpretation Gate

- Promising if a variant improves mIoU by >= +0.003, or improves picture by >= +0.02 while keeping mIoU within -0.002.
- If no variant passes that gate, retrieval/prototype readout is treated as no-go under this protocol and the next line should be LP-FT / class-safe LoRA.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/sonata_recovery_retrieval/retrieval_variants.csv`
- Class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/sonata_recovery_retrieval/retrieval_class_metrics.csv`
- Metadata: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/sonata_recovery_retrieval/metadata.json`
