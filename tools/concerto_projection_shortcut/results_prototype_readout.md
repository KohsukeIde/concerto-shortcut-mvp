# Retrieval / Prototype Readout

Eval-only frozen decoder-feature baselines for the `concerto_base_origin` decoder probe. This tests whether local nonparametric evidence can recover the oracle/actionability headroom that fixed-logit rerankers and pair-emphasis adapters failed to recover.

## Setup

- Config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Bank points: `200000`
- Seen val batches: `312`
- Bank caps: max_train_batches `256`, max_bank_points `200000`, max_per_class `10000`

## Headline

- Base: `base` mIoU=0.7787 (Δ+0.0000), picture=0.4063 (Δ+0.0000), p->wall=0.4377 (Δ+0.0000)
- Best mIoU: `proto_tau0p1_lam0p2` mIoU=0.7790 (Δ+0.0002), picture=0.4064 (Δ+0.0001), p->wall=0.4173 (Δ-0.0204)
- Best picture: `multiproto4_tau0p2_lam0p4` mIoU=0.7788 (Δ+0.0001), picture=0.4071 (Δ+0.0008), p->wall=0.4206 (Δ-0.0171)
- Best safe picture (mIoU >= base - 0.002): `multiproto4_tau0p2_lam0p4` mIoU=0.7788 (Δ+0.0001), picture=0.4071 (Δ+0.0008), p->wall=0.4206 (Δ-0.0171)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `proto_tau0p1_lam0p2` | 0.7790 | +0.0002 | 0.6800 | +0.0006 | 0.4064 | +0.0001 | 0.4173 | -0.0204 |
| 2 | `proto_tau0p2_lam0p4` | 0.7790 | +0.0002 | 0.6799 | +0.0005 | 0.4062 | -0.0001 | 0.4135 | -0.0242 |
| 3 | `proto_tau0p05_lam0p1` | 0.7789 | +0.0002 | 0.6799 | +0.0005 | 0.4060 | -0.0003 | 0.4268 | -0.0109 |
| 4 | `proto_tau0p1_lam0p1` | 0.7789 | +0.0002 | 0.6798 | +0.0004 | 0.4064 | +0.0001 | 0.4286 | -0.0091 |
| 5 | `proto_tau0p2_lam0p2` | 0.7789 | +0.0002 | 0.6798 | +0.0004 | 0.4065 | +0.0002 | 0.4282 | -0.0095 |
| 6 | `proto_tau0p05_lam0p05` | 0.7789 | +0.0001 | 0.6797 | +0.0003 | 0.4063 | -0.0000 | 0.4325 | -0.0052 |
| 7 | `proto_tau0p1_lam0p05` | 0.7789 | +0.0001 | 0.6797 | +0.0003 | 0.4064 | +0.0001 | 0.4334 | -0.0043 |
| 8 | `multiproto8_tau0p2_lam0p4` | 0.7789 | +0.0001 | 0.6797 | +0.0003 | 0.4069 | +0.0006 | 0.4226 | -0.0151 |
| 9 | `multiproto8_tau0p1_lam0p2` | 0.7789 | +0.0001 | 0.6797 | +0.0003 | 0.4069 | +0.0006 | 0.4218 | -0.0159 |
| 10 | `proto_adapt_tau0p05_lam0p2` | 0.7789 | +0.0001 | 0.6797 | +0.0003 | 0.4057 | -0.0006 | 0.4321 | -0.0056 |
| 11 | `proto_adapt_tau0p2_lam0p4` | 0.7789 | +0.0001 | 0.6796 | +0.0002 | 0.4062 | -0.0002 | 0.4332 | -0.0045 |
| 12 | `proto_adapt_tau0p1_lam0p4` | 0.7789 | +0.0001 | 0.6797 | +0.0003 | 0.4055 | -0.0008 | 0.4280 | -0.0097 |
| 13 | `proto_adapt_tau0p1_lam0p2` | 0.7789 | +0.0001 | 0.6796 | +0.0002 | 0.4061 | -0.0002 | 0.4331 | -0.0046 |
| 14 | `proto_tau0p2_lam0p1` | 0.7789 | +0.0001 | 0.6796 | +0.0002 | 0.4064 | +0.0001 | 0.4334 | -0.0043 |
| 15 | `multiproto4_tau0p2_lam0p2` | 0.7788 | +0.0001 | 0.6796 | +0.0002 | 0.4066 | +0.0003 | 0.4312 | -0.0065 |
| 16 | `multiproto4_tau0p2_lam0p4` | 0.7788 | +0.0001 | 0.6796 | +0.0002 | 0.4071 | +0.0008 | 0.4206 | -0.0171 |
| 17 | `multiproto8_tau0p05_lam0p1` | 0.7788 | +0.0001 | 0.6796 | +0.0002 | 0.4065 | +0.0001 | 0.4273 | -0.0104 |
| 18 | `multiproto8_tau0p2_lam0p2` | 0.7788 | +0.0001 | 0.6796 | +0.0002 | 0.4065 | +0.0002 | 0.4322 | -0.0055 |
| 19 | `proto_adapt_tau0p05_lam0p1` | 0.7788 | +0.0001 | 0.6796 | +0.0002 | 0.4059 | -0.0004 | 0.4351 | -0.0026 |
| 20 | `multiproto4_tau0p1_lam0p05` | 0.7788 | +0.0001 | 0.6795 | +0.0001 | 0.4065 | +0.0002 | 0.4340 | -0.0037 |

## Interpretation Gate

- Promising if a variant improves mIoU by >= +0.003, or improves picture by >= +0.02 while keeping mIoU within -0.002.
- If no variant passes that gate, retrieval/prototype readout is treated as no-go under this protocol and the next line should be LP-FT / class-safe LoRA.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/prototype_readout/retrieval_variants.csv`
- Class CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/prototype_readout/retrieval_class_metrics.csv`
- Metadata: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/prototype_readout/metadata.json`
