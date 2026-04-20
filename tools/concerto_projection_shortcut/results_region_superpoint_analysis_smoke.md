# Region / Superpoint Diagnostic

Quick diagnostic for whether weak-class failures are coherent at a local region granularity.

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Region voxel sizes: `8`
- Class pairs: `picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall`

## Top Val Variants / Oracles

| rank | variant | mIoU | ΔmIoU | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `region_oracle_s8_top5` | 0.5910 | +0.1819 | 0.9280 | +0.2995 | 0.0059 | -0.0201 |
| 2 | `point_oracle_top5` | 0.5901 | +0.1810 | 0.9451 | +0.3165 | 0.0059 | -0.0201 |
| 3 | `region_majority_oracle_s8` | 0.5383 | +0.1292 | 0.9038 | +0.2752 | 0.0508 | +0.0248 |
| 4 | `point_oracle_top2` | 0.5256 | +0.1165 | 0.8555 | +0.2269 | 0.0112 | -0.0148 |
| 5 | `region_oracle_s8_top2` | 0.5211 | +0.1120 | 0.8754 | +0.2468 | 0.0153 | -0.0106 |
| 6 | `region_logits_s8` | 0.3946 | -0.0145 | 0.6488 | +0.0202 | 0.0396 | +0.0136 |
| 7 | `base` | 0.4091 | +0.0000 | 0.6286 | +0.0000 | 0.0260 | +0.0000 |

## Picture Region Metrics On Val

| region size | target pts | base correct | base->wall | region correct | region->wall | point top2 | region top2 | point top5 | region top5 | picture-majority | wall-majority | mean purity | hard-wall point frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1694 | 0.6623 | 0.0260 | 0.6718 | 0.0396 | 0.8595 | 0.8778 | 0.9451 | 0.9286 | 0.9481 | 0.0508 | 0.9264 | 0.0000 |

## Interpretation Gate

- If region logits/oracles improve `picture` top-K substantially over point logits, region-level readout has direct headroom.
- If `picture` points mostly sit in wall-majority / low-purity regions, point-wise correction is not the only bottleneck; region granularity may be too coarse or object masks are needed.
- If region-level predictions worsen while oracle top-K remains high, region smoothing alone is not the method; region structure is diagnostic only.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/region_superpoint_smoke/region_superpoint_summary.csv`
- Pair metrics CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/region_superpoint_smoke/region_superpoint_pair_metrics.csv`
