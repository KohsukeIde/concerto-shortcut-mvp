# Region / Superpoint Diagnostic

Quick diagnostic for whether weak-class failures are coherent at a local region granularity.

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Region voxel sizes: `4,8,16,32`
- Class pairs: `picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall`

## Top Val Variants / Oracles

| rank | variant | mIoU | ΔmIoU | picture | Δpicture | picture->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `region_oracle_s4_top5` | 0.9783 | +0.1995 | 0.9462 | +0.5386 | 0.0284 | -0.4093 |
| 2 | `region_oracle_s8_top5` | 0.9780 | +0.1992 | 0.9456 | +0.5379 | 0.0291 | -0.4086 |
| 3 | `point_oracle_top5` | 0.9777 | +0.1989 | 0.9448 | +0.5372 | 0.0292 | -0.4086 |
| 4 | `region_oracle_s16_top5` | 0.9770 | +0.1982 | 0.9423 | +0.5347 | 0.0339 | -0.4038 |
| 5 | `region_oracle_s32_top5` | 0.9747 | +0.1959 | 0.9074 | +0.4998 | 0.0694 | -0.3684 |
| 6 | `region_majority_oracle_s4` | 0.9487 | +0.1699 | 0.8841 | +0.4765 | 0.0617 | -0.3761 |
| 7 | `region_oracle_s4_top2` | 0.9218 | +0.1430 | 0.8674 | +0.4598 | 0.0828 | -0.3549 |
| 8 | `region_oracle_s8_top2` | 0.9210 | +0.1422 | 0.8677 | +0.4601 | 0.0866 | -0.3512 |
| 9 | `point_oracle_top2` | 0.9204 | +0.1416 | 0.8599 | +0.4523 | 0.0881 | -0.3497 |
| 10 | `region_oracle_s16_top2` | 0.9200 | +0.1412 | 0.8671 | +0.4595 | 0.0921 | -0.3456 |
| 11 | `region_oracle_s32_top2` | 0.9116 | +0.1328 | 0.8225 | +0.4149 | 0.1405 | -0.2973 |
| 12 | `region_majority_oracle_s8` | 0.8992 | +0.1204 | 0.8011 | +0.3935 | 0.1102 | -0.3276 |
| 13 | `region_majority_oracle_s16` | 0.7925 | +0.0137 | 0.6666 | +0.2589 | 0.2020 | -0.2358 |
| 14 | `region_logits_s4` | 0.7695 | -0.0093 | 0.3974 | -0.0102 | 0.4506 | +0.0128 |
| 15 | `region_logits_s8` | 0.7447 | -0.0341 | 0.3747 | -0.0329 | 0.4799 | +0.0421 |
| 16 | `region_logits_s16` | 0.6708 | -0.1080 | 0.3137 | -0.0939 | 0.5652 | +0.1274 |
| 17 | `region_majority_oracle_s32` | 0.6082 | -0.1706 | 0.4696 | +0.0620 | 0.3556 | -0.0822 |
| 18 | `region_logits_s32` | 0.5200 | -0.2588 | 0.1796 | -0.2280 | 0.7341 | +0.2963 |

## Picture Region Metrics On Val

| region size | target pts | base correct | base->wall | region correct | region->wall | point top2 | region top2 | point top5 | region top5 | picture-majority | wall-majority | mean purity | hard-wall point frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 207442 | 0.5394 | 0.4378 | 0.5262 | 0.4506 | 0.8954 | 0.9009 | 0.9615 | 0.9624 | 0.9348 | 0.0617 | 0.9182 | 0.0251 |
| 8 | 207442 | 0.5394 | 0.4378 | 0.4956 | 0.4799 | 0.8954 | 0.8961 | 0.9615 | 0.9606 | 0.8808 | 0.1102 | 0.8533 | 0.1815 |
| 16 | 207442 | 0.5394 | 0.4378 | 0.4007 | 0.5652 | 0.8954 | 0.8803 | 0.9615 | 0.9533 | 0.7739 | 0.2020 | 0.7418 | 0.3223 |
| 32 | 207442 | 0.5394 | 0.4378 | 0.2148 | 0.7341 | 0.8954 | 0.7889 | 0.9615 | 0.9063 | 0.5813 | 0.3556 | 0.5614 | 0.3153 |

## Interpretation Gate

- If region logits/oracles improve `picture` top-K substantially over point logits, region-level readout has direct headroom.
- If `picture` points mostly sit in wall-majority / low-purity regions, point-wise correction is not the only bottleneck; region granularity may be too coarse or object masks are needed.
- If region-level predictions worsen while oracle top-K remains high, region smoothing alone is not the method; region structure is diagnostic only.

## Files

- Summary CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/region_superpoint_analysis/region_superpoint_summary.csv`
- Pair metrics CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/region_superpoint_analysis/region_superpoint_pair_metrics.csv`
