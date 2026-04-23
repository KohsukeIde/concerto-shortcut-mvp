# Utonia Oracle Actionability Analysis

## Setup
- utonia weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia.pth`
- seg head weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia_linear_prob_head_sc.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- weak classes: `picture,counter,door`
- class pairs: `picture:wall,door:wall,counter:cabinet`
- train batches seen: 14
- val batches seen: 64

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9992 | +0.2418 | 1.0000 | +0.7048 |
| oracle_graph_top5 | 0.9911 | +0.2337 | 1.0000 | +0.7048 |
| oracle_top5 | 0.9908 | +0.2333 | 1.0000 | +0.7048 |
| oracle_graph_top3 | 0.9739 | +0.2165 | 0.9992 | +0.7040 |
| oracle_top3 | 0.9727 | +0.2153 | 0.9992 | +0.7040 |
| oracle_graph_top2 | 0.9402 | +0.1828 | 0.9979 | +0.7027 |
| oracle_top2 | 0.9367 | +0.1793 | 0.9747 | +0.6795 |
| oracle_graph_top1 | 0.8057 | +0.0483 | 0.9664 | +0.6712 |
| base | 0.7574 | +0.0000 | 0.2952 | +0.0000 |
| prior_alpha0p25 | 0.7388 | -0.0186 | 0.2695 | -0.0257 |
| pair_probe_top2 | 0.7219 | -0.0355 | 0.1235 | -0.1717 |
| prior_alpha0p5 | 0.6875 | -0.0699 | 0.2361 | -0.0591 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.8716 | [0.8621, 0.8811] | 4750 |
| picture | topk | 2 | 0.9994 | [0.9987, 1.0000] | 4750 |
| picture | topk | 3 | 1.0000 | [1.0000, 1.0000] | 4750 |
| picture | topk | 5 | 1.0000 | [1.0000, 1.0000] | 4750 |
| picture | topk | 10 | 1.0000 | [1.0000, 1.0000] | 4750 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 4750 |
| picture | topk_plus_confusion_graph | 1 | 1.0000 | [1.0000, 1.0000] | 4750 |
| picture | topk_plus_confusion_graph | 2 | 1.0000 | [1.0000, 1.0000] | 4750 |
| picture | topk_plus_confusion_graph | 3 | 1.0000 | [1.0000, 1.0000] | 4750 |
| picture | topk_plus_confusion_graph | 5 | 1.0000 | [1.0000, 1.0000] | 4750 |
| counter | topk | 1 | 0.8703 | [0.8675, 0.8731] | 56243 |
| counter | topk | 2 | 0.9858 | [0.9848, 0.9868] | 56243 |
| counter | topk | 3 | 0.9970 | [0.9965, 0.9974] | 56243 |
| counter | topk | 5 | 0.9993 | [0.9991, 0.9995] | 56243 |
| counter | topk | 10 | 1.0000 | [1.0000, 1.0000] | 56243 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 56243 |
| counter | topk_plus_confusion_graph | 1 | 0.9222 | [0.9200, 0.9244] | 56243 |
| counter | topk_plus_confusion_graph | 2 | 0.9950 | [0.9944, 0.9956] | 56243 |
| counter | topk_plus_confusion_graph | 3 | 0.9992 | [0.9989, 0.9994] | 56243 |
| counter | topk_plus_confusion_graph | 5 | 0.9996 | [0.9994, 0.9998] | 56243 |
| door | topk | 1 | 0.9201 | [0.9193, 0.9209] | 435410 |
| door | topk | 2 | 0.9900 | [0.9897, 0.9903] | 435410 |
| door | topk | 3 | 0.9956 | [0.9954, 0.9958] | 435410 |
| door | topk | 5 | 0.9994 | [0.9993, 0.9994] | 435410 |
| door | topk | 10 | 1.0000 | [1.0000, 1.0000] | 435410 |
| door | topk | 20 | 1.0000 | [1.0000, 1.0000] | 435410 |
| door | topk_plus_confusion_graph | 1 | 0.9787 | [0.9783, 0.9791] | 435410 |
| door | topk_plus_confusion_graph | 2 | 0.9986 | [0.9984, 0.9987] | 435410 |
| door | topk_plus_confusion_graph | 3 | 0.9995 | [0.9994, 0.9996] | 435410 |
| door | topk_plus_confusion_graph | 5 | 1.0000 | [0.9999, 1.0000] | 435410 |

## Key Readout Headroom

- base mIoU: `0.7574`
- base picture IoU: `0.2952`
- base picture -> wall fraction: `0.1284`
- best non-base mIoU variant: `oracle_top10` (`0.9992`, delta `+0.2418`)
- best non-base picture variant: `oracle_top5` (`1.0000`, delta `+0.7048`)

## Output Files

- `oracle_topk_hit_rates.csv`
- `oracle_variants.csv`
- `oracle_confusion_distribution.csv`
- `oracle_top3_distribution.csv`
- `oracle_feature_geometry.csv`
- `oracle_pair_probe_train.csv`
- `metadata.json`
