# Utonia Oracle Actionability Analysis

## Setup
- utonia weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia.pth`
- seg head weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia_linear_prob_head_sc.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- weak classes: `picture,counter,door`
- class pairs: `picture:wall,door:wall,counter:cabinet`
- train batches seen: 14
- val batches seen: 312

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9973 | +0.2401 | 0.9830 | +0.6080 |
| oracle_graph_top5 | 0.9867 | +0.2294 | 0.9990 | +0.6241 |
| oracle_top5 | 0.9821 | +0.2248 | 0.9281 | +0.5531 |
| oracle_graph_top3 | 0.9630 | +0.2058 | 0.9950 | +0.6200 |
| oracle_top3 | 0.9541 | +0.1968 | 0.8670 | +0.4921 |
| oracle_graph_top2 | 0.9270 | +0.1698 | 0.9905 | +0.6155 |
| oracle_top2 | 0.9116 | +0.1543 | 0.8043 | +0.4293 |
| oracle_graph_top1 | 0.8071 | +0.0498 | 0.9780 | +0.6030 |
| multiproto4_adapt_tau0p05_lam0p2 | 0.7574 | +0.0001 | 0.3755 | +0.0006 |
| multiproto4_adapt_tau0p1_lam0p2 | 0.7574 | +0.0001 | 0.3754 | +0.0005 |
| multiproto4_tau0p2_lam0p2 | 0.7574 | +0.0001 | 0.3756 | +0.0006 |
| multiproto4_adapt_tau0p05_lam0p1 | 0.7574 | +0.0001 | 0.3755 | +0.0005 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.5058 | [0.5037, 0.5080] | 207442 |
| picture | topk | 2 | 0.8106 | [0.8089, 0.8123] | 207442 |
| picture | topk | 3 | 0.8692 | [0.8677, 0.8706] | 207442 |
| picture | topk | 5 | 0.9286 | [0.9275, 0.9297] | 207442 |
| picture | topk | 10 | 0.9831 | [0.9825, 0.9836] | 207442 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 207442 |
| picture | topk_plus_confusion_graph | 1 | 0.9872 | [0.9867, 0.9877] | 207442 |
| picture | topk_plus_confusion_graph | 2 | 0.9959 | [0.9956, 0.9961] | 207442 |
| picture | topk_plus_confusion_graph | 3 | 0.9973 | [0.9971, 0.9975] | 207442 |
| picture | topk_plus_confusion_graph | 5 | 0.9996 | [0.9995, 0.9996] | 207442 |
| counter | topk | 1 | 0.8439 | [0.8424, 0.8454] | 221560 |
| counter | topk | 2 | 0.9539 | [0.9531, 0.9548] | 221560 |
| counter | topk | 3 | 0.9809 | [0.9803, 0.9815] | 221560 |
| counter | topk | 5 | 0.9901 | [0.9896, 0.9905] | 221560 |
| counter | topk | 10 | 0.9978 | [0.9976, 0.9980] | 221560 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 221560 |
| counter | topk_plus_confusion_graph | 1 | 0.9321 | [0.9310, 0.9331] | 221560 |
| counter | topk_plus_confusion_graph | 2 | 0.9950 | [0.9947, 0.9953] | 221560 |
| counter | topk_plus_confusion_graph | 3 | 0.9994 | [0.9993, 0.9995] | 221560 |
| counter | topk_plus_confusion_graph | 5 | 0.9999 | [0.9999, 0.9999] | 221560 |
| door | topk | 1 | 0.8634 | [0.8628, 0.8639] | 1600191 |
| door | topk | 2 | 0.9756 | [0.9753, 0.9758] | 1600191 |
| door | topk | 3 | 0.9904 | [0.9902, 0.9905] | 1600191 |
| door | topk | 5 | 0.9970 | [0.9969, 0.9971] | 1600191 |
| door | topk | 10 | 0.9999 | [0.9998, 0.9999] | 1600191 |
| door | topk | 20 | 1.0000 | [1.0000, 1.0000] | 1600191 |
| door | topk_plus_confusion_graph | 1 | 0.9683 | [0.9680, 0.9686] | 1600191 |
| door | topk_plus_confusion_graph | 2 | 0.9980 | [0.9980, 0.9981] | 1600191 |
| door | topk_plus_confusion_graph | 3 | 0.9996 | [0.9996, 0.9997] | 1600191 |
| door | topk_plus_confusion_graph | 5 | 1.0000 | [1.0000, 1.0000] | 1600191 |

## Key Readout Headroom

- base mIoU: `0.7573`
- base picture IoU: `0.3749`
- base picture -> wall fraction: `0.4813`
- best non-base mIoU variant: `oracle_top10` (`0.9973`, delta `+0.2401`)
- best non-base picture variant: `oracle_graph_top5` (`0.9990`, delta `+0.6241`)

## Output Files

- `oracle_topk_hit_rates.csv`
- `oracle_variants.csv`
- `oracle_confusion_distribution.csv`
- `oracle_top3_distribution.csv`
- `oracle_feature_geometry.csv`
- `oracle_pair_probe_train.csv`
- `metadata.json`
