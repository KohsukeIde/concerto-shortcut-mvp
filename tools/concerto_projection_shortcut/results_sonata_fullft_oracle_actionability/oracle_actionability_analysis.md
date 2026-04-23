# Oracle Actionability Analysis

## Setup
- config: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/config.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- weak classes: `picture,counter,door`
- class pairs: `picture:wall,door:wall,counter:cabinet`
- train batches seen: 18
- val batches seen: 312

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9884 | +0.2114 | 0.9413 | +0.5905 |
| oracle_graph_top5 | 0.9663 | +0.1893 | 0.9924 | +0.6416 |
| oracle_top5 | 0.9519 | +0.1749 | 0.7700 | +0.4192 |
| oracle_graph_top3 | 0.9422 | +0.1652 | 0.9878 | +0.6370 |
| oracle_top3 | 0.9205 | +0.1435 | 0.6741 | +0.3233 |
| oracle_graph_top2 | 0.9154 | +0.1384 | 0.9824 | +0.6316 |
| oracle_top2 | 0.8856 | +0.1086 | 0.6003 | +0.2495 |
| oracle_graph_top1 | 0.8256 | +0.0486 | 0.9672 | +0.6164 |
| prior_alpha0p25 | 0.7772 | +0.0002 | 0.3539 | +0.0031 |
| base | 0.7770 | +0.0000 | 0.3508 | +0.0000 |
| prior_alpha0p5 | 0.7769 | -0.0001 | 0.3565 | +0.0057 |
| prior_alpha0p75 | 0.7764 | -0.0006 | 0.3596 | +0.0088 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.4366 | [0.4345, 0.4388] | 207442 |
| picture | topk | 2 | 0.6206 | [0.6185, 0.6227] | 207442 |
| picture | topk | 3 | 0.6814 | [0.6794, 0.6834] | 207442 |
| picture | topk | 5 | 0.7746 | [0.7728, 0.7764] | 207442 |
| picture | topk | 10 | 0.9421 | [0.9411, 0.9432] | 207442 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 207442 |
| picture | topk_plus_confusion_graph | 1 | 0.9845 | [0.9839, 0.9850] | 207442 |
| picture | topk_plus_confusion_graph | 2 | 0.9960 | [0.9958, 0.9963] | 207442 |
| picture | topk_plus_confusion_graph | 3 | 0.9968 | [0.9965, 0.9970] | 207442 |
| picture | topk_plus_confusion_graph | 5 | 0.9973 | [0.9971, 0.9975] | 207442 |
| counter | topk | 1 | 0.8139 | [0.8123, 0.8155] | 221560 |
| counter | topk | 2 | 0.8897 | [0.8884, 0.8910] | 221560 |
| counter | topk | 3 | 0.9119 | [0.9107, 0.9131] | 221560 |
| counter | topk | 5 | 0.9326 | [0.9316, 0.9337] | 221560 |
| counter | topk | 10 | 0.9901 | [0.9897, 0.9905] | 221560 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 221560 |
| counter | topk_plus_confusion_graph | 1 | 0.9085 | [0.9073, 0.9097] | 221560 |
| counter | topk_plus_confusion_graph | 2 | 0.9620 | [0.9613, 0.9628] | 221560 |
| counter | topk_plus_confusion_graph | 3 | 0.9684 | [0.9677, 0.9691] | 221560 |
| counter | topk_plus_confusion_graph | 5 | 0.9706 | [0.9699, 0.9713] | 221560 |
| door | topk | 1 | 0.8421 | [0.8416, 0.8427] | 1600191 |
| door | topk | 2 | 0.9450 | [0.9446, 0.9453] | 1600191 |
| door | topk | 5 | 0.9946 | [0.9944, 0.9947] | 1600191 |
| door | topk_plus_confusion_graph | 1 | 0.9446 | [0.9443, 0.9450] | 1600191 |
| door | topk_plus_confusion_graph | 2 | 0.9944 | [0.9943, 0.9946] | 1600191 |
| door | topk_plus_confusion_graph | 5 | 0.9996 | [0.9996, 0.9996] | 1600191 |

## Key Readout Headroom

- base mIoU: `0.7770`
- base picture IoU: `0.3508`
- base picture -> wall fraction: `0.5478`
- best non-base mIoU variant: `oracle_top10` (`0.9884`, delta `+0.2114`)
- best non-base picture variant: `oracle_graph_top5` (`0.9924`, delta `+0.6416`)

## Output Files

- `oracle_topk_hit_rates.csv`
- `oracle_variants.csv`
- `oracle_confusion_distribution.csv`
- `oracle_top3_distribution.csv`
- `oracle_feature_geometry.csv`
- `oracle_pair_probe_train.csv`
- `metadata.json`

## Interpretation Notes

- `oracle_topK` variants are upper bounds: if the ground-truth class is in the candidate set, prediction is replaced by the ground truth.
- `oracle_graph_topK` expands top-K by the predefined confusion graph before applying the same oracle rule.
- `pair_probe_top2` is a learned readout-family variant: when a configured pair appears as the top-2 classes, a train-fitted binary point-feature probe chooses between them.
- `prior_alpha*`, `bias_unweighted`, and `bias_balanced` are train-derived calibration variants, not val-tuned oracle variants.
