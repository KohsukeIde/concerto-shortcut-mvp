# PTv3 v1.5.1 Oracle Actionability Analysis

## Setup
- official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- segment key: `segment20`
- weak classes: `picture,counter,desk,sink,cabinet,shower curtain,door`
- class pairs: `picture:wall,counter:cabinet,door:wall`
- train scenes seen: 14
- val scenes seen: 128

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9935 | +0.2190 | 0.9955 | +0.5047 |
| oracle_graph_top5 | 0.9695 | +0.1950 | 1.0000 | +0.5092 |
| oracle_top5 | 0.9690 | +0.1945 | 0.9952 | +0.5045 |
| oracle_graph_top3 | 0.9420 | +0.1675 | 1.0000 | +0.5092 |
| oracle_top3 | 0.9390 | +0.1645 | 0.9697 | +0.4789 |
| oracle_graph_top2 | 0.9152 | +0.1407 | 0.9994 | +0.5086 |
| oracle_top2 | 0.9038 | +0.1293 | 0.8785 | +0.3878 |
| oracle_graph_top1 | 0.8147 | +0.0402 | 0.9763 | +0.4855 |
| base | 0.7745 | +0.0000 | 0.4908 | +0.0000 |
| prior_alpha0p25 | 0.7714 | -0.0031 | 0.4868 | -0.0040 |
| prior_alpha0p5 | 0.7596 | -0.0149 | 0.4825 | -0.0082 |
| bias_unweighted | 0.7553 | -0.0192 | 0.3891 | -0.1017 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.7452 | [0.7405, 0.7500] | 32521 |
| picture | topk | 2 | 0.8791 | [0.8755, 0.8826] | 32521 |
| picture | topk | 3 | 0.9697 | [0.9678, 0.9715] | 32521 |
| picture | topk | 5 | 0.9952 | [0.9945, 0.9960] | 32521 |
| picture | topk | 10 | 0.9955 | [0.9948, 0.9962] | 32521 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 32521 |
| picture | topk_plus_confusion_graph | 1 | 0.9779 | [0.9763, 0.9795] | 32521 |
| picture | topk_plus_confusion_graph | 2 | 1.0000 | [0.9999, 1.0000] | 32521 |
| picture | topk_plus_confusion_graph | 3 | 1.0000 | [1.0000, 1.0000] | 32521 |
| picture | topk_plus_confusion_graph | 5 | 1.0000 | [1.0000, 1.0000] | 32521 |
| counter | topk | 1 | 0.8952 | [0.8932, 0.8971] | 95560 |
| counter | topk | 2 | 0.9700 | [0.9689, 0.9711] | 95560 |
| counter | topk | 3 | 0.9872 | [0.9865, 0.9879] | 95560 |
| counter | topk | 5 | 0.9953 | [0.9949, 0.9957] | 95560 |
| counter | topk | 10 | 0.9986 | [0.9984, 0.9989] | 95560 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 95560 |
| counter | topk_plus_confusion_graph | 1 | 0.9172 | [0.9155, 0.9190] | 95560 |
| counter | topk_plus_confusion_graph | 2 | 0.9843 | [0.9835, 0.9850] | 95560 |
| counter | topk_plus_confusion_graph | 3 | 0.9923 | [0.9918, 0.9929] | 95560 |
| counter | topk_plus_confusion_graph | 5 | 0.9969 | [0.9966, 0.9973] | 95560 |
| desk | topk | 1 | 0.8766 | [0.8753, 0.8780] | 223974 |
| desk | topk | 2 | 0.9433 | [0.9423, 0.9442] | 223974 |
| desk | topk | 3 | 0.9603 | [0.9595, 0.9611] | 223974 |
| desk | topk | 5 | 0.9738 | [0.9731, 0.9745] | 223974 |
| desk | topk | 10 | 0.9871 | [0.9867, 0.9876] | 223974 |
| desk | topk | 20 | 1.0000 | [1.0000, 1.0000] | 223974 |
| desk | topk_plus_confusion_graph | 1 | 0.8766 | [0.8753, 0.8780] | 223974 |
| desk | topk_plus_confusion_graph | 2 | 0.9433 | [0.9423, 0.9442] | 223974 |
| desk | topk_plus_confusion_graph | 3 | 0.9603 | [0.9595, 0.9611] | 223974 |
| desk | topk_plus_confusion_graph | 5 | 0.9738 | [0.9731, 0.9745] | 223974 |
| sink | topk | 1 | 0.8650 | [0.8613, 0.8687] | 32421 |
| sink | topk | 2 | 0.9475 | [0.9450, 0.9499] | 32421 |
| sink | topk | 3 | 0.9680 | [0.9661, 0.9700] | 32421 |
| sink | topk | 5 | 0.9768 | [0.9752, 0.9785] | 32421 |
| sink | topk | 10 | 0.9996 | [0.9994, 0.9998] | 32421 |
| sink | topk | 20 | 1.0000 | [1.0000, 1.0000] | 32421 |
| sink | topk_plus_confusion_graph | 1 | 0.8650 | [0.8613, 0.8687] | 32421 |
| sink | topk_plus_confusion_graph | 2 | 0.9475 | [0.9450, 0.9499] | 32421 |
| sink | topk_plus_confusion_graph | 3 | 0.9680 | [0.9661, 0.9700] | 32421 |
| sink | topk_plus_confusion_graph | 5 | 0.9768 | [0.9752, 0.9785] | 32421 |
| cabinet | topk | 1 | 0.8495 | [0.8484, 0.8505] | 457364 |
| cabinet | topk | 2 | 0.9154 | [0.9146, 0.9162] | 457364 |
| cabinet | topk | 5 | 0.9791 | [0.9787, 0.9796] | 457364 |
| cabinet | topk_plus_confusion_graph | 1 | 0.8694 | [0.8684, 0.8704] | 457364 |
| cabinet | topk_plus_confusion_graph | 2 | 0.9231 | [0.9224, 0.9239] | 457364 |
| cabinet | topk_plus_confusion_graph | 5 | 0.9792 | [0.9788, 0.9796] | 457364 |
| shower curtain | topk | 1 | 0.9690 | [0.9671, 0.9709] | 32505 |
| shower curtain | topk | 2 | 0.9969 | [0.9963, 0.9975] | 32505 |
| shower curtain | topk | 5 | 0.9989 | [0.9986, 0.9993] | 32505 |
| shower curtain | topk_plus_confusion_graph | 1 | 0.9690 | [0.9671, 0.9709] | 32505 |
| shower curtain | topk_plus_confusion_graph | 2 | 0.9969 | [0.9963, 0.9975] | 32505 |
| shower curtain | topk_plus_confusion_graph | 5 | 0.9989 | [0.9986, 0.9993] | 32505 |
| door | topk | 1 | 0.8514 | [0.8505, 0.8524] | 564391 |
| door | topk | 2 | 0.9766 | [0.9762, 0.9770] | 564391 |
| door | topk | 3 | 0.9932 | [0.9929, 0.9934] | 564391 |
| door | topk | 5 | 0.9985 | [0.9983, 0.9986] | 564391 |
| door | topk | 10 | 0.9997 | [0.9997, 0.9998] | 564391 |
| door | topk | 20 | 1.0000 | [1.0000, 1.0000] | 564391 |
| door | topk_plus_confusion_graph | 1 | 0.9514 | [0.9508, 0.9519] | 564391 |
| door | topk_plus_confusion_graph | 2 | 0.9972 | [0.9970, 0.9973] | 564391 |
| door | topk_plus_confusion_graph | 3 | 0.9994 | [0.9994, 0.9995] | 564391 |
| door | topk_plus_confusion_graph | 5 | 0.9999 | [0.9999, 0.9999] | 564391 |

## Key Readout Headroom

- base mIoU: `0.7745`
- base picture IoU: `0.4908`
- base picture -> wall fraction: `0.2326`
- best non-base mIoU variant: `oracle_top10` (`0.9935`, delta `+0.2190`)
- best non-base picture variant: `oracle_graph_top3` (`1.0000`, delta `+0.5092`)

## Output Files

- `oracle_topk_hit_rates.csv`
- `oracle_variants.csv`
- `oracle_confusion_distribution.csv`
- `oracle_top3_distribution.csv`
- `oracle_feature_geometry.csv`
- `oracle_pair_probe_train.csv`
- `metadata.json`
