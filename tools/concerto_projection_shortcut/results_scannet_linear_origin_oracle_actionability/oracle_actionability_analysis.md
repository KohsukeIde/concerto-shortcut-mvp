# Oracle Actionability Analysis

## Setup
- config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-lin-origin-e100.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/exp/concerto/scannet-lin-origin-e100/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- weak classes: `picture,counter,desk,sink,cabinet,shower curtain,door`
- class pairs: `picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall`
- train batches seen: 18
- val batches seen: 312

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9955 | +0.2341 | 0.9815 | +0.5801 |
| oracle_graph_top5 | 0.9916 | +0.2302 | 0.9959 | +0.5944 |
| oracle_top5 | 0.9839 | +0.2225 | 0.9394 | +0.5380 |
| oracle_graph_top3 | 0.9757 | +0.2143 | 0.9886 | +0.5872 |
| oracle_top3 | 0.9595 | +0.1980 | 0.8837 | +0.4823 |
| oracle_graph_top2 | 0.9458 | +0.1844 | 0.9772 | +0.5757 |
| oracle_top2 | 0.9171 | +0.1556 | 0.8013 | +0.3998 |
| oracle_graph_top1 | 0.8299 | +0.0685 | 0.9431 | +0.5416 |
| base | 0.7615 | +0.0000 | 0.4014 | +0.0000 |
| prior_alpha0p25 | 0.7569 | -0.0046 | 0.3830 | -0.0184 |
| bias_unweighted | 0.7466 | -0.0149 | 0.3959 | -0.0055 |
| prior_alpha0p5 | 0.7461 | -0.0154 | 0.3218 | -0.0796 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.5380 | [0.5358, 0.5401] | 207442 |
| picture | topk | 2 | 0.8164 | [0.8147, 0.8180] | 207442 |
| picture | topk | 3 | 0.8921 | [0.8907, 0.8934] | 207442 |
| picture | topk | 5 | 0.9431 | [0.9421, 0.9441] | 207442 |
| picture | topk | 10 | 0.9815 | [0.9810, 0.9821] | 207442 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 207442 |
| picture | topk_plus_confusion_graph | 1 | 0.9603 | [0.9595, 0.9612] | 207442 |
| picture | topk_plus_confusion_graph | 2 | 0.9938 | [0.9934, 0.9941] | 207442 |
| picture | topk_plus_confusion_graph | 3 | 0.9974 | [0.9972, 0.9977] | 207442 |
| picture | topk_plus_confusion_graph | 5 | 0.9995 | [0.9995, 0.9996] | 207442 |
| counter | topk | 1 | 0.8437 | [0.8422, 0.8452] | 221560 |
| counter | topk | 2 | 0.9542 | [0.9533, 0.9551] | 221560 |
| counter | topk | 3 | 0.9755 | [0.9749, 0.9762] | 221560 |
| counter | topk | 5 | 0.9849 | [0.9844, 0.9855] | 221560 |
| counter | topk | 10 | 0.9958 | [0.9956, 0.9961] | 221560 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 221560 |
| counter | topk_plus_confusion_graph | 1 | 0.9305 | [0.9294, 0.9315] | 221560 |
| counter | topk_plus_confusion_graph | 2 | 0.9966 | [0.9963, 0.9968] | 221560 |
| counter | topk_plus_confusion_graph | 3 | 0.9997 | [0.9996, 0.9998] | 221560 |
| counter | topk_plus_confusion_graph | 5 | 1.0000 | [1.0000, 1.0000] | 221560 |
| desk | topk | 1 | 0.8450 | [0.8442, 0.8459] | 655163 |
| desk | topk | 2 | 0.9632 | [0.9627, 0.9636] | 655163 |
| desk | topk | 3 | 0.9821 | [0.9818, 0.9825] | 655163 |
| desk | topk | 5 | 0.9952 | [0.9950, 0.9953] | 655163 |
| desk | topk | 10 | 1.0000 | [0.9999, 1.0000] | 655163 |
| desk | topk | 20 | 1.0000 | [1.0000, 1.0000] | 655163 |
| desk | topk_plus_confusion_graph | 1 | 0.8839 | [0.8831, 0.8846] | 655163 |
| desk | topk_plus_confusion_graph | 2 | 0.9736 | [0.9732, 0.9740] | 655163 |
| desk | topk_plus_confusion_graph | 3 | 0.9898 | [0.9896, 0.9900] | 655163 |
| desk | topk_plus_confusion_graph | 5 | 0.9969 | [0.9968, 0.9971] | 655163 |
| sink | topk | 1 | 0.8191 | [0.8168, 0.8215] | 102489 |
| sink | topk | 2 | 0.9066 | [0.9048, 0.9084] | 102489 |
| sink | topk | 3 | 0.9488 | [0.9475, 0.9502] | 102489 |
| sink | topk | 5 | 0.9728 | [0.9719, 0.9738] | 102489 |
| sink | topk | 10 | 0.9873 | [0.9866, 0.9880] | 102489 |
| sink | topk | 20 | 1.0000 | [1.0000, 1.0000] | 102489 |
| sink | topk_plus_confusion_graph | 1 | 0.9004 | [0.8986, 0.9022] | 102489 |
| sink | topk_plus_confusion_graph | 2 | 0.9788 | [0.9779, 0.9797] | 102489 |
| sink | topk_plus_confusion_graph | 3 | 0.9923 | [0.9917, 0.9928] | 102489 |
| sink | topk_plus_confusion_graph | 5 | 0.9994 | [0.9992, 0.9995] | 102489 |
| cabinet | topk | 1 | 0.8058 | [0.8051, 0.8064] | 1399895 |
| cabinet | topk | 2 | 0.9372 | [0.9367, 0.9376] | 1399895 |
| cabinet | topk | 5 | 0.9949 | [0.9947, 0.9950] | 1399895 |
| cabinet | topk_plus_confusion_graph | 1 | 0.8327 | [0.8321, 0.8333] | 1399895 |
| cabinet | topk_plus_confusion_graph | 2 | 0.9430 | [0.9426, 0.9434] | 1399895 |
| cabinet | topk_plus_confusion_graph | 5 | 0.9957 | [0.9956, 0.9958] | 1399895 |
| shower curtain | topk | 1 | 0.8012 | [0.7992, 0.8031] | 158208 |
| shower curtain | topk | 2 | 0.8906 | [0.8890, 0.8921] | 158208 |
| shower curtain | topk | 5 | 0.9843 | [0.9837, 0.9849] | 158208 |
| shower curtain | topk_plus_confusion_graph | 1 | 0.9418 | [0.9407, 0.9430] | 158208 |
| shower curtain | topk_plus_confusion_graph | 2 | 0.9936 | [0.9932, 0.9940] | 158208 |
| shower curtain | topk_plus_confusion_graph | 5 | 0.9998 | [0.9998, 0.9999] | 158208 |
| door | topk | 1 | 0.8797 | [0.8792, 0.8802] | 1600191 |
| door | topk | 2 | 0.9806 | [0.9804, 0.9808] | 1600191 |
| door | topk | 5 | 0.9985 | [0.9984, 0.9986] | 1600191 |
| door | topk_plus_confusion_graph | 1 | 0.9639 | [0.9636, 0.9642] | 1600191 |
| door | topk_plus_confusion_graph | 2 | 0.9974 | [0.9974, 0.9975] | 1600191 |
| door | topk_plus_confusion_graph | 5 | 0.9999 | [0.9999, 0.9999] | 1600191 |

## Key Readout Headroom

- base mIoU: `0.7615`
- base picture IoU: `0.4014`
- base picture -> wall fraction: `0.4224`
- best non-base mIoU variant: `oracle_top10` (`0.9955`, delta `+0.2341`)
- best non-base picture variant: `oracle_graph_top5` (`0.9959`, delta `+0.5944`)

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
