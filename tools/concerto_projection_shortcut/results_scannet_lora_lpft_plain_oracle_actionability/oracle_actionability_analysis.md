# Oracle Actionability Analysis

## Setup
- config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-ft-lora-origin-e100.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/exp/concerto/scannet-lora-origin-lpft-plain-e100/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- weak classes: `picture,counter,desk,sink,cabinet,shower curtain,door`
- class pairs: `picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall`
- train batches seen: 18
- val batches seen: 312

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9970 | +0.2190 | 0.9938 | +0.5799 |
| oracle_graph_top5 | 0.9931 | +0.2151 | 0.9964 | +0.5825 |
| oracle_top5 | 0.9860 | +0.2080 | 0.9534 | +0.5395 |
| oracle_graph_top3 | 0.9803 | +0.2023 | 0.9852 | +0.5713 |
| oracle_top3 | 0.9663 | +0.1882 | 0.9103 | +0.4964 |
| oracle_graph_top2 | 0.9534 | +0.1754 | 0.9812 | +0.5673 |
| oracle_top2 | 0.9263 | +0.1483 | 0.8198 | +0.4059 |
| oracle_graph_top1 | 0.8465 | +0.0685 | 0.9645 | +0.5506 |
| base | 0.7780 | +0.0000 | 0.4139 | +0.0000 |
| prior_alpha0p25 | 0.7755 | -0.0025 | 0.4120 | -0.0020 |
| prior_alpha0p5 | 0.7690 | -0.0090 | 0.3750 | -0.0390 |
| bias_unweighted | 0.7629 | -0.0152 | 0.4127 | -0.0012 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.5403 | [0.5381, 0.5424] | 207442 |
| picture | topk | 2 | 0.8354 | [0.8338, 0.8370] | 207442 |
| picture | topk | 3 | 0.9230 | [0.9219, 0.9242] | 207442 |
| picture | topk | 5 | 0.9567 | [0.9558, 0.9576] | 207442 |
| picture | topk | 10 | 0.9938 | [0.9935, 0.9942] | 207442 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 207442 |
| picture | topk_plus_confusion_graph | 1 | 0.9817 | [0.9811, 0.9823] | 207442 |
| picture | topk_plus_confusion_graph | 2 | 0.9960 | [0.9957, 0.9963] | 207442 |
| picture | topk_plus_confusion_graph | 3 | 0.9983 | [0.9982, 0.9985] | 207442 |
| picture | topk_plus_confusion_graph | 5 | 0.9997 | [0.9997, 0.9998] | 207442 |
| counter | topk | 1 | 0.8389 | [0.8374, 0.8405] | 221560 |
| counter | topk | 2 | 0.9534 | [0.9525, 0.9542] | 221560 |
| counter | topk | 3 | 0.9828 | [0.9823, 0.9834] | 221560 |
| counter | topk | 5 | 0.9937 | [0.9933, 0.9940] | 221560 |
| counter | topk | 10 | 0.9992 | [0.9990, 0.9993] | 221560 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 221560 |
| counter | topk_plus_confusion_graph | 1 | 0.9318 | [0.9308, 0.9329] | 221560 |
| counter | topk_plus_confusion_graph | 2 | 0.9949 | [0.9946, 0.9952] | 221560 |
| counter | topk_plus_confusion_graph | 3 | 0.9995 | [0.9994, 0.9996] | 221560 |
| counter | topk_plus_confusion_graph | 5 | 1.0000 | [0.9999, 1.0000] | 221560 |
| desk | topk | 1 | 0.8366 | [0.8357, 0.8375] | 655163 |
| desk | topk | 2 | 0.9606 | [0.9601, 0.9611] | 655163 |
| desk | topk | 3 | 0.9822 | [0.9819, 0.9825] | 655163 |
| desk | topk | 5 | 0.9959 | [0.9957, 0.9961] | 655163 |
| desk | topk | 10 | 1.0000 | [0.9999, 1.0000] | 655163 |
| desk | topk | 20 | 1.0000 | [1.0000, 1.0000] | 655163 |
| desk | topk_plus_confusion_graph | 1 | 0.8820 | [0.8812, 0.8828] | 655163 |
| desk | topk_plus_confusion_graph | 2 | 0.9719 | [0.9715, 0.9723] | 655163 |
| desk | topk_plus_confusion_graph | 3 | 0.9908 | [0.9906, 0.9911] | 655163 |
| desk | topk_plus_confusion_graph | 5 | 0.9978 | [0.9977, 0.9980] | 655163 |
| sink | topk | 1 | 0.8230 | [0.8207, 0.8254] | 102489 |
| sink | topk | 2 | 0.9083 | [0.9066, 0.9101] | 102489 |
| sink | topk | 3 | 0.9501 | [0.9487, 0.9514] | 102489 |
| sink | topk | 5 | 0.9771 | [0.9762, 0.9780] | 102489 |
| sink | topk | 10 | 0.9984 | [0.9982, 0.9987] | 102489 |
| sink | topk | 20 | 1.0000 | [1.0000, 1.0000] | 102489 |
| sink | topk_plus_confusion_graph | 1 | 0.9102 | [0.9084, 0.9119] | 102489 |
| sink | topk_plus_confusion_graph | 2 | 0.9780 | [0.9771, 0.9789] | 102489 |
| sink | topk_plus_confusion_graph | 3 | 0.9880 | [0.9874, 0.9887] | 102489 |
| sink | topk_plus_confusion_graph | 5 | 0.9987 | [0.9985, 0.9989] | 102489 |
| cabinet | topk | 1 | 0.8449 | [0.8443, 0.8455] | 1399895 |
| cabinet | topk | 2 | 0.9514 | [0.9510, 0.9517] | 1399895 |
| cabinet | topk | 5 | 0.9972 | [0.9971, 0.9973] | 1399895 |
| cabinet | topk_plus_confusion_graph | 1 | 0.8680 | [0.8675, 0.8686] | 1399895 |
| cabinet | topk_plus_confusion_graph | 2 | 0.9554 | [0.9550, 0.9557] | 1399895 |
| cabinet | topk_plus_confusion_graph | 5 | 0.9977 | [0.9976, 0.9977] | 1399895 |
| shower curtain | topk | 1 | 0.8078 | [0.8058, 0.8097] | 158208 |
| shower curtain | topk | 2 | 0.8774 | [0.8758, 0.8790] | 158208 |
| shower curtain | topk | 5 | 0.9608 | [0.9598, 0.9617] | 158208 |
| shower curtain | topk_plus_confusion_graph | 1 | 0.9538 | [0.9528, 0.9548] | 158208 |
| shower curtain | topk_plus_confusion_graph | 2 | 0.9967 | [0.9965, 0.9970] | 158208 |
| shower curtain | topk_plus_confusion_graph | 5 | 0.9998 | [0.9997, 0.9999] | 158208 |
| door | topk | 1 | 0.8964 | [0.8960, 0.8969] | 1600191 |
| door | topk | 2 | 0.9840 | [0.9838, 0.9842] | 1600191 |
| door | topk | 5 | 0.9981 | [0.9980, 0.9982] | 1600191 |
| door | topk_plus_confusion_graph | 1 | 0.9754 | [0.9752, 0.9757] | 1600191 |
| door | topk_plus_confusion_graph | 2 | 0.9980 | [0.9979, 0.9980] | 1600191 |
| door | topk_plus_confusion_graph | 5 | 0.9998 | [0.9998, 0.9998] | 1600191 |

## Key Readout Headroom

- base mIoU: `0.7780`
- base picture IoU: `0.4139`
- base picture -> wall fraction: `0.4414`
- best non-base mIoU variant: `oracle_top10` (`0.9970`, delta `+0.2190`)
- best non-base picture variant: `oracle_graph_top5` (`0.9964`, delta `+0.5825`)

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
