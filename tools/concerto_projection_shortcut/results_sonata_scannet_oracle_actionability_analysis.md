# Oracle Actionability Analysis

## Setup
- config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/sonata/sonata_scannet_linear_merged.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- weak classes: `picture,counter,desk,sink,cabinet,shower curtain,door`
- class pairs: `picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall`
- train batches seen: 256
- val batches seen: 312

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9921 | +0.2835 | 0.9698 | +0.6116 |
| oracle_graph_top5 | 0.9804 | +0.2717 | 0.9922 | +0.6339 |
| oracle_top5 | 0.9670 | +0.2583 | 0.8867 | +0.5284 |
| oracle_graph_top3 | 0.9504 | +0.2417 | 0.9872 | +0.6290 |
| oracle_top3 | 0.9260 | +0.2174 | 0.7870 | +0.4288 |
| oracle_graph_top2 | 0.9119 | +0.2032 | 0.9774 | +0.6191 |
| oracle_top2 | 0.8747 | +0.1660 | 0.6972 | +0.3389 |
| oracle_graph_top1 | 0.7817 | +0.0731 | 0.9411 | +0.5828 |
| base | 0.7086 | +0.0000 | 0.3582 | +0.0000 |
| prior_alpha0p25 | 0.7086 | -0.0001 | 0.3582 | +0.0000 |
| prior_alpha0p5 | 0.7085 | -0.0001 | 0.3582 | +0.0000 |
| prior_alpha0p75 | 0.7084 | -0.0002 | 0.3582 | +0.0000 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.4806 | [0.4784, 0.4827] | 207442 |
| picture | topk | 2 | 0.7102 | [0.7082, 0.7121] | 207442 |
| picture | topk | 3 | 0.7937 | [0.7919, 0.7954] | 207442 |
| picture | topk | 5 | 0.8926 | [0.8913, 0.8940] | 207442 |
| picture | topk | 10 | 0.9700 | [0.9692, 0.9707] | 207442 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 207442 |
| picture | topk_plus_confusion_graph | 1 | 0.9589 | [0.9580, 0.9598] | 207442 |
| picture | topk_plus_confusion_graph | 2 | 0.9923 | [0.9919, 0.9927] | 207442 |
| picture | topk_plus_confusion_graph | 3 | 0.9953 | [0.9950, 0.9956] | 207442 |
| picture | topk_plus_confusion_graph | 5 | 0.9987 | [0.9986, 0.9989] | 207442 |
| counter | topk | 1 | 0.7782 | [0.7764, 0.7799] | 221560 |
| counter | topk | 2 | 0.9052 | [0.9040, 0.9064] | 221560 |
| counter | topk | 3 | 0.9572 | [0.9563, 0.9580] | 221560 |
| counter | topk | 5 | 0.9804 | [0.9798, 0.9810] | 221560 |
| counter | topk | 10 | 0.9956 | [0.9954, 0.9959] | 221560 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 221560 |
| counter | topk_plus_confusion_graph | 1 | 0.8648 | [0.8634, 0.8663] | 221560 |
| counter | topk_plus_confusion_graph | 2 | 0.9590 | [0.9582, 0.9599] | 221560 |
| counter | topk_plus_confusion_graph | 3 | 0.9904 | [0.9900, 0.9908] | 221560 |
| counter | topk_plus_confusion_graph | 5 | 0.9989 | [0.9987, 0.9990] | 221560 |
| desk | topk | 1 | 0.8413 | [0.8404, 0.8422] | 655163 |
| desk | topk | 2 | 0.9549 | [0.9544, 0.9554] | 655163 |
| desk | topk | 3 | 0.9835 | [0.9832, 0.9838] | 655163 |
| desk | topk | 5 | 0.9973 | [0.9971, 0.9974] | 655163 |
| desk | topk | 10 | 0.9998 | [0.9997, 0.9998] | 655163 |
| desk | topk | 20 | 1.0000 | [1.0000, 1.0000] | 655163 |
| desk | topk_plus_confusion_graph | 1 | 0.8853 | [0.8845, 0.8861] | 655163 |
| desk | topk_plus_confusion_graph | 2 | 0.9699 | [0.9695, 0.9703] | 655163 |
| desk | topk_plus_confusion_graph | 3 | 0.9897 | [0.9895, 0.9900] | 655163 |
| desk | topk_plus_confusion_graph | 5 | 0.9987 | [0.9986, 0.9988] | 655163 |
| sink | topk | 1 | 0.7829 | [0.7803, 0.7854] | 102489 |
| sink | topk | 2 | 0.8846 | [0.8827, 0.8866] | 102489 |
| sink | topk | 3 | 0.9292 | [0.9276, 0.9308] | 102489 |
| sink | topk | 5 | 0.9601 | [0.9589, 0.9613] | 102489 |
| sink | topk | 10 | 0.9843 | [0.9835, 0.9850] | 102489 |
| sink | topk | 20 | 1.0000 | [1.0000, 1.0000] | 102489 |
| sink | topk_plus_confusion_graph | 1 | 0.8652 | [0.8631, 0.8673] | 102489 |
| sink | topk_plus_confusion_graph | 2 | 0.9567 | [0.9555, 0.9580] | 102489 |
| sink | topk_plus_confusion_graph | 3 | 0.9728 | [0.9718, 0.9738] | 102489 |
| sink | topk_plus_confusion_graph | 5 | 0.9895 | [0.9889, 0.9902] | 102489 |
| cabinet | topk | 1 | 0.7476 | [0.7468, 0.7483] | 1399895 |
| cabinet | topk | 2 | 0.9116 | [0.9111, 0.9120] | 1399895 |
| cabinet | topk | 5 | 0.9887 | [0.9885, 0.9889] | 1399895 |
| cabinet | topk_plus_confusion_graph | 1 | 0.7732 | [0.7725, 0.7739] | 1399895 |
| cabinet | topk_plus_confusion_graph | 2 | 0.9169 | [0.9164, 0.9174] | 1399895 |
| cabinet | topk_plus_confusion_graph | 5 | 0.9890 | [0.9888, 0.9892] | 1399895 |
| shower curtain | topk | 1 | 0.7974 | [0.7954, 0.7994] | 158208 |
| shower curtain | topk | 2 | 0.8689 | [0.8673, 0.8706] | 158208 |
| shower curtain | topk | 5 | 0.9281 | [0.9268, 0.9294] | 158208 |
| shower curtain | topk_plus_confusion_graph | 1 | 0.9493 | [0.9482, 0.9503] | 158208 |
| shower curtain | topk_plus_confusion_graph | 2 | 0.9951 | [0.9948, 0.9955] | 158208 |
| shower curtain | topk_plus_confusion_graph | 5 | 0.9994 | [0.9993, 0.9995] | 158208 |
| door | topk | 1 | 0.7933 | [0.7927, 0.7939] | 1600191 |
| door | topk | 2 | 0.9615 | [0.9612, 0.9618] | 1600191 |
| door | topk | 5 | 0.9970 | [0.9969, 0.9971] | 1600191 |
| door | topk_plus_confusion_graph | 1 | 0.9078 | [0.9073, 0.9082] | 1600191 |
| door | topk_plus_confusion_graph | 2 | 0.9950 | [0.9949, 0.9951] | 1600191 |
| door | topk_plus_confusion_graph | 5 | 0.9997 | [0.9997, 0.9997] | 1600191 |

## Key Readout Headroom

- base mIoU: `0.7086`
- base picture IoU: `0.3582`
- base picture -> wall fraction: `0.4783`
- best non-base mIoU variant: `oracle_top10` (`0.9921`, delta `+0.2835`)
- best non-base picture variant: `oracle_graph_top5` (`0.9922`, delta `+0.6339`)

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
