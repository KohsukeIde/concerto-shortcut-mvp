# PTv3 v1.5.1 Oracle Actionability Analysis

## Setup
- official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- segment key: `segment20`
- weak classes: `picture,counter,desk,sink,cabinet,shower curtain,door`
- class pairs: `picture:wall,counter:cabinet,door:wall`
- train scenes seen: 12
- val scenes seen: 12

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9927 | +0.1973 | 1.0000 | +0.0912 |
| oracle_graph_top5 | 0.9737 | +0.1783 | 1.0000 | +0.0912 |
| oracle_top5 | 0.9731 | +0.1777 | 1.0000 | +0.0912 |
| oracle_graph_top3 | 0.9450 | +0.1497 | 1.0000 | +0.0912 |
| oracle_top3 | 0.9436 | +0.1482 | 1.0000 | +0.0912 |
| oracle_graph_top2 | 0.9220 | +0.1267 | 1.0000 | +0.0912 |
| oracle_top2 | 0.9185 | +0.1231 | 1.0000 | +0.0912 |
| oracle_graph_top1 | 0.8144 | +0.0191 | 1.0000 | +0.0912 |
| base | 0.7954 | +0.0000 | 0.9088 | +0.0000 |
| prior_alpha0p25 | 0.7873 | -0.0080 | 0.8868 | -0.0220 |
| bias_unweighted | 0.7686 | -0.0267 | 0.5981 | -0.3106 |
| prior_alpha0p5 | 0.7683 | -0.0271 | 0.8701 | -0.0387 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.9980 | [0.9942, 1.0000] | 509 |
| picture | topk | 2 | 1.0000 | [1.0000, 1.0000] | 509 |
| picture | topk | 3 | 1.0000 | [1.0000, 1.0000] | 509 |
| picture | topk | 5 | 1.0000 | [1.0000, 1.0000] | 509 |
| picture | topk | 10 | 1.0000 | [1.0000, 1.0000] | 509 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 509 |
| picture | topk_plus_confusion_graph | 1 | 1.0000 | [1.0000, 1.0000] | 509 |
| picture | topk_plus_confusion_graph | 2 | 1.0000 | [1.0000, 1.0000] | 509 |
| picture | topk_plus_confusion_graph | 3 | 1.0000 | [1.0000, 1.0000] | 509 |
| picture | topk_plus_confusion_graph | 5 | 1.0000 | [1.0000, 1.0000] | 509 |
| counter | topk | 1 | 0.8402 | [0.8327, 0.8476] | 9347 |
| counter | topk | 2 | 0.9168 | [0.9112, 0.9224] | 9347 |
| counter | topk | 3 | 0.9436 | [0.9389, 0.9483] | 9347 |
| counter | topk | 5 | 0.9580 | [0.9539, 0.9620] | 9347 |
| counter | topk | 10 | 0.9827 | [0.9800, 0.9853] | 9347 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 9347 |
| counter | topk_plus_confusion_graph | 1 | 0.8826 | [0.8761, 0.8892] | 9347 |
| counter | topk_plus_confusion_graph | 2 | 0.9310 | [0.9259, 0.9361] | 9347 |
| counter | topk_plus_confusion_graph | 3 | 0.9512 | [0.9468, 0.9556] | 9347 |
| counter | topk_plus_confusion_graph | 5 | 0.9607 | [0.9568, 0.9647] | 9347 |
| desk | topk | 1 | 0.8864 | [0.8820, 0.8907] | 20446 |
| desk | topk | 2 | 0.9578 | [0.9551, 0.9606] | 20446 |
| desk | topk | 3 | 0.9682 | [0.9658, 0.9706] | 20446 |
| desk | topk | 5 | 0.9876 | [0.9861, 0.9891] | 20446 |
| desk | topk | 10 | 1.0000 | [1.0000, 1.0000] | 20446 |
| desk | topk | 20 | 1.0000 | [1.0000, 1.0000] | 20446 |
| desk | topk_plus_confusion_graph | 1 | 0.8864 | [0.8820, 0.8907] | 20446 |
| desk | topk_plus_confusion_graph | 2 | 0.9578 | [0.9551, 0.9606] | 20446 |
| desk | topk_plus_confusion_graph | 3 | 0.9682 | [0.9658, 0.9706] | 20446 |
| desk | topk_plus_confusion_graph | 5 | 0.9876 | [0.9861, 0.9891] | 20446 |
| sink | topk | 1 | 0.9087 | [0.8950, 0.9223] | 1708 |
| sink | topk | 2 | 0.9789 | [0.9721, 0.9857] | 1708 |
| sink | topk | 3 | 0.9924 | [0.9883, 0.9965] | 1708 |
| sink | topk | 5 | 0.9959 | [0.9929, 0.9989] | 1708 |
| sink | topk | 10 | 1.0000 | [1.0000, 1.0000] | 1708 |
| sink | topk | 20 | 1.0000 | [1.0000, 1.0000] | 1708 |
| sink | topk_plus_confusion_graph | 1 | 0.9087 | [0.8950, 0.9223] | 1708 |
| sink | topk_plus_confusion_graph | 2 | 0.9789 | [0.9721, 0.9857] | 1708 |
| sink | topk_plus_confusion_graph | 3 | 0.9924 | [0.9883, 0.9965] | 1708 |
| sink | topk_plus_confusion_graph | 5 | 0.9959 | [0.9929, 0.9989] | 1708 |
| cabinet | topk | 1 | 0.5699 | [0.5650, 0.5749] | 38563 |
| cabinet | topk | 2 | 0.6956 | [0.6910, 0.7002] | 38563 |
| cabinet | topk | 5 | 0.8724 | [0.8691, 0.8757] | 38563 |
| cabinet | topk_plus_confusion_graph | 1 | 0.5826 | [0.5777, 0.5875] | 38563 |
| cabinet | topk_plus_confusion_graph | 2 | 0.7017 | [0.6971, 0.7063] | 38563 |
| cabinet | topk_plus_confusion_graph | 5 | 0.8727 | [0.8694, 0.8761] | 38563 |
| shower curtain | topk | 1 | 0.9662 | [0.9600, 0.9724] | 3254 |
| shower curtain | topk | 2 | 0.9779 | [0.9728, 0.9829] | 3254 |
| shower curtain | topk | 5 | 0.9972 | [0.9954, 0.9990] | 3254 |
| shower curtain | topk_plus_confusion_graph | 1 | 0.9662 | [0.9600, 0.9724] | 3254 |
| shower curtain | topk_plus_confusion_graph | 2 | 0.9779 | [0.9728, 0.9829] | 3254 |
| shower curtain | topk_plus_confusion_graph | 5 | 0.9972 | [0.9954, 0.9990] | 3254 |
| door | topk | 1 | 0.8708 | [0.8682, 0.8735] | 60477 |
| door | topk | 2 | 0.9760 | [0.9748, 0.9772] | 60477 |
| door | topk | 3 | 0.9958 | [0.9953, 0.9963] | 60477 |
| door | topk | 5 | 0.9981 | [0.9977, 0.9984] | 60477 |
| door | topk | 10 | 0.9986 | [0.9983, 0.9989] | 60477 |
| door | topk | 20 | 1.0000 | [1.0000, 1.0000] | 60477 |
| door | topk_plus_confusion_graph | 1 | 0.9900 | [0.9892, 0.9908] | 60477 |
| door | topk_plus_confusion_graph | 2 | 1.0000 | [0.9999, 1.0000] | 60477 |
| door | topk_plus_confusion_graph | 3 | 1.0000 | [1.0000, 1.0000] | 60477 |
| door | topk_plus_confusion_graph | 5 | 1.0000 | [1.0000, 1.0000] | 60477 |

## Key Readout Headroom

- base mIoU: `0.7954`
- base picture IoU: `0.9088`
- base picture -> wall fraction: `0.0020`
- best non-base mIoU variant: `oracle_top10` (`0.9927`, delta `+0.1973`)
- best non-base picture variant: `oracle_top2` (`1.0000`, delta `+0.0912`)

## Output Files

- `oracle_topk_hit_rates.csv`
- `oracle_variants.csv`
- `oracle_confusion_distribution.csv`
- `oracle_top3_distribution.csv`
- `oracle_feature_geometry.csv`
- `oracle_pair_probe_train.csv`
- `metadata.json`
