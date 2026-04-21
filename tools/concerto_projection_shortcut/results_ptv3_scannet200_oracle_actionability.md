# PTv3 v1.5.1 Oracle Actionability Analysis

## Setup
- official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- config: `configs/scannet200/semseg-pt-v3m1-0-base.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet200-semseg-pt-v3m1-0-base/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- segment key: `segment200`
- weak classes: `picture,counter,desk,sink,cabinet,door,window,bookshelf`
- class pairs: `picture:wall,counter:cabinet,door:wall`
- train scenes seen: 9
- val scenes seen: 128

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.6009 | +0.3036 | 0.8893 | +0.4269 |
| oracle_graph_top5 | 0.5331 | +0.2358 | 0.9148 | +0.4524 |
| oracle_top5 | 0.5311 | +0.2338 | 0.8388 | +0.3764 |
| oracle_graph_top3 | 0.4669 | +0.1696 | 0.9061 | +0.4437 |
| oracle_top3 | 0.4640 | +0.1667 | 0.7884 | +0.3260 |
| oracle_graph_top2 | 0.4086 | +0.1113 | 0.9029 | +0.4404 |
| oracle_top2 | 0.4055 | +0.1082 | 0.7505 | +0.2881 |
| oracle_graph_top1 | 0.3008 | +0.0035 | 0.8742 | +0.4118 |
| base | 0.2973 | +0.0000 | 0.4624 | +0.0000 |
| pair_probe_top2 | 0.2950 | -0.0023 | 0.1371 | -0.3254 |
| prior_alpha0p25 | 0.2947 | -0.0026 | 0.4463 | -0.0161 |
| prior_alpha0p5 | 0.2723 | -0.0250 | 0.4263 | -0.0361 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.7018 | [0.6962, 0.7073] | 26160 |
| picture | topk | 2 | 0.8461 | [0.8418, 0.8505] | 26160 |
| picture | topk | 3 | 0.8700 | [0.8659, 0.8740] | 26160 |
| picture | topk | 5 | 0.9168 | [0.9134, 0.9201] | 26160 |
| picture | topk | 10 | 0.9632 | [0.9609, 0.9655] | 26160 |
| picture | topk | 20 | 0.9966 | [0.9959, 0.9973] | 26160 |
| picture | topk_plus_confusion_graph | 1 | 0.9685 | [0.9663, 0.9706] | 26160 |
| picture | topk_plus_confusion_graph | 2 | 0.9977 | [0.9971, 0.9983] | 26160 |
| picture | topk_plus_confusion_graph | 3 | 0.9994 | [0.9991, 0.9997] | 26160 |
| picture | topk_plus_confusion_graph | 5 | 0.9998 | [0.9997, 1.0000] | 26160 |
| counter | topk | 1 | 0.8127 | [0.8086, 0.8168] | 35013 |
| counter | topk | 2 | 0.9474 | [0.9451, 0.9498] | 35013 |
| counter | topk | 3 | 0.9729 | [0.9712, 0.9746] | 35013 |
| counter | topk | 5 | 0.9864 | [0.9852, 0.9876] | 35013 |
| counter | topk | 10 | 0.9914 | [0.9905, 0.9924] | 35013 |
| counter | topk | 20 | 0.9948 | [0.9940, 0.9955] | 35013 |
| counter | topk_plus_confusion_graph | 1 | 0.8270 | [0.8231, 0.8310] | 35013 |
| counter | topk_plus_confusion_graph | 2 | 0.9505 | [0.9482, 0.9527] | 35013 |
| counter | topk_plus_confusion_graph | 3 | 0.9763 | [0.9747, 0.9779] | 35013 |
| counter | topk_plus_confusion_graph | 5 | 0.9912 | [0.9902, 0.9922] | 35013 |
| desk | topk | 1 | 0.8472 | [0.8457, 0.8487] | 224011 |
| desk | topk | 2 | 0.9287 | [0.9276, 0.9298] | 224011 |
| desk | topk | 3 | 0.9443 | [0.9433, 0.9452] | 224011 |
| desk | topk | 5 | 0.9550 | [0.9541, 0.9558] | 224011 |
| desk | topk | 10 | 0.9735 | [0.9728, 0.9741] | 224011 |
| desk | topk | 20 | 0.9901 | [0.9897, 0.9905] | 224011 |
| desk | topk_plus_confusion_graph | 1 | 0.8472 | [0.8457, 0.8487] | 224011 |
| desk | topk_plus_confusion_graph | 2 | 0.9287 | [0.9276, 0.9298] | 224011 |
| desk | topk_plus_confusion_graph | 3 | 0.9443 | [0.9433, 0.9452] | 224011 |
| desk | topk_plus_confusion_graph | 5 | 0.9550 | [0.9541, 0.9558] | 224011 |
| sink | topk | 1 | 0.9091 | [0.9060, 0.9123] | 32412 |
| sink | topk | 2 | 0.9684 | [0.9665, 0.9703] | 32412 |
| sink | topk | 3 | 0.9810 | [0.9795, 0.9825] | 32412 |
| sink | topk | 5 | 0.9882 | [0.9870, 0.9893] | 32412 |
| sink | topk | 10 | 0.9949 | [0.9941, 0.9957] | 32412 |
| sink | topk | 20 | 0.9984 | [0.9980, 0.9989] | 32412 |
| sink | topk_plus_confusion_graph | 1 | 0.9091 | [0.9060, 0.9123] | 32412 |
| sink | topk_plus_confusion_graph | 2 | 0.9684 | [0.9665, 0.9703] | 32412 |
| sink | topk_plus_confusion_graph | 3 | 0.9810 | [0.9795, 0.9825] | 32412 |
| sink | topk_plus_confusion_graph | 5 | 0.9882 | [0.9870, 0.9893] | 32412 |
| cabinet | topk | 1 | 0.5540 | [0.5518, 0.5561] | 208384 |
| cabinet | topk | 2 | 0.6802 | [0.6782, 0.6822] | 208384 |
| cabinet | topk | 5 | 0.9011 | [0.8998, 0.9024] | 208384 |
| cabinet | topk_plus_confusion_graph | 1 | 0.6032 | [0.6011, 0.6053] | 208384 |
| cabinet | topk_plus_confusion_graph | 2 | 0.7115 | [0.7095, 0.7134] | 208384 |
| cabinet | topk_plus_confusion_graph | 5 | 0.9145 | [0.9133, 0.9157] | 208384 |
| door | topk | 1 | 0.7942 | [0.7931, 0.7954] | 477659 |
| door | topk | 2 | 0.9335 | [0.9328, 0.9342] | 477659 |
| door | topk | 3 | 0.9744 | [0.9740, 0.9749] | 477659 |
| door | topk | 5 | 0.9868 | [0.9865, 0.9871] | 477659 |
| door | topk | 10 | 0.9954 | [0.9952, 0.9956] | 477659 |
| door | topk | 20 | 0.9995 | [0.9994, 0.9995] | 477659 |
| door | topk_plus_confusion_graph | 1 | 0.8992 | [0.8984, 0.9001] | 477659 |
| door | topk_plus_confusion_graph | 2 | 0.9934 | [0.9932, 0.9937] | 477659 |
| door | topk_plus_confusion_graph | 3 | 0.9989 | [0.9988, 0.9990] | 477659 |
| door | topk_plus_confusion_graph | 5 | 0.9995 | [0.9994, 0.9996] | 477659 |
| window | topk | 1 | 0.8097 | [0.8083, 0.8111] | 296448 |
| window | topk | 2 | 0.9284 | [0.9275, 0.9294] | 296448 |
| window | topk | 5 | 0.9813 | [0.9808, 0.9818] | 296448 |
| window | topk_plus_confusion_graph | 1 | 0.8097 | [0.8083, 0.8111] | 296448 |
| window | topk_plus_confusion_graph | 2 | 0.9284 | [0.9275, 0.9294] | 296448 |
| window | topk_plus_confusion_graph | 5 | 0.9813 | [0.9808, 0.9818] | 296448 |
| bookshelf | topk | 1 | 0.6276 | [0.6261, 0.6291] | 398280 |
| bookshelf | topk | 2 | 0.8902 | [0.8892, 0.8911] | 398280 |
| bookshelf | topk | 5 | 0.9810 | [0.9806, 0.9814] | 398280 |
| bookshelf | topk_plus_confusion_graph | 1 | 0.6276 | [0.6261, 0.6291] | 398280 |
| bookshelf | topk_plus_confusion_graph | 2 | 0.8902 | [0.8892, 0.8911] | 398280 |
| bookshelf | topk_plus_confusion_graph | 5 | 0.9810 | [0.9806, 0.9814] | 398280 |

## Key Readout Headroom

- base mIoU: `0.2973`
- base picture IoU: `0.4624`
- base picture -> wall fraction: `0.2667`
- best non-base mIoU variant: `oracle_top10` (`0.6009`, delta `+0.3036`)
- best non-base picture variant: `oracle_graph_top5` (`0.9148`, delta `+0.4524`)

## Output Files

- `oracle_topk_hit_rates.csv`
- `oracle_variants.csv`
- `oracle_confusion_distribution.csv`
- `oracle_top3_distribution.csv`
- `oracle_feature_geometry.csv`
- `oracle_pair_probe_train.csv`
- `metadata.json`
