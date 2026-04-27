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
- val scenes seen: 312

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9915 | +0.2200 | 0.9950 | +0.6039 |
| oracle_graph_top5 | 0.9700 | +0.1984 | 0.9994 | +0.6084 |
| oracle_top5 | 0.9647 | +0.1931 | 0.9547 | +0.5637 |
| oracle_graph_top3 | 0.9414 | +0.1698 | 0.9948 | +0.6038 |
| oracle_top3 | 0.9280 | +0.1564 | 0.8111 | +0.4201 |
| oracle_graph_top2 | 0.9120 | +0.1404 | 0.9841 | +0.5930 |
| oracle_top2 | 0.8889 | +0.1173 | 0.6763 | +0.2852 |
| oracle_graph_top1 | 0.8175 | +0.0459 | 0.9701 | +0.5790 |
| prior_alpha0p25 | 0.7729 | +0.0013 | 0.3934 | +0.0023 |
| base | 0.7716 | +0.0000 | 0.3910 | +0.0000 |
| multiproto4_adapt_tau0p1_lam0p05 | 0.7716 | -0.0000 | 0.3910 | -0.0000 |
| multiproto4_adapt_tau0p2_lam0p05 | 0.7716 | -0.0000 | 0.3910 | -0.0000 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.4932 | [0.4906, 0.4957] | 149608 |
| picture | topk | 2 | 0.6865 | [0.6842, 0.6889] | 149608 |
| picture | topk | 3 | 0.8152 | [0.8133, 0.8172] | 149608 |
| picture | topk | 5 | 0.9553 | [0.9543, 0.9564] | 149608 |
| picture | topk | 10 | 0.9950 | [0.9946, 0.9954] | 149608 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 149608 |
| picture | topk_plus_confusion_graph | 1 | 0.9870 | [0.9864, 0.9876] | 149608 |
| picture | topk_plus_confusion_graph | 2 | 0.9989 | [0.9987, 0.9990] | 149608 |
| picture | topk_plus_confusion_graph | 3 | 0.9998 | [0.9997, 0.9998] | 149608 |
| picture | topk_plus_confusion_graph | 5 | 1.0000 | [1.0000, 1.0000] | 149608 |
| counter | topk | 1 | 0.8212 | [0.8194, 0.8231] | 158167 |
| counter | topk | 2 | 0.8954 | [0.8939, 0.8969] | 158167 |
| counter | topk | 3 | 0.9126 | [0.9112, 0.9140] | 158167 |
| counter | topk | 5 | 0.9397 | [0.9386, 0.9409] | 158167 |
| counter | topk | 10 | 0.9735 | [0.9727, 0.9743] | 158167 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 158167 |
| counter | topk_plus_confusion_graph | 1 | 0.8925 | [0.8910, 0.8940] | 158167 |
| counter | topk_plus_confusion_graph | 2 | 0.9528 | [0.9517, 0.9538] | 158167 |
| counter | topk_plus_confusion_graph | 3 | 0.9617 | [0.9607, 0.9626] | 158167 |
| counter | topk_plus_confusion_graph | 5 | 0.9876 | [0.9871, 0.9882] | 158167 |
| desk | topk | 1 | 0.8720 | [0.8710, 0.8729] | 466678 |
| desk | topk | 2 | 0.9479 | [0.9472, 0.9485] | 466678 |
| desk | topk | 3 | 0.9701 | [0.9696, 0.9705] | 466678 |
| desk | topk | 5 | 0.9837 | [0.9833, 0.9840] | 466678 |
| desk | topk | 10 | 0.9937 | [0.9934, 0.9939] | 466678 |
| desk | topk | 20 | 1.0000 | [1.0000, 1.0000] | 466678 |
| desk | topk_plus_confusion_graph | 1 | 0.8720 | [0.8710, 0.8729] | 466678 |
| desk | topk_plus_confusion_graph | 2 | 0.9479 | [0.9472, 0.9485] | 466678 |
| desk | topk_plus_confusion_graph | 3 | 0.9701 | [0.9696, 0.9705] | 466678 |
| desk | topk_plus_confusion_graph | 5 | 0.9837 | [0.9833, 0.9840] | 466678 |
| sink | topk | 1 | 0.7925 | [0.7894, 0.7955] | 67259 |
| sink | topk | 2 | 0.9076 | [0.9054, 0.9098] | 67259 |
| sink | topk | 3 | 0.9444 | [0.9427, 0.9461] | 67259 |
| sink | topk | 5 | 0.9578 | [0.9563, 0.9593] | 67259 |
| sink | topk | 10 | 0.9928 | [0.9922, 0.9935] | 67259 |
| sink | topk | 20 | 1.0000 | [1.0000, 1.0000] | 67259 |
| sink | topk_plus_confusion_graph | 1 | 0.7925 | [0.7894, 0.7955] | 67259 |
| sink | topk_plus_confusion_graph | 2 | 0.9076 | [0.9054, 0.9098] | 67259 |
| sink | topk_plus_confusion_graph | 3 | 0.9444 | [0.9427, 0.9461] | 67259 |
| sink | topk_plus_confusion_graph | 5 | 0.9578 | [0.9563, 0.9593] | 67259 |
| cabinet | topk | 1 | 0.8281 | [0.8274, 0.8289] | 999311 |
| cabinet | topk | 2 | 0.9132 | [0.9126, 0.9137] | 999311 |
| cabinet | topk | 5 | 0.9776 | [0.9773, 0.9779] | 999311 |
| cabinet | topk_plus_confusion_graph | 1 | 0.8406 | [0.8398, 0.8413] | 999311 |
| cabinet | topk_plus_confusion_graph | 2 | 0.9175 | [0.9169, 0.9180] | 999311 |
| cabinet | topk_plus_confusion_graph | 5 | 0.9776 | [0.9773, 0.9779] | 999311 |
| shower curtain | topk | 1 | 0.7565 | [0.7539, 0.7592] | 101035 |
| shower curtain | topk | 2 | 0.8465 | [0.8443, 0.8488] | 101035 |
| shower curtain | topk | 5 | 0.9724 | [0.9714, 0.9734] | 101035 |
| shower curtain | topk_plus_confusion_graph | 1 | 0.7565 | [0.7539, 0.7592] | 101035 |
| shower curtain | topk_plus_confusion_graph | 2 | 0.8465 | [0.8443, 0.8488] | 101035 |
| shower curtain | topk_plus_confusion_graph | 5 | 0.9724 | [0.9714, 0.9734] | 101035 |
| door | topk | 1 | 0.8560 | [0.8553, 0.8567] | 1086635 |
| door | topk | 2 | 0.9671 | [0.9667, 0.9674] | 1086635 |
| door | topk | 3 | 0.9917 | [0.9915, 0.9919] | 1086635 |
| door | topk | 5 | 0.9979 | [0.9978, 0.9980] | 1086635 |
| door | topk | 10 | 0.9995 | [0.9995, 0.9996] | 1086635 |
| door | topk | 20 | 1.0000 | [1.0000, 1.0000] | 1086635 |
| door | topk_plus_confusion_graph | 1 | 0.9545 | [0.9541, 0.9549] | 1086635 |
| door | topk_plus_confusion_graph | 2 | 0.9961 | [0.9960, 0.9963] | 1086635 |
| door | topk_plus_confusion_graph | 3 | 0.9994 | [0.9994, 0.9995] | 1086635 |
| door | topk_plus_confusion_graph | 5 | 0.9999 | [0.9999, 0.9999] | 1086635 |

## Key Readout Headroom

- base mIoU: `0.7716`
- base picture IoU: `0.3910`
- base picture -> wall fraction: `0.4938`
- best non-base mIoU variant: `oracle_top10` (`0.9915`, delta `+0.2200`)
- best non-base picture variant: `oracle_graph_top5` (`0.9994`, delta `+0.6084`)

## Output Files

- `oracle_topk_hit_rates.csv`
- `oracle_variants.csv`
- `oracle_confusion_distribution.csv`
- `oracle_top3_distribution.csv`
- `oracle_feature_geometry.csv`
- `oracle_pair_probe_train.csv`
- `metadata.json`
