# Oracle Actionability Analysis

## Setup
- config: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/config.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- weak classes: `picture,counter,door`
- class pairs: `picture:wall,door:wall,counter:cabinet`
- train batches seen: 18
- val batches seen: 312

## Aggregate Variants

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle_top10 | 0.9934 | +0.1962 | 0.9891 | +0.5553 |
| oracle_graph_top5 | 0.9774 | +0.1802 | 0.9922 | +0.5584 |
| oracle_top5 | 0.9744 | +0.1771 | 0.9567 | +0.5229 |
| oracle_graph_top3 | 0.9574 | +0.1601 | 0.9851 | +0.5513 |
| oracle_top3 | 0.9502 | +0.1530 | 0.9064 | +0.4725 |
| oracle_graph_top2 | 0.9320 | +0.1348 | 0.9795 | +0.5457 |
| oracle_top2 | 0.9165 | +0.1193 | 0.8304 | +0.3966 |
| oracle_graph_top1 | 0.8412 | +0.0440 | 0.9695 | +0.5356 |
| base | 0.7972 | +0.0000 | 0.4338 | +0.0000 |
| prior_alpha0p25 | 0.7966 | -0.0006 | 0.4271 | -0.0067 |
| prior_alpha0p5 | 0.7956 | -0.0016 | 0.4155 | -0.0183 |
| prior_alpha0p75 | 0.7939 | -0.0033 | 0.3968 | -0.0370 |

## Top-K Hit Rates

| class | kind | K | hit rate | 95% CI | target count |
| --- | --- | ---: | ---: | ---: | ---: |
| picture | topk | 1 | 0.5937 | [0.5916, 0.5958] | 207442 |
| picture | topk | 2 | 0.8525 | [0.8509, 0.8540] | 207442 |
| picture | topk | 3 | 0.9213 | [0.9201, 0.9225] | 207442 |
| picture | topk | 5 | 0.9643 | [0.9635, 0.9651] | 207442 |
| picture | topk | 10 | 0.9917 | [0.9913, 0.9921] | 207442 |
| picture | topk | 20 | 1.0000 | [1.0000, 1.0000] | 207442 |
| picture | topk_plus_confusion_graph | 1 | 0.9893 | [0.9889, 0.9897] | 207442 |
| picture | topk_plus_confusion_graph | 2 | 0.9989 | [0.9988, 0.9990] | 207442 |
| picture | topk_plus_confusion_graph | 3 | 0.9998 | [0.9997, 0.9998] | 207442 |
| picture | topk_plus_confusion_graph | 5 | 1.0000 | [1.0000, 1.0000] | 207442 |
| counter | topk | 1 | 0.8251 | [0.8235, 0.8267] | 221560 |
| counter | topk | 2 | 0.9117 | [0.9105, 0.9129] | 221560 |
| counter | topk | 3 | 0.9399 | [0.9389, 0.9409] | 221560 |
| counter | topk | 5 | 0.9649 | [0.9641, 0.9656] | 221560 |
| counter | topk | 10 | 0.9894 | [0.9890, 0.9898] | 221560 |
| counter | topk | 20 | 1.0000 | [1.0000, 1.0000] | 221560 |
| counter | topk_plus_confusion_graph | 1 | 0.8947 | [0.8934, 0.8960] | 221560 |
| counter | topk_plus_confusion_graph | 2 | 0.9586 | [0.9577, 0.9594] | 221560 |
| counter | topk_plus_confusion_graph | 3 | 0.9653 | [0.9645, 0.9660] | 221560 |
| counter | topk_plus_confusion_graph | 5 | 0.9760 | [0.9754, 0.9767] | 221560 |
| door | topk | 1 | 0.8792 | [0.8786, 0.8797] | 1600191 |
| door | topk | 2 | 0.9625 | [0.9622, 0.9628] | 1600191 |
| door | topk | 5 | 0.9963 | [0.9962, 0.9964] | 1600191 |
| door | topk_plus_confusion_graph | 1 | 0.9646 | [0.9643, 0.9649] | 1600191 |
| door | topk_plus_confusion_graph | 2 | 0.9951 | [0.9949, 0.9952] | 1600191 |
| door | topk_plus_confusion_graph | 5 | 0.9998 | [0.9998, 0.9998] | 1600191 |

## Key Readout Headroom

- base mIoU: `0.7972`
- base picture IoU: `0.4338`
- base picture -> wall fraction: `0.3956`
- best non-base mIoU variant: `oracle_top10` (`0.9934`, delta `+0.1962`)
- best non-base picture variant: `oracle_graph_top5` (`0.9922`, delta `+0.5584`)

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
