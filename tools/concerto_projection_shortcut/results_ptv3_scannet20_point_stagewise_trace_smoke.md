# PTv3 v1.5.1 Point-Level Stage-Wise Trace

## Setup
- official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- segment key: `segment20`
- train scenes seen: 8
- val scenes seen: 8
- train class counts: {'wall': 60000, 'cabinet': 60000, 'door': 47434, 'picture': 4373, 'counter': 5324}
- val class counts: {'wall': 60000, 'cabinet': 38591, 'door': 40168, 'picture': 0, 'counter': 9359}
- bootstrap iters: 100

## Results

| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| counter_vs_cabinet | point_feature | unweighted | 0.9329 | [0.9304, 0.9362] | 0.9660 | [0.9641, 0.9690] | 9359/38591 |
| counter_vs_cabinet | point_feature | balanced | 0.9319 | [0.9295, 0.9346] | 0.9605 | [0.9578, 0.9628] | 9359/38591 |
| counter_vs_cabinet | point_feature | weighted | 0.9328 | [0.9293, 0.9354] | 0.9659 | [0.9632, 0.9678] | 9359/38591 |
| counter_vs_cabinet | linear_logits | unweighted | 0.9352 | [0.9322, 0.9382] | 0.9787 | [0.9770, 0.9803] | 9359/38591 |
| counter_vs_cabinet | linear_logits | balanced | 0.9407 | [0.9379, 0.9432] | 0.9798 | [0.9781, 0.9811] | 9359/38591 |
| counter_vs_cabinet | linear_logits | weighted | 0.9451 | [0.9418, 0.9475] | 0.9813 | [0.9797, 0.9827] | 9359/38591 |
| counter_vs_cabinet | linear_logits | direct_pair_margin | 0.9598 | [0.9574, 0.9620] | 0.9842 | [0.9827, 0.9857] | 9359/38591 |
| door_vs_wall | point_feature | unweighted | 0.9137 | [0.9122, 0.9154] | 0.9241 | [0.9224, 0.9261] | 40168/60000 |
| door_vs_wall | point_feature | balanced | 0.9133 | [0.9118, 0.9145] | 0.9245 | [0.9225, 0.9261] | 40168/60000 |
| door_vs_wall | point_feature | weighted | 0.9131 | [0.9110, 0.9149] | 0.9246 | [0.9225, 0.9267] | 40168/60000 |
| door_vs_wall | linear_logits | unweighted | 0.9039 | [0.9026, 0.9059] | 0.9410 | [0.9399, 0.9426] | 40168/60000 |
| door_vs_wall | linear_logits | balanced | 0.9038 | [0.9019, 0.9054] | 0.9426 | [0.9412, 0.9439] | 40168/60000 |
| door_vs_wall | linear_logits | weighted | 0.9034 | [0.9021, 0.9052] | 0.9426 | [0.9411, 0.9439] | 40168/60000 |
| door_vs_wall | linear_logits | direct_pair_margin | 0.9102 | [0.9085, 0.9121] | 0.9545 | [0.9533, 0.9562] | 40168/60000 |

## Interpretation Guide
- `point_feature` is the official PTv3 decoder feature before the segmentation head.
- `linear_logits` with `unweighted`/`balanced`/`weighted` fits a binary probe on the fixed 20-way logits.
- `linear_logits` with `direct_pair_margin` uses the fixed logit margin `logit(positive_class) - logit(negative_class)` without refitting a pair probe.
- The companion confusion CSV reports full 20-way predictions for the validation rows belonging to each class pair.
- Train and validation both use the official deterministic validation-style transform so the trace is not affected by train-time random augmentations.
