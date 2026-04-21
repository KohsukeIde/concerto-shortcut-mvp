# PTv3 v1.5.1 Point-Level Stage-Wise Trace

## Setup
- official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- config: `configs/scannet/semseg-pt-v3m1-0-base.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- segment key: `segment20`
- train scenes seen: 256
- val scenes seen: 128
- train class counts: {'wall': 60000, 'cabinet': 60000, 'door': 60000, 'picture': 60000, 'counter': 60000}
- val class counts: {'wall': 60000, 'cabinet': 60000, 'door': 60000, 'picture': 32517, 'counter': 60000}
- bootstrap iters: 100

## Results

| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| picture_vs_wall | point_feature | unweighted | 0.9626 | [0.9613, 0.9639] | 0.9878 | [0.9869, 0.9885] | 32517/60000 |
| picture_vs_wall | point_feature | balanced | 0.9626 | [0.9614, 0.9638] | 0.9878 | [0.9868, 0.9886] | 32517/60000 |
| picture_vs_wall | point_feature | weighted | 0.9626 | [0.9615, 0.9637] | 0.9878 | [0.9871, 0.9887] | 32517/60000 |
| picture_vs_wall | linear_logits | unweighted | 0.9529 | [0.9514, 0.9540] | 0.9861 | [0.9854, 0.9869] | 32517/60000 |
| picture_vs_wall | linear_logits | balanced | 0.9529 | [0.9513, 0.9542] | 0.9861 | [0.9853, 0.9870] | 32517/60000 |
| picture_vs_wall | linear_logits | weighted | 0.9529 | [0.9517, 0.9544] | 0.9861 | [0.9852, 0.9869] | 32517/60000 |
| picture_vs_wall | linear_logits | direct_pair_margin | 0.8892 | [0.8870, 0.8915] | 0.9889 | [0.9885, 0.9894] | 32517/60000 |
| counter_vs_cabinet | point_feature | unweighted | 0.9581 | [0.9569, 0.9591] | 0.9847 | [0.9840, 0.9852] | 60000/60000 |
| counter_vs_cabinet | point_feature | balanced | 0.9582 | [0.9569, 0.9594] | 0.9847 | [0.9841, 0.9853] | 60000/60000 |
| counter_vs_cabinet | point_feature | weighted | 0.9581 | [0.9570, 0.9594] | 0.9847 | [0.9838, 0.9853] | 60000/60000 |
| counter_vs_cabinet | linear_logits | unweighted | 0.9701 | [0.9693, 0.9711] | 0.9881 | [0.9875, 0.9886] | 60000/60000 |
| counter_vs_cabinet | linear_logits | balanced | 0.9701 | [0.9692, 0.9709] | 0.9881 | [0.9876, 0.9886] | 60000/60000 |
| counter_vs_cabinet | linear_logits | weighted | 0.9701 | [0.9693, 0.9711] | 0.9881 | [0.9876, 0.9887] | 60000/60000 |
| counter_vs_cabinet | linear_logits | direct_pair_margin | 0.9709 | [0.9701, 0.9719] | 0.9905 | [0.9901, 0.9910] | 60000/60000 |
| door_vs_wall | point_feature | unweighted | 0.9361 | [0.9347, 0.9374] | 0.9541 | [0.9530, 0.9556] | 60000/60000 |
| door_vs_wall | point_feature | balanced | 0.9361 | [0.9348, 0.9371] | 0.9541 | [0.9530, 0.9553] | 60000/60000 |
| door_vs_wall | point_feature | weighted | 0.9361 | [0.9350, 0.9374] | 0.9541 | [0.9531, 0.9557] | 60000/60000 |
| door_vs_wall | linear_logits | unweighted | 0.9267 | [0.9249, 0.9280] | 0.9595 | [0.9587, 0.9606] | 60000/60000 |
| door_vs_wall | linear_logits | balanced | 0.9267 | [0.9253, 0.9280] | 0.9595 | [0.9584, 0.9605] | 60000/60000 |
| door_vs_wall | linear_logits | weighted | 0.9267 | [0.9251, 0.9282] | 0.9595 | [0.9582, 0.9607] | 60000/60000 |
| door_vs_wall | linear_logits | direct_pair_margin | 0.9385 | [0.9374, 0.9398] | 0.9683 | [0.9674, 0.9692] | 60000/60000 |

## Interpretation Guide
- `point_feature` is the official PTv3 decoder feature before the segmentation head.
- `linear_logits` with `unweighted`/`balanced`/`weighted` fits a binary probe on the fixed 20-way logits.
- `linear_logits` with `direct_pair_margin` uses the fixed logit margin `logit(positive_class) - logit(negative_class)` without refitting a pair probe.
- The companion confusion CSV reports full 20-way predictions for the validation rows belonging to each class pair.
- Train and validation both use the official deterministic validation-style transform so the trace is not affected by train-time random augmentations.
