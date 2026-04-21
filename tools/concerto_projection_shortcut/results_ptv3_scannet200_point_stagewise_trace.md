# PTv3 v1.5.1 Point-Level Stage-Wise Trace

## Setup
- official root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/tmp/Pointcept-v1.5.1`
- config: `configs/scannet200/semseg-pt-v3m1-0-base.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/ptv3/scannet200-semseg-pt-v3m1-0-base/model/model_best.pth`
- data root: `/groups/qgah50055/ide/3d-sans-3dscans/scannet`
- segment key: `segment200`
- train scenes seen: 256
- val scenes seen: 128
- train class counts: {'wall': 60000, 'door': 60000, 'cabinet': 60000, 'picture': 60000, 'counter': 31430}
- val class counts: {'wall': 60000, 'door': 60000, 'cabinet': 60000, 'picture': 26160, 'counter': 34986}
- bootstrap iters: 100

## Results

| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| picture_vs_wall | point_feature | unweighted | 0.9098 | [0.9081, 0.9121] | 0.9604 | [0.9591, 0.9618] | 26160/60000 |
| picture_vs_wall | point_feature | balanced | 0.9097 | [0.9075, 0.9120] | 0.9604 | [0.9591, 0.9618] | 26160/60000 |
| picture_vs_wall | point_feature | weighted | 0.9098 | [0.9079, 0.9127] | 0.9604 | [0.9592, 0.9619] | 26160/60000 |
| picture_vs_wall | linear_logits | unweighted | 0.9094 | [0.9071, 0.9115] | 0.9537 | [0.9518, 0.9554] | 26160/60000 |
| picture_vs_wall | linear_logits | balanced | 0.9094 | [0.9068, 0.9119] | 0.9537 | [0.9518, 0.9555] | 26160/60000 |
| picture_vs_wall | linear_logits | weighted | 0.9094 | [0.9073, 0.9113] | 0.9537 | [0.9520, 0.9556] | 26160/60000 |
| picture_vs_wall | linear_logits | direct_pair_margin | 0.8613 | [0.8582, 0.8641] | 0.9697 | [0.9684, 0.9710] | 26160/60000 |
| counter_vs_cabinet | point_feature | unweighted | 0.9605 | [0.9592, 0.9616] | 0.9926 | [0.9922, 0.9930] | 34986/60000 |
| counter_vs_cabinet | point_feature | balanced | 0.9575 | [0.9563, 0.9588] | 0.9920 | [0.9916, 0.9925] | 34986/60000 |
| counter_vs_cabinet | point_feature | weighted | 0.9578 | [0.9567, 0.9591] | 0.9920 | [0.9916, 0.9925] | 34986/60000 |
| counter_vs_cabinet | linear_logits | unweighted | 0.9629 | [0.9618, 0.9639] | 0.9935 | [0.9930, 0.9939] | 34986/60000 |
| counter_vs_cabinet | linear_logits | balanced | 0.9649 | [0.9636, 0.9661] | 0.9933 | [0.9928, 0.9938] | 34986/60000 |
| counter_vs_cabinet | linear_logits | weighted | 0.9652 | [0.9639, 0.9663] | 0.9935 | [0.9930, 0.9939] | 34986/60000 |
| counter_vs_cabinet | linear_logits | direct_pair_margin | 0.9729 | [0.9718, 0.9738] | 0.9956 | [0.9954, 0.9959] | 34986/60000 |
| door_vs_wall | point_feature | unweighted | 0.8973 | [0.8959, 0.8988] | 0.9288 | [0.9273, 0.9303] | 60000/60000 |
| door_vs_wall | point_feature | balanced | 0.8973 | [0.8956, 0.8989] | 0.9288 | [0.9278, 0.9299] | 60000/60000 |
| door_vs_wall | point_feature | weighted | 0.8973 | [0.8957, 0.8987] | 0.9288 | [0.9272, 0.9301] | 60000/60000 |
| door_vs_wall | linear_logits | unweighted | 0.8898 | [0.8882, 0.8914] | 0.9269 | [0.9256, 0.9282] | 60000/60000 |
| door_vs_wall | linear_logits | balanced | 0.8898 | [0.8882, 0.8914] | 0.9269 | [0.9255, 0.9281] | 60000/60000 |
| door_vs_wall | linear_logits | weighted | 0.8898 | [0.8881, 0.8913] | 0.9269 | [0.9255, 0.9282] | 60000/60000 |
| door_vs_wall | linear_logits | direct_pair_margin | 0.8931 | [0.8914, 0.8949] | 0.9815 | [0.9811, 0.9821] | 60000/60000 |

## Interpretation Guide
- `point_feature` is the official PTv3 decoder feature before the segmentation head.
- `linear_logits` with `unweighted`/`balanced`/`weighted` fits a binary probe on the fixed 20-way logits.
- `linear_logits` with `direct_pair_margin` uses the fixed logit margin `logit(positive_class) - logit(negative_class)` without refitting a pair probe.
- The companion confusion CSV reports full 20-way predictions for the validation rows belonging to each class pair.
- Train and validation both use the official deterministic validation-style transform so the trace is not affected by train-time random augmentations.
