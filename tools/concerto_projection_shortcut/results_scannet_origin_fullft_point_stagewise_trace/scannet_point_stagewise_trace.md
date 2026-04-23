# ScanNet Point-Level Stage-Wise Trace

## Setup
- linear config: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/config.py`
- linear weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_best.pth`
- data root: `data/scannet`
- train batches seen: 256
- val batches seen: 128
- train class counts: {'wall': 60000, 'cabinet': 60000, 'door': 60000, 'picture': 60000, 'counter': 60000}
- val class counts: {'wall': 60000, 'cabinet': 60000, 'door': 60000, 'picture': 60000, 'counter': 56679}
- bootstrap iters: 100

## Results

| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| picture_vs_wall | point_feature | unweighted | 0.7175 | [0.7154, 0.7199] | 0.8286 | [0.8264, 0.8310] | 60000/60000 |
| picture_vs_wall | point_feature | balanced | 0.7175 | [0.7154, 0.7195] | 0.8287 | [0.8260, 0.8309] | 60000/60000 |
| picture_vs_wall | point_feature | weighted | 0.7175 | [0.7161, 0.7198] | 0.8286 | [0.8262, 0.8312] | 60000/60000 |
| picture_vs_wall | linear_logits | unweighted | 0.7206 | [0.7189, 0.7227] | 0.8568 | [0.8549, 0.8589] | 60000/60000 |
| picture_vs_wall | linear_logits | balanced | 0.7205 | [0.7186, 0.7223] | 0.8563 | [0.8549, 0.8582] | 60000/60000 |
| picture_vs_wall | linear_logits | weighted | 0.7206 | [0.7185, 0.7232] | 0.8568 | [0.8550, 0.8596] | 60000/60000 |
| picture_vs_wall | linear_logits | direct_pair_margin | 0.7052 | [0.7034, 0.7072] | 0.8271 | [0.8248, 0.8294] | 60000/60000 |
| door_vs_wall | point_feature | unweighted | 0.9548 | [0.9538, 0.9562] | 0.9890 | [0.9884, 0.9895] | 60000/60000 |
| door_vs_wall | point_feature | balanced | 0.9548 | [0.9537, 0.9558] | 0.9890 | [0.9885, 0.9896] | 60000/60000 |
| door_vs_wall | point_feature | weighted | 0.9548 | [0.9535, 0.9557] | 0.9890 | [0.9884, 0.9895] | 60000/60000 |
| door_vs_wall | linear_logits | unweighted | 0.9537 | [0.9524, 0.9547] | 0.9882 | [0.9877, 0.9888] | 60000/60000 |
| door_vs_wall | linear_logits | balanced | 0.9537 | [0.9524, 0.9549] | 0.9883 | [0.9877, 0.9889] | 60000/60000 |
| door_vs_wall | linear_logits | weighted | 0.9537 | [0.9524, 0.9545] | 0.9882 | [0.9876, 0.9887] | 60000/60000 |
| door_vs_wall | linear_logits | direct_pair_margin | 0.9528 | [0.9515, 0.9540] | 0.9989 | [0.9988, 0.9990] | 60000/60000 |
| counter_vs_cabinet | point_feature | unweighted | 0.9510 | [0.9497, 0.9520] | 0.9719 | [0.9708, 0.9729] | 56679/60000 |
| counter_vs_cabinet | point_feature | balanced | 0.9510 | [0.9501, 0.9525] | 0.9719 | [0.9710, 0.9730] | 56679/60000 |
| counter_vs_cabinet | point_feature | weighted | 0.9510 | [0.9498, 0.9525] | 0.9719 | [0.9706, 0.9730] | 56679/60000 |
| counter_vs_cabinet | linear_logits | unweighted | 0.9520 | [0.9506, 0.9531] | 0.9734 | [0.9726, 0.9743] | 56679/60000 |
| counter_vs_cabinet | linear_logits | balanced | 0.9520 | [0.9511, 0.9532] | 0.9734 | [0.9724, 0.9744] | 56679/60000 |
| counter_vs_cabinet | linear_logits | weighted | 0.9520 | [0.9507, 0.9532] | 0.9734 | [0.9722, 0.9745] | 56679/60000 |
| counter_vs_cabinet | linear_logits | direct_pair_margin | 0.9505 | [0.9493, 0.9516] | 0.9779 | [0.9770, 0.9788] | 56679/60000 |

## Interpretation Guide
- `point_feature` is the Concerto backbone/decoder point feature returned by the ScanNet semseg model before the segmentation head.
- `linear_logits` with `unweighted`/`balanced`/`weighted` fits a binary probe on the 20-way linear logits.
- `linear_logits` with `direct_pair_margin` uses the fixed logit margin `logit(positive_class) - logit(negative_class)` without refitting a pair probe.
- The companion confusion CSV reports full 20-way predictions for the validation rows belonging to each class pair.
- Train and validation use the deterministic ScanNet validation-style transform so the stage trace is not affected by train-time random augmentations.
