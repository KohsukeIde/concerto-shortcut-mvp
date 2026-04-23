# ScanNet Point-Level Stage-Wise Trace

## Setup
- linear config: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/config.py`
- linear weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/model/model_best.pth`
- data root: `data/scannet`
- train batches seen: 256
- val batches seen: 128
- train class counts: {'wall': 60000, 'cabinet': 60000, 'door': 60000, 'picture': 60000, 'counter': 60000}
- val class counts: {'wall': 60000, 'cabinet': 60000, 'door': 60000, 'picture': 60000, 'counter': 56679}
- bootstrap iters: 100

## Results

| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| picture_vs_wall | point_feature | unweighted | 0.6742 | [0.6726, 0.6763] | 0.7756 | [0.7732, 0.7789] | 60000/60000 |
| picture_vs_wall | point_feature | balanced | 0.6742 | [0.6722, 0.6764] | 0.7757 | [0.7729, 0.7781] | 60000/60000 |
| picture_vs_wall | point_feature | weighted | 0.6742 | [0.6730, 0.6764] | 0.7756 | [0.7733, 0.7787] | 60000/60000 |
| picture_vs_wall | linear_logits | unweighted | 0.6734 | [0.6716, 0.6754] | 0.7941 | [0.7921, 0.7962] | 60000/60000 |
| picture_vs_wall | linear_logits | balanced | 0.6734 | [0.6718, 0.6754] | 0.7942 | [0.7914, 0.7968] | 60000/60000 |
| picture_vs_wall | linear_logits | weighted | 0.6734 | [0.6716, 0.6754] | 0.7941 | [0.7922, 0.7969] | 60000/60000 |
| picture_vs_wall | linear_logits | direct_pair_margin | 0.6485 | [0.6471, 0.6498] | 0.7222 | [0.7192, 0.7247] | 60000/60000 |
| door_vs_wall | point_feature | unweighted | 0.9550 | [0.9540, 0.9562] | 0.9854 | [0.9849, 0.9860] | 60000/60000 |
| door_vs_wall | point_feature | balanced | 0.9550 | [0.9538, 0.9562] | 0.9854 | [0.9848, 0.9860] | 60000/60000 |
| door_vs_wall | point_feature | weighted | 0.9550 | [0.9538, 0.9563] | 0.9854 | [0.9849, 0.9860] | 60000/60000 |
| door_vs_wall | linear_logits | unweighted | 0.9598 | [0.9584, 0.9607] | 0.9871 | [0.9865, 0.9876] | 60000/60000 |
| door_vs_wall | linear_logits | balanced | 0.9597 | [0.9588, 0.9608] | 0.9871 | [0.9866, 0.9876] | 60000/60000 |
| door_vs_wall | linear_logits | weighted | 0.9598 | [0.9590, 0.9606] | 0.9871 | [0.9866, 0.9876] | 60000/60000 |
| door_vs_wall | linear_logits | direct_pair_margin | 0.9445 | [0.9430, 0.9456] | 0.9839 | [0.9832, 0.9845] | 60000/60000 |
| counter_vs_cabinet | point_feature | unweighted | 0.9576 | [0.9566, 0.9584] | 0.9744 | [0.9734, 0.9753] | 56679/60000 |
| counter_vs_cabinet | point_feature | balanced | 0.9576 | [0.9566, 0.9588] | 0.9744 | [0.9736, 0.9756] | 56679/60000 |
| counter_vs_cabinet | point_feature | weighted | 0.9576 | [0.9565, 0.9587] | 0.9744 | [0.9735, 0.9753] | 56679/60000 |
| counter_vs_cabinet | linear_logits | unweighted | 0.9596 | [0.9583, 0.9605] | 0.9679 | [0.9665, 0.9690] | 56679/60000 |
| counter_vs_cabinet | linear_logits | balanced | 0.9596 | [0.9586, 0.9605] | 0.9679 | [0.9665, 0.9689] | 56679/60000 |
| counter_vs_cabinet | linear_logits | weighted | 0.9596 | [0.9585, 0.9607] | 0.9679 | [0.9669, 0.9692] | 56679/60000 |
| counter_vs_cabinet | linear_logits | direct_pair_margin | 0.9469 | [0.9453, 0.9479] | 0.9846 | [0.9837, 0.9854] | 56679/60000 |

## Interpretation Guide
- `point_feature` is the Concerto backbone/decoder point feature returned by the ScanNet semseg model before the segmentation head.
- `linear_logits` with `unweighted`/`balanced`/`weighted` fits a binary probe on the 20-way linear logits.
- `linear_logits` with `direct_pair_margin` uses the fixed logit margin `logit(positive_class) - logit(negative_class)` without refitting a pair probe.
- The companion confusion CSV reports full 20-way predictions for the validation rows belonging to each class pair.
- Train and validation use the deterministic ScanNet validation-style transform so the stage trace is not affected by train-time random augmentations.
