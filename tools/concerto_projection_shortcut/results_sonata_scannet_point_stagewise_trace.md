# ScanNet Point-Level Stage-Wise Trace

## Setup
- linear config: `configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
- linear weight: `data/weights/sonata/sonata_scannet_linear_merged.pth`
- data root: `data/scannet`
- train batches seen: 256
- val batches seen: 128
- train class counts: {'wall': 60000, 'cabinet': 60000, 'table': 60000, 'door': 60000, 'picture': 60000, 'counter': 60000, 'desk': 60000, 'shower curtain': 49013, 'sink': 46293}
- val class counts: {'wall': 60000, 'cabinet': 60000, 'table': 60000, 'door': 60000, 'picture': 60000, 'counter': 56677, 'desk': 60000, 'shower curtain': 55778, 'sink': 33413}
- bootstrap iters: 100

## Results

| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| picture_vs_wall | point_feature | unweighted | 0.7501 | [0.7475, 0.7520] | 0.8413 | [0.8395, 0.8433] | 60000/60000 |
| picture_vs_wall | point_feature | balanced | 0.7501 | [0.7481, 0.7522] | 0.8414 | [0.8397, 0.8436] | 60000/60000 |
| picture_vs_wall | point_feature | weighted | 0.7501 | [0.7477, 0.7527] | 0.8413 | [0.8390, 0.8438] | 60000/60000 |
| picture_vs_wall | linear_logits | unweighted | 0.8155 | [0.8131, 0.8175] | 0.9093 | [0.9075, 0.9110] | 60000/60000 |
| picture_vs_wall | linear_logits | balanced | 0.8155 | [0.8137, 0.8177] | 0.9093 | [0.9078, 0.9109] | 60000/60000 |
| picture_vs_wall | linear_logits | weighted | 0.8155 | [0.8136, 0.8176] | 0.9093 | [0.9081, 0.9108] | 60000/60000 |
| picture_vs_wall | linear_logits | direct_pair_margin | 0.6452 | [0.6437, 0.6467] | 0.8946 | [0.8930, 0.8961] | 60000/60000 |
| counter_vs_cabinet | point_feature | unweighted | 0.9487 | [0.9473, 0.9498] | 0.9863 | [0.9858, 0.9867] | 56677/60000 |
| counter_vs_cabinet | point_feature | balanced | 0.9487 | [0.9476, 0.9499] | 0.9863 | [0.9858, 0.9869] | 56677/60000 |
| counter_vs_cabinet | point_feature | weighted | 0.9487 | [0.9474, 0.9502] | 0.9863 | [0.9858, 0.9868] | 56677/60000 |
| counter_vs_cabinet | linear_logits | unweighted | 0.9483 | [0.9469, 0.9496] | 0.9872 | [0.9866, 0.9878] | 56677/60000 |
| counter_vs_cabinet | linear_logits | balanced | 0.9483 | [0.9471, 0.9494] | 0.9872 | [0.9867, 0.9877] | 56677/60000 |
| counter_vs_cabinet | linear_logits | weighted | 0.9483 | [0.9470, 0.9497] | 0.9872 | [0.9867, 0.9878] | 56677/60000 |
| counter_vs_cabinet | linear_logits | direct_pair_margin | 0.9453 | [0.9441, 0.9465] | 0.9898 | [0.9893, 0.9903] | 56677/60000 |
| desk_vs_wall | point_feature | unweighted | 0.9358 | [0.9345, 0.9371] | 0.9920 | [0.9917, 0.9922] | 60000/60000 |
| desk_vs_wall | point_feature | balanced | 0.9358 | [0.9342, 0.9368] | 0.9920 | [0.9916, 0.9923] | 60000/60000 |
| desk_vs_wall | point_feature | weighted | 0.9358 | [0.9345, 0.9370] | 0.9920 | [0.9917, 0.9922] | 60000/60000 |
| desk_vs_wall | linear_logits | unweighted | 0.9508 | [0.9498, 0.9522] | 0.9942 | [0.9939, 0.9944] | 60000/60000 |
| desk_vs_wall | linear_logits | balanced | 0.9508 | [0.9496, 0.9522] | 0.9942 | [0.9939, 0.9945] | 60000/60000 |
| desk_vs_wall | linear_logits | weighted | 0.9508 | [0.9495, 0.9519] | 0.9942 | [0.9939, 0.9944] | 60000/60000 |
| desk_vs_wall | linear_logits | direct_pair_margin | 0.9733 | [0.9724, 0.9741] | 0.9981 | [0.9980, 0.9982] | 60000/60000 |
| desk_vs_table | point_feature | unweighted | 0.8708 | [0.8685, 0.8726] | 0.9145 | [0.9131, 0.9165] | 60000/60000 |
| desk_vs_table | point_feature | balanced | 0.8708 | [0.8689, 0.8724] | 0.9145 | [0.9129, 0.9159] | 60000/60000 |
| desk_vs_table | point_feature | weighted | 0.8708 | [0.8691, 0.8727] | 0.9145 | [0.9128, 0.9161] | 60000/60000 |
| desk_vs_table | linear_logits | unweighted | 0.8622 | [0.8605, 0.8638] | 0.9396 | [0.9387, 0.9405] | 60000/60000 |
| desk_vs_table | linear_logits | balanced | 0.8622 | [0.8603, 0.8644] | 0.9396 | [0.9384, 0.9408] | 60000/60000 |
| desk_vs_table | linear_logits | weighted | 0.8622 | [0.8604, 0.8639] | 0.9396 | [0.9385, 0.9407] | 60000/60000 |
| desk_vs_table | linear_logits | direct_pair_margin | 0.9062 | [0.9046, 0.9081] | 0.9572 | [0.9564, 0.9583] | 60000/60000 |
| sink_vs_cabinet | point_feature | unweighted | 0.9418 | [0.9402, 0.9434] | 0.9881 | [0.9876, 0.9886] | 33413/60000 |
| sink_vs_cabinet | point_feature | balanced | 0.9411 | [0.9395, 0.9425] | 0.9882 | [0.9877, 0.9885] | 33413/60000 |
| sink_vs_cabinet | point_feature | weighted | 0.9418 | [0.9404, 0.9432] | 0.9883 | [0.9878, 0.9887] | 33413/60000 |
| sink_vs_cabinet | linear_logits | unweighted | 0.9433 | [0.9419, 0.9451] | 0.9914 | [0.9911, 0.9918] | 33413/60000 |
| sink_vs_cabinet | linear_logits | balanced | 0.9437 | [0.9423, 0.9451] | 0.9909 | [0.9905, 0.9913] | 33413/60000 |
| sink_vs_cabinet | linear_logits | weighted | 0.9440 | [0.9425, 0.9453] | 0.9910 | [0.9907, 0.9914] | 33413/60000 |
| sink_vs_cabinet | linear_logits | direct_pair_margin | 0.9338 | [0.9320, 0.9355] | 0.9971 | [0.9969, 0.9973] | 33413/60000 |
| door_vs_wall | point_feature | unweighted | 0.9207 | [0.9191, 0.9219] | 0.9759 | [0.9752, 0.9767] | 60000/60000 |
| door_vs_wall | point_feature | balanced | 0.9207 | [0.9193, 0.9220] | 0.9759 | [0.9751, 0.9767] | 60000/60000 |
| door_vs_wall | point_feature | weighted | 0.9207 | [0.9195, 0.9221] | 0.9759 | [0.9752, 0.9766] | 60000/60000 |
| door_vs_wall | linear_logits | unweighted | 0.9507 | [0.9496, 0.9519] | 0.9890 | [0.9885, 0.9894] | 60000/60000 |
| door_vs_wall | linear_logits | balanced | 0.9507 | [0.9496, 0.9518] | 0.9890 | [0.9886, 0.9895] | 60000/60000 |
| door_vs_wall | linear_logits | weighted | 0.9507 | [0.9497, 0.9523] | 0.9890 | [0.9885, 0.9895] | 60000/60000 |
| door_vs_wall | linear_logits | direct_pair_margin | 0.9298 | [0.9284, 0.9311] | 0.9930 | [0.9927, 0.9933] | 60000/60000 |
| shower_curtain_vs_wall | point_feature | unweighted | 0.9387 | [0.9373, 0.9400] | 0.9962 | [0.9959, 0.9964] | 55778/60000 |
| shower_curtain_vs_wall | point_feature | balanced | 0.9382 | [0.9368, 0.9396] | 0.9960 | [0.9958, 0.9962] | 55778/60000 |
| shower_curtain_vs_wall | point_feature | weighted | 0.9385 | [0.9371, 0.9397] | 0.9962 | [0.9960, 0.9964] | 55778/60000 |
| shower_curtain_vs_wall | linear_logits | unweighted | 0.9407 | [0.9395, 0.9418] | 0.9912 | [0.9910, 0.9915] | 55778/60000 |
| shower_curtain_vs_wall | linear_logits | balanced | 0.9415 | [0.9402, 0.9426] | 0.9914 | [0.9911, 0.9916] | 55778/60000 |
| shower_curtain_vs_wall | linear_logits | weighted | 0.9414 | [0.9403, 0.9426] | 0.9913 | [0.9910, 0.9916] | 55778/60000 |
| shower_curtain_vs_wall | linear_logits | direct_pair_margin | 0.8880 | [0.8863, 0.8893] | 0.9977 | [0.9976, 0.9978] | 55778/60000 |

## Interpretation Guide
- `point_feature` is the Concerto backbone/decoder point feature returned by the ScanNet semseg model before the segmentation head.
- `linear_logits` with `unweighted`/`balanced`/`weighted` fits a binary probe on the 20-way linear logits.
- `linear_logits` with `direct_pair_margin` uses the fixed logit margin `logit(positive_class) - logit(negative_class)` without refitting a pair probe.
- The companion confusion CSV reports full 20-way predictions for the validation rows belonging to each class pair.
- Train and validation use the deterministic ScanNet validation-style transform so the stage trace is not affected by train-time random augmentations.
