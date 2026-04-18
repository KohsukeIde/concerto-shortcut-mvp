# ScanNet Point-Level Stage-Wise Trace

## Setup
- linear config: `exp/concerto/scannet-proxy-large-video-official-lin/config.py`
- linear weight: `exp/concerto/scannet-proxy-large-video-official-lin/model/model_last.pth`
- data root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/scannet`
- train batches seen: 256
- val batches seen: 128
- train class counts: {'wall': 60000, 'cabinet': 60000, 'table': 60000, 'door': 60000, 'picture': 60000, 'counter': 60000, 'desk': 60000, 'shower curtain': 48986, 'sink': 46302}
- val class counts: {'wall': 60000, 'cabinet': 60000, 'table': 60000, 'door': 60000, 'picture': 60000, 'counter': 56741, 'desk': 60000, 'shower curtain': 55794, 'sink': 33454}
- bootstrap iters: 100

## Results

| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| picture_vs_wall | point_feature | unweighted | 0.7040 | [0.7012, 0.7067] | 0.7753 | [0.7725, 0.7786] | 60000/60000 |
| picture_vs_wall | point_feature | balanced | 0.7041 | [0.7015, 0.7070] | 0.7753 | [0.7727, 0.7784] | 60000/60000 |
| picture_vs_wall | point_feature | weighted | 0.7040 | [0.7018, 0.7066] | 0.7753 | [0.7731, 0.7779] | 60000/60000 |
| picture_vs_wall | linear_logits | unweighted | 0.7602 | [0.7578, 0.7625] | 0.8831 | [0.8812, 0.8849] | 60000/60000 |
| picture_vs_wall | linear_logits | balanced | 0.7602 | [0.7587, 0.7620] | 0.8830 | [0.8815, 0.8846] | 60000/60000 |
| picture_vs_wall | linear_logits | weighted | 0.7602 | [0.7581, 0.7621] | 0.8831 | [0.8818, 0.8848] | 60000/60000 |
| picture_vs_wall | linear_logits | direct_pair_margin | 0.6987 | [0.6970, 0.7006] | 0.9245 | [0.9232, 0.9257] | 60000/60000 |
| counter_vs_cabinet | point_feature | unweighted | 0.9277 | [0.9262, 0.9288] | 0.9757 | [0.9750, 0.9764] | 56741/60000 |
| counter_vs_cabinet | point_feature | balanced | 0.9276 | [0.9261, 0.9291] | 0.9757 | [0.9749, 0.9765] | 56741/60000 |
| counter_vs_cabinet | point_feature | weighted | 0.9277 | [0.9263, 0.9291] | 0.9757 | [0.9750, 0.9767] | 56741/60000 |
| counter_vs_cabinet | linear_logits | unweighted | 0.9366 | [0.9353, 0.9380] | 0.9898 | [0.9894, 0.9903] | 56741/60000 |
| counter_vs_cabinet | linear_logits | balanced | 0.9366 | [0.9351, 0.9377] | 0.9898 | [0.9893, 0.9903] | 56741/60000 |
| counter_vs_cabinet | linear_logits | weighted | 0.9366 | [0.9350, 0.9382] | 0.9898 | [0.9894, 0.9903] | 56741/60000 |
| counter_vs_cabinet | linear_logits | direct_pair_margin | 0.9305 | [0.9289, 0.9318] | 0.9898 | [0.9894, 0.9902] | 56741/60000 |
| desk_vs_wall | point_feature | unweighted | 0.9264 | [0.9248, 0.9278] | 0.9870 | [0.9866, 0.9875] | 60000/60000 |
| desk_vs_wall | point_feature | balanced | 0.9264 | [0.9246, 0.9275] | 0.9870 | [0.9865, 0.9875] | 60000/60000 |
| desk_vs_wall | point_feature | weighted | 0.9264 | [0.9251, 0.9275] | 0.9870 | [0.9865, 0.9874] | 60000/60000 |
| desk_vs_wall | linear_logits | unweighted | 0.9475 | [0.9461, 0.9486] | 0.9947 | [0.9945, 0.9949] | 60000/60000 |
| desk_vs_wall | linear_logits | balanced | 0.9476 | [0.9463, 0.9490] | 0.9947 | [0.9945, 0.9950] | 60000/60000 |
| desk_vs_wall | linear_logits | weighted | 0.9475 | [0.9464, 0.9490] | 0.9947 | [0.9945, 0.9949] | 60000/60000 |
| desk_vs_wall | linear_logits | direct_pair_margin | 0.9739 | [0.9731, 0.9749] | 0.9988 | [0.9987, 0.9989] | 60000/60000 |
| desk_vs_table | point_feature | unweighted | 0.8711 | [0.8695, 0.8733] | 0.9127 | [0.9110, 0.9144] | 60000/60000 |
| desk_vs_table | point_feature | balanced | 0.8711 | [0.8695, 0.8729] | 0.9127 | [0.9110, 0.9141] | 60000/60000 |
| desk_vs_table | point_feature | weighted | 0.8711 | [0.8693, 0.8727] | 0.9127 | [0.9108, 0.9143] | 60000/60000 |
| desk_vs_table | linear_logits | unweighted | 0.9101 | [0.9089, 0.9117] | 0.9636 | [0.9626, 0.9646] | 60000/60000 |
| desk_vs_table | linear_logits | balanced | 0.9101 | [0.9087, 0.9119] | 0.9636 | [0.9627, 0.9646] | 60000/60000 |
| desk_vs_table | linear_logits | weighted | 0.9101 | [0.9086, 0.9115] | 0.9636 | [0.9624, 0.9643] | 60000/60000 |
| desk_vs_table | linear_logits | direct_pair_margin | 0.9195 | [0.9182, 0.9208] | 0.9630 | [0.9622, 0.9640] | 60000/60000 |
| sink_vs_cabinet | point_feature | unweighted | 0.9714 | [0.9705, 0.9723] | 0.9969 | [0.9967, 0.9971] | 33454/60000 |
| sink_vs_cabinet | point_feature | balanced | 0.9707 | [0.9695, 0.9715] | 0.9968 | [0.9965, 0.9969] | 33454/60000 |
| sink_vs_cabinet | point_feature | weighted | 0.9710 | [0.9701, 0.9719] | 0.9967 | [0.9966, 0.9969] | 33454/60000 |
| sink_vs_cabinet | linear_logits | unweighted | 0.9620 | [0.9608, 0.9634] | 0.9959 | [0.9957, 0.9961] | 33454/60000 |
| sink_vs_cabinet | linear_logits | balanced | 0.9625 | [0.9612, 0.9639] | 0.9958 | [0.9956, 0.9961] | 33454/60000 |
| sink_vs_cabinet | linear_logits | weighted | 0.9623 | [0.9608, 0.9641] | 0.9958 | [0.9956, 0.9961] | 33454/60000 |
| sink_vs_cabinet | linear_logits | direct_pair_margin | 0.9572 | [0.9558, 0.9589] | 0.9981 | [0.9979, 0.9982] | 33454/60000 |
| door_vs_wall | point_feature | unweighted | 0.8662 | [0.8646, 0.8681] | 0.9595 | [0.9589, 0.9606] | 60000/60000 |
| door_vs_wall | point_feature | balanced | 0.8662 | [0.8642, 0.8677] | 0.9595 | [0.9587, 0.9605] | 60000/60000 |
| door_vs_wall | point_feature | weighted | 0.8662 | [0.8645, 0.8681] | 0.9595 | [0.9585, 0.9604] | 60000/60000 |
| door_vs_wall | linear_logits | unweighted | 0.9517 | [0.9506, 0.9527] | 0.9897 | [0.9893, 0.9903] | 60000/60000 |
| door_vs_wall | linear_logits | balanced | 0.9517 | [0.9506, 0.9529] | 0.9897 | [0.9893, 0.9901] | 60000/60000 |
| door_vs_wall | linear_logits | weighted | 0.9517 | [0.9505, 0.9528] | 0.9897 | [0.9893, 0.9902] | 60000/60000 |
| door_vs_wall | linear_logits | direct_pair_margin | 0.9518 | [0.9504, 0.9529] | 0.9970 | [0.9968, 0.9972] | 60000/60000 |
| shower_curtain_vs_wall | point_feature | unweighted | 0.9784 | [0.9774, 0.9791] | 0.9997 | [0.9996, 0.9997] | 55794/60000 |
| shower_curtain_vs_wall | point_feature | balanced | 0.9801 | [0.9793, 0.9810] | 0.9997 | [0.9997, 0.9998] | 55794/60000 |
| shower_curtain_vs_wall | point_feature | weighted | 0.9791 | [0.9784, 0.9798] | 0.9997 | [0.9996, 0.9997] | 55794/60000 |
| shower_curtain_vs_wall | linear_logits | unweighted | 0.9785 | [0.9777, 0.9792] | 0.9994 | [0.9993, 0.9995] | 55794/60000 |
| shower_curtain_vs_wall | linear_logits | balanced | 0.9802 | [0.9795, 0.9810] | 0.9994 | [0.9993, 0.9995] | 55794/60000 |
| shower_curtain_vs_wall | linear_logits | weighted | 0.9806 | [0.9800, 0.9815] | 0.9994 | [0.9993, 0.9995] | 55794/60000 |
| shower_curtain_vs_wall | linear_logits | direct_pair_margin | 0.8770 | [0.8751, 0.8783] | 0.9985 | [0.9984, 0.9986] | 55794/60000 |

## Interpretation Guide
- `point_feature` is the frozen Concerto backbone feature returned by the official ScanNet linear-probe model before the segmentation head.
- `linear_logits` with `unweighted`/`balanced`/`weighted` fits a binary probe on the 20-way linear logits.
- `linear_logits` with `direct_pair_margin` uses the fixed logit margin `logit(positive_class) - logit(negative_class)` without refitting a pair probe.
- The companion confusion CSV reports full 20-way predictions for the validation rows belonging to each class pair.
- Train and validation use the deterministic ScanNet validation-style transform so the stage trace is not affected by train-time random augmentations.
