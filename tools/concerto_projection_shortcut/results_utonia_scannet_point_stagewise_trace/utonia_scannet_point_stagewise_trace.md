# Utonia ScanNet Point-Level Stage-Wise Trace

## Setup
- utonia weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia.pth`
- seg head weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia_linear_prob_head_sc.pth`
- data root: `data/scannet`
- train batches seen: 128
- val batches seen: 64
- train class counts: {'wall': 60000, 'cabinet': 60000, 'door': 60000, 'picture': 51612, 'counter': 21864}
- val class counts: {'wall': 60000, 'cabinet': 60000, 'door': 60000, 'picture': 4750, 'counter': 56243}
- bootstrap iters: 100

## Results

| pair | stage | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| picture_vs_wall | point_feature | unweighted | 0.8861 | [0.8820, 0.8899] | 0.9657 | [0.9637, 0.9678] | 4750/60000 |
| picture_vs_wall | point_feature | balanced | 0.8847 | [0.8812, 0.8890] | 0.9651 | [0.9632, 0.9672] | 4750/60000 |
| picture_vs_wall | point_feature | weighted | 0.8844 | [0.8806, 0.8885] | 0.9657 | [0.9637, 0.9681] | 4750/60000 |
| picture_vs_wall | linear_logits | unweighted | 0.9054 | [0.9018, 0.9082] | 0.9707 | [0.9690, 0.9722] | 4750/60000 |
| picture_vs_wall | linear_logits | balanced | 0.9039 | [0.9009, 0.9067] | 0.9706 | [0.9691, 0.9721] | 4750/60000 |
| picture_vs_wall | linear_logits | weighted | 0.9048 | [0.9013, 0.9080] | 0.9708 | [0.9692, 0.9723] | 4750/60000 |
| picture_vs_wall | linear_logits | direct_pair_margin | 0.9320 | [0.9272, 0.9371] | nan | [0.9996, 0.9997] | 4750/60000 |
| door_vs_wall | point_feature | unweighted | 0.7295 | [0.7271, 0.7322] | 0.8528 | [0.8509, 0.8550] | 60000/60000 |
| door_vs_wall | point_feature | balanced | 0.7294 | [0.7267, 0.7324] | 0.8528 | [0.8504, 0.8553] | 60000/60000 |
| door_vs_wall | point_feature | weighted | 0.7295 | [0.7268, 0.7317] | 0.8528 | [0.8506, 0.8551] | 60000/60000 |
| door_vs_wall | linear_logits | unweighted | 0.9122 | [0.9107, 0.9140] | 0.9815 | [0.9810, 0.9821] | 60000/60000 |
| door_vs_wall | linear_logits | balanced | 0.9122 | [0.9106, 0.9141] | 0.9815 | [0.9810, 0.9821] | 60000/60000 |
| door_vs_wall | linear_logits | weighted | 0.9122 | [0.9106, 0.9141] | 0.9815 | [0.9811, 0.9820] | 60000/60000 |
| door_vs_wall | linear_logits | direct_pair_margin | 0.9624 | [0.9612, 0.9634] | nan | [0.9938, 0.9944] | 60000/60000 |
| counter_vs_cabinet | point_feature | unweighted | 0.6932 | [0.6917, 0.6952] | 0.9572 | [0.9564, 0.9581] | 56243/60000 |
| counter_vs_cabinet | point_feature | balanced | 0.6740 | [0.6724, 0.6761] | 0.9567 | [0.9558, 0.9578] | 56243/60000 |
| counter_vs_cabinet | point_feature | weighted | 0.6823 | [0.6806, 0.6843] | 0.9592 | [0.9582, 0.9602] | 56243/60000 |
| counter_vs_cabinet | linear_logits | unweighted | 0.8361 | [0.8340, 0.8377] | 0.9204 | [0.9191, 0.9219] | 56243/60000 |
| counter_vs_cabinet | linear_logits | balanced | 0.8366 | [0.8344, 0.8387] | 0.9221 | [0.9207, 0.9234] | 56243/60000 |
| counter_vs_cabinet | linear_logits | weighted | 0.8411 | [0.8394, 0.8429] | 0.9255 | [0.9245, 0.9267] | 56243/60000 |
| counter_vs_cabinet | linear_logits | direct_pair_margin | 0.9499 | [0.9489, 0.9512] | nan | [0.9896, 0.9904] | 56243/60000 |

## Notes
- `point_feature` is the Utonia point feature after unpooling the released backbone to the segmentation-head resolution.
- `linear_logits` uses the released ScanNet linear probing head bundled with Utonia.
- Validation rows are expanded back to raw points through the transform `inverse` mapping so the trace matches the original scene labels.
