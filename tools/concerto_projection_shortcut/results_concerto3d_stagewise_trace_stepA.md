# Concerto 3D / DINO Exact-Patch Controls Step A

## Setup
- config: `pretrain-concerto-v1m1-2-large-video`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/concerto/pretrain-concerto-v1m1-2-large-video.pth`
- data root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/concerto_scannet_imagepoint_absmeta`
- train batches seen: 256
- val batches seen: 128
- train class counts: {'wall': 12000, 'cabinet': 4277, 'table': 1994, 'door': 956, 'picture': 113, 'counter': 189, 'desk': 836, 'shower curtain': 8, 'sink': 5}
- val class counts: {'wall': 9074, 'cabinet': 1076, 'table': 1439, 'door': 433, 'picture': 45, 'counter': 45, 'desk': 184, 'shower curtain': 0, 'sink': 10}
- bootstrap iters: 100

## Results

| pair | feature | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| picture_vs_wall | dino_exact | unweighted | 0.5086 | [0.4969, 0.5417] | 0.4089 | [0.3346, 0.4887] | 45/9074 |
| picture_vs_wall | dino_exact | balanced | 0.6146 | [0.5564, 0.6671] | 0.6333 | [0.5550, 0.6861] | 45/9074 |
| picture_vs_wall | dino_exact | weighted | 0.4969 | [0.4743, 0.5256] | 0.4163 | [0.3060, 0.4978] | 45/9074 |
| picture_vs_wall | encoder_pooled | unweighted | 0.5184 | [0.4857, 0.5509] | 0.5630 | [0.5035, 0.6268] | 45/9074 |
| picture_vs_wall | encoder_pooled | balanced | 0.5548 | [0.4884, 0.6127] | 0.5656 | [0.4801, 0.6312] | 45/9074 |
| picture_vs_wall | encoder_pooled | weighted | 0.5169 | [0.4616, 0.5785] | 0.5534 | [0.4853, 0.6366] | 45/9074 |
| picture_vs_wall | patch_proj | unweighted | 0.5104 | [0.4871, 0.5449] | 0.7830 | [0.7038, 0.8709] | 45/9074 |
| picture_vs_wall | patch_proj | balanced | 0.5359 | [0.4706, 0.6088] | 0.5816 | [0.5053, 0.6417] | 45/9074 |
| picture_vs_wall | patch_proj | weighted | 0.6227 | [0.5707, 0.6903] | 0.7598 | [0.6994, 0.8348] | 45/9074 |
| picture_vs_wall | linear_feat_pooled | unweighted | 0.5187 | [0.4852, 0.5635] | 0.4140 | [0.3254, 0.5065] | 45/9074 |
| picture_vs_wall | linear_feat_pooled | balanced | 0.5653 | [0.4947, 0.6290] | 0.5527 | [0.4717, 0.6152] | 45/9074 |
| picture_vs_wall | linear_feat_pooled | weighted | 0.5114 | [0.4772, 0.5560] | 0.4293 | [0.3432, 0.5221] | 45/9074 |
| picture_vs_wall | linear_logits_pooled | unweighted | 0.5000 | [0.5000, 0.5000] | 0.6854 | [0.6444, 0.7365] | 45/9074 |
| picture_vs_wall | linear_logits_pooled | balanced | 0.7128 | [0.6654, 0.7521] | 0.7432 | [0.6954, 0.7910] | 45/9074 |
| picture_vs_wall | linear_logits_pooled | weighted | 0.6171 | [0.5609, 0.6814] | 0.7233 | [0.6740, 0.7737] | 45/9074 |
| counter_vs_cabinet | dino_exact | unweighted | 0.5101 | [0.4732, 0.5549] | 0.5842 | [0.4665, 0.6711] | 45/1076 |
| counter_vs_cabinet | dino_exact | balanced | 0.5458 | [0.4707, 0.6138] | 0.5773 | [0.4982, 0.6593] | 45/1076 |
| counter_vs_cabinet | dino_exact | weighted | 0.5308 | [0.4716, 0.5983] | 0.5900 | [0.5187, 0.6686] | 45/1076 |
| counter_vs_cabinet | encoder_pooled | unweighted | 0.5156 | [0.4788, 0.5620] | 0.5776 | [0.5141, 0.6529] | 45/1076 |
| counter_vs_cabinet | encoder_pooled | balanced | 0.4848 | [0.4152, 0.5420] | 0.4872 | [0.4167, 0.5513] | 45/1076 |
| counter_vs_cabinet | encoder_pooled | weighted | 0.5123 | [0.4695, 0.5678] | 0.5698 | [0.4926, 0.6538] | 45/1076 |
| counter_vs_cabinet | patch_proj | unweighted | 0.4989 | [0.4581, 0.5480] | 0.5417 | [0.4781, 0.6129] | 45/1076 |
| counter_vs_cabinet | patch_proj | balanced | 0.5569 | [0.4829, 0.6302] | 0.6167 | [0.5205, 0.6965] | 45/1076 |
| counter_vs_cabinet | patch_proj | weighted | 0.4924 | [0.4558, 0.5369] | 0.5317 | [0.4819, 0.5992] | 45/1076 |
| counter_vs_cabinet | linear_feat_pooled | unweighted | 0.5055 | [0.4786, 0.5456] | 0.5593 | [0.4954, 0.6210] | 45/1076 |
| counter_vs_cabinet | linear_feat_pooled | balanced | 0.4454 | [0.3804, 0.5151] | 0.4625 | [0.3831, 0.5367] | 45/1076 |
| counter_vs_cabinet | linear_feat_pooled | weighted | 0.4929 | [0.4568, 0.5305] | 0.5719 | [0.5131, 0.6430] | 45/1076 |
| counter_vs_cabinet | linear_logits_pooled | unweighted | 0.4962 | [0.4730, 0.5234] | 0.5809 | [0.5150, 0.6438] | 45/1076 |
| counter_vs_cabinet | linear_logits_pooled | balanced | 0.5348 | [0.4779, 0.5995] | 0.5158 | [0.4307, 0.5875] | 45/1076 |
| counter_vs_cabinet | linear_logits_pooled | weighted | 0.5325 | [0.4774, 0.6151] | 0.5404 | [0.4718, 0.6300] | 45/1076 |
| desk_vs_wall | dino_exact | unweighted | 0.5351 | [0.5157, 0.5574] | 0.6696 | [0.6295, 0.6993] | 184/9074 |
| desk_vs_wall | dino_exact | balanced | 0.6389 | [0.6063, 0.6696] | 0.6957 | [0.6545, 0.7316] | 184/9074 |
| desk_vs_wall | dino_exact | weighted | 0.5601 | [0.5373, 0.5947] | 0.6641 | [0.6325, 0.7056] | 184/9074 |
| desk_vs_wall | encoder_pooled | unweighted | 0.6348 | [0.6007, 0.6789] | 0.7700 | [0.7443, 0.7995] | 184/9074 |
| desk_vs_wall | encoder_pooled | balanced | 0.6399 | [0.6092, 0.6666] | 0.6494 | [0.6207, 0.6759] | 184/9074 |
| desk_vs_wall | encoder_pooled | weighted | 0.6473 | [0.6180, 0.6787] | 0.7561 | [0.7349, 0.7820] | 184/9074 |
| desk_vs_wall | patch_proj | unweighted | 0.5505 | [0.5267, 0.5820] | 0.7692 | [0.7481, 0.7908] | 184/9074 |
| desk_vs_wall | patch_proj | balanced | 0.5711 | [0.5281, 0.6132] | 0.6099 | [0.5782, 0.6466] | 184/9074 |
| desk_vs_wall | patch_proj | weighted | 0.5828 | [0.5459, 0.6202] | 0.7290 | [0.7080, 0.7516] | 184/9074 |
| desk_vs_wall | linear_feat_pooled | unweighted | 0.5540 | [0.5203, 0.5841] | 0.6724 | [0.6402, 0.7012] | 184/9074 |
| desk_vs_wall | linear_feat_pooled | balanced | 0.6103 | [0.5758, 0.6307] | 0.6254 | [0.5917, 0.6483] | 184/9074 |
| desk_vs_wall | linear_feat_pooled | weighted | 0.5710 | [0.5300, 0.5996] | 0.6467 | [0.6062, 0.6785] | 184/9074 |
| desk_vs_wall | linear_logits_pooled | unweighted | 0.5331 | [0.5198, 0.5524] | 0.8982 | [0.8832, 0.9178] | 184/9074 |
| desk_vs_wall | linear_logits_pooled | balanced | 0.8106 | [0.7815, 0.8445] | 0.8905 | [0.8735, 0.9118] | 184/9074 |
| desk_vs_wall | linear_logits_pooled | weighted | 0.8058 | [0.7702, 0.8363] | 0.8928 | [0.8731, 0.9144] | 184/9074 |
| desk_vs_table | dino_exact | unweighted | 0.8841 | [0.8652, 0.9029] | 0.9449 | [0.9310, 0.9572] | 184/1439 |
| desk_vs_table | dino_exact | balanced | 0.8495 | [0.8304, 0.8667] | 0.9338 | [0.9208, 0.9468] | 184/1439 |
| desk_vs_table | dino_exact | weighted | 0.8855 | [0.8662, 0.9100] | 0.9442 | [0.9278, 0.9601] | 184/1439 |
| desk_vs_table | encoder_pooled | unweighted | 0.8602 | [0.8371, 0.8820] | 0.9353 | [0.9144, 0.9571] | 184/1439 |
| desk_vs_table | encoder_pooled | balanced | 0.8692 | [0.8477, 0.8892] | 0.9446 | [0.9262, 0.9604] | 184/1439 |
| desk_vs_table | encoder_pooled | weighted | 0.8646 | [0.8398, 0.8862] | 0.9348 | [0.9124, 0.9529] | 184/1439 |
| desk_vs_table | patch_proj | unweighted | 0.9170 | [0.9003, 0.9310] | 0.9711 | [0.9624, 0.9782] | 184/1439 |
| desk_vs_table | patch_proj | balanced | 0.8828 | [0.8645, 0.9028] | 0.9562 | [0.9456, 0.9661] | 184/1439 |
| desk_vs_table | patch_proj | weighted | 0.9111 | [0.8881, 0.9308] | 0.9702 | [0.9612, 0.9789] | 184/1439 |
| desk_vs_table | linear_feat_pooled | unweighted | 0.8451 | [0.8207, 0.8752] | 0.9247 | [0.9037, 0.9473] | 184/1439 |
| desk_vs_table | linear_feat_pooled | balanced | 0.8609 | [0.8375, 0.8808] | 0.9288 | [0.9091, 0.9424] | 184/1439 |
| desk_vs_table | linear_feat_pooled | weighted | 0.8426 | [0.8182, 0.8683] | 0.9174 | [0.8956, 0.9382] | 184/1439 |
| desk_vs_table | linear_logits_pooled | unweighted | 0.8935 | [0.8630, 0.9181] | 0.9679 | [0.9555, 0.9769] | 184/1439 |
| desk_vs_table | linear_logits_pooled | balanced | 0.9060 | [0.8881, 0.9275] | 0.9664 | [0.9553, 0.9768] | 184/1439 |
| desk_vs_table | linear_logits_pooled | weighted | 0.9114 | [0.8909, 0.9303] | 0.9669 | [0.9551, 0.9766] | 184/1439 |
| sink_vs_cabinet | dino_exact | unweighted | 0.5000 | [0.5000, 0.5000] | 0.4114 | [0.2547, 0.5836] | 10/1076 |
| sink_vs_cabinet | dino_exact | balanced | 0.4812 | [0.3752, 0.6149] | 0.5047 | [0.3743, 0.6456] | 10/1076 |
| sink_vs_cabinet | dino_exact | weighted | 0.5000 | [0.5000, 0.5000] | 0.3796 | [0.2048, 0.5410] | 10/1076 |
| sink_vs_cabinet | encoder_pooled | unweighted | 0.4912 | [0.4867, 0.4951] | 0.6486 | [0.5527, 0.7383] | 10/1076 |
| sink_vs_cabinet | encoder_pooled | balanced | 0.4820 | [0.3396, 0.6274] | 0.4826 | [0.3061, 0.6780] | 10/1076 |
| sink_vs_cabinet | encoder_pooled | weighted | 0.4703 | [0.4635, 0.4777] | 0.7760 | [0.6908, 0.8688] | 10/1076 |
| sink_vs_cabinet | patch_proj | unweighted | 0.4972 | [0.4946, 0.4991] | 0.3065 | [0.1839, 0.4132] | 10/1076 |
| sink_vs_cabinet | patch_proj | balanced | 0.5550 | [0.4522, 0.6108] | 0.3669 | [0.2649, 0.4703] | 10/1076 |
| sink_vs_cabinet | patch_proj | weighted | 0.4828 | [0.4777, 0.4863] | 0.4189 | [0.2927, 0.5481] | 10/1076 |
| sink_vs_cabinet | linear_feat_pooled | unweighted | 0.4926 | [0.4895, 0.4961] | 0.6064 | [0.4232, 0.7396] | 10/1076 |
| sink_vs_cabinet | linear_feat_pooled | balanced | 0.4708 | [0.3252, 0.5827] | 0.4592 | [0.3328, 0.6191] | 10/1076 |
| sink_vs_cabinet | linear_feat_pooled | weighted | 0.6203 | [0.5111, 0.7486] | 0.8016 | [0.7084, 0.8960] | 10/1076 |
| sink_vs_cabinet | linear_logits_pooled | unweighted | 0.5000 | [0.5000, 0.5000] | 0.3690 | [0.2134, 0.5141] | 10/1076 |
| sink_vs_cabinet | linear_logits_pooled | balanced | 0.5223 | [0.4187, 0.6647] | 0.5070 | [0.3539, 0.7155] | 10/1076 |
| sink_vs_cabinet | linear_logits_pooled | weighted | 0.4814 | [0.4763, 0.4868] | 0.5217 | [0.3030, 0.6571] | 10/1076 |
| door_vs_wall | dino_exact | unweighted | 0.4994 | [0.4901, 0.5144] | 0.4841 | [0.4552, 0.5135] | 433/9074 |
| door_vs_wall | dino_exact | balanced | 0.4838 | [0.4603, 0.5103] | 0.4960 | [0.4712, 0.5209] | 433/9074 |
| door_vs_wall | dino_exact | weighted | 0.5105 | [0.4936, 0.5256] | 0.4828 | [0.4519, 0.5118] | 433/9074 |
| door_vs_wall | encoder_pooled | unweighted | 0.5587 | [0.5465, 0.5793] | 0.5804 | [0.5536, 0.6113] | 433/9074 |
| door_vs_wall | encoder_pooled | balanced | 0.5230 | [0.5013, 0.5494] | 0.5116 | [0.4835, 0.5455] | 433/9074 |
| door_vs_wall | encoder_pooled | weighted | 0.5617 | [0.5420, 0.5818] | 0.5856 | [0.5598, 0.6144] | 433/9074 |
| door_vs_wall | patch_proj | unweighted | 0.5091 | [0.4958, 0.5271] | 0.5561 | [0.5268, 0.5832] | 433/9074 |
| door_vs_wall | patch_proj | balanced | 0.5831 | [0.5563, 0.6048] | 0.6228 | [0.5923, 0.6500] | 433/9074 |
| door_vs_wall | patch_proj | weighted | 0.5187 | [0.4985, 0.5431] | 0.5654 | [0.5366, 0.6005] | 433/9074 |
| door_vs_wall | linear_feat_pooled | unweighted | 0.4825 | [0.4678, 0.4965] | 0.4816 | [0.4596, 0.5055] | 433/9074 |
| door_vs_wall | linear_feat_pooled | balanced | 0.5085 | [0.4882, 0.5282] | 0.5039 | [0.4724, 0.5293] | 433/9074 |
| door_vs_wall | linear_feat_pooled | weighted | 0.4817 | [0.4699, 0.4975] | 0.5087 | [0.4864, 0.5372] | 433/9074 |
| door_vs_wall | linear_logits_pooled | unweighted | 0.5088 | [0.5014, 0.5181] | 0.5838 | [0.5610, 0.6138] | 433/9074 |
| door_vs_wall | linear_logits_pooled | balanced | 0.5557 | [0.5307, 0.5772] | 0.5916 | [0.5692, 0.6226] | 433/9074 |
| door_vs_wall | linear_logits_pooled | weighted | 0.5684 | [0.5476, 0.5895] | 0.6053 | [0.5811, 0.6284] | 433/9074 |

## Interpretation Guide
- `dino_exact` is the frozen DINO target feature from `model.ENC2D_forward` on the exact same augmented image patches used for the Concerto rows.
- `encoder_pooled` is the Concerto 3D encoder feature pooled to those same patch ids through point-pixel correspondence.
- `patch_proj` is the Concerto enc2d patch projection of `encoder_pooled`.
- `linear_feat_pooled` is the official ScanNet linear-probe backbone feature pooled to the same patch ids.
- `linear_logits_pooled` is the official ScanNet linear-probe logits pooled to the same patch ids.
- `balanced` trains the binary probe on a class-balanced train subset; validation metrics are still reported on all validation rows through balanced accuracy and AUC.
- Confidence intervals bootstrap validation rows with class-stratified resampling.
