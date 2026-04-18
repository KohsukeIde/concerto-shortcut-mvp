# Concerto 3D / DINO Exact-Patch Controls Step A

## Setup
- config: `pretrain-concerto-v1m1-2-large-video`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/concerto/pretrain-concerto-v1m1-2-large-video.pth`
- data root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/concerto_scannet_imagepoint_absmeta`
- job: `133155.qjcm`, `Exit_status=0`, walltime `00:05:48`
- train batches seen: 256
- val batches seen: 128
- train class counts: {'wall': 12000, 'cabinet': 3043, 'table': 1409, 'door': 984, 'picture': 64, 'counter': 47, 'desk': 774, 'shower curtain': 9, 'sink': 19}
- val class counts: {'wall': 12000, 'cabinet': 971, 'table': 1353, 'door': 922, 'picture': 118, 'counter': 20, 'desk': 232, 'shower curtain': 74, 'sink': 14}
- bootstrap iters: 100

## Key Takeaways

- The original DINO Step A' vs Concerto A'' comparison was not apples-to-apples.
  On the exact same A'' patch subset, `picture_vs_wall` DINO is only `0.5801`
  balanced accuracy, while Concerto `encoder_pooled` is `0.5772` and
  `patch_proj` is `0.5217` under the balanced probe.
- Therefore the earlier `raw_dino 0.7797` vs `Concerto 0.5381/0.5547` gap is
  largely explained by patch-subset mismatch and class rarity, not a confirmed
  generic semantic transfer loss.
- Across the checked pairs, Concerto 3D is often comparable to or better than
  `dino_exact` (`desk_vs_wall`, `desk_vs_table`, `shower_curtain_vs_wall`).
  `door_vs_wall` is near chance for all features. Rare-class pairs such as
  `counter_vs_cabinet` and `sink_vs_cabinet` have very small positive counts,
  so their confidence intervals are wide.
- Current conclusion: the bottleneck is pair/subset-specific. The data do not
  support a broad "DINO semantics are lost in 3D alignment" claim.

## Results

| pair | feature | probe | bal acc | 95% CI | AUC | 95% CI | val pos/neg |
|---|---|---|---:|---:|---:|---:|---:|
| picture_vs_wall | dino_exact | unweighted | 0.4954 | [0.4946, 0.4962] | 0.4128 | [0.3566, 0.4547] | 118/12000 |
| picture_vs_wall | dino_exact | balanced | 0.5801 | [0.5324, 0.6148] | 0.5997 | [0.5427, 0.6514] | 118/12000 |
| picture_vs_wall | dino_exact | weighted | 0.4889 | [0.4836, 0.4977] | 0.4227 | [0.3840, 0.4726] | 118/12000 |
| picture_vs_wall | encoder_pooled | unweighted | 0.4847 | [0.4788, 0.4928] | 0.4806 | [0.4409, 0.5160] | 118/12000 |
| picture_vs_wall | encoder_pooled | balanced | 0.5772 | [0.5459, 0.6156] | 0.5649 | [0.5169, 0.6118] | 118/12000 |
| picture_vs_wall | encoder_pooled | weighted | 0.4676 | [0.4511, 0.4939] | 0.5197 | [0.4722, 0.5614] | 118/12000 |
| picture_vs_wall | patch_proj | unweighted | 0.4903 | [0.4844, 0.4999] | 0.5700 | [0.5349, 0.6031] | 118/12000 |
| picture_vs_wall | patch_proj | balanced | 0.5217 | [0.4830, 0.5686] | 0.4999 | [0.4446, 0.5603] | 118/12000 |
| picture_vs_wall | patch_proj | weighted | 0.4727 | [0.4637, 0.4830] | 0.5078 | [0.4615, 0.5545] | 118/12000 |
| counter_vs_cabinet | dino_exact | unweighted | 0.4912 | [0.4876, 0.4946] | 0.5161 | [0.3789, 0.6239] | 20/971 |
| counter_vs_cabinet | dino_exact | balanced | 0.3870 | [0.2618, 0.4932] | 0.4266 | [0.2950, 0.5653] | 20/971 |
| counter_vs_cabinet | dino_exact | weighted | 0.4967 | [0.4683, 0.5712] | 0.5139 | [0.3903, 0.6436] | 20/971 |
| counter_vs_cabinet | encoder_pooled | unweighted | 0.5140 | [0.4280, 0.5901] | 0.5077 | [0.3580, 0.6250] | 20/971 |
| counter_vs_cabinet | encoder_pooled | balanced | 0.5204 | [0.4150, 0.6078] | 0.5118 | [0.3881, 0.6355] | 20/971 |
| counter_vs_cabinet | encoder_pooled | weighted | 0.5205 | [0.4167, 0.6154] | 0.4913 | [0.3466, 0.6093] | 20/971 |
| counter_vs_cabinet | patch_proj | unweighted | 0.4710 | [0.4044, 0.5458] | 0.4596 | [0.3461, 0.5770] | 20/971 |
| counter_vs_cabinet | patch_proj | balanced | 0.5710 | [0.4939, 0.6217] | 0.7116 | [0.5735, 0.8226] | 20/971 |
| counter_vs_cabinet | patch_proj | weighted | 0.4839 | [0.4007, 0.5683] | 0.4743 | [0.3403, 0.5843] | 20/971 |
| desk_vs_wall | dino_exact | unweighted | 0.6165 | [0.5857, 0.6418] | 0.8093 | [0.7842, 0.8323] | 232/12000 |
| desk_vs_wall | dino_exact | balanced | 0.6731 | [0.6477, 0.7003] | 0.7180 | [0.6916, 0.7460] | 232/12000 |
| desk_vs_wall | dino_exact | weighted | 0.6594 | [0.6260, 0.6883] | 0.8064 | [0.7747, 0.8315] | 232/12000 |
| desk_vs_wall | encoder_pooled | unweighted | 0.7205 | [0.6887, 0.7505] | 0.8123 | [0.7829, 0.8418] | 232/12000 |
| desk_vs_wall | encoder_pooled | balanced | 0.6973 | [0.6804, 0.7207] | 0.7444 | [0.7189, 0.7720] | 232/12000 |
| desk_vs_wall | encoder_pooled | weighted | 0.7011 | [0.6677, 0.7357] | 0.7871 | [0.7541, 0.8216] | 232/12000 |
| desk_vs_wall | patch_proj | unweighted | 0.6726 | [0.6408, 0.7062] | 0.8273 | [0.8007, 0.8504] | 232/12000 |
| desk_vs_wall | patch_proj | balanced | 0.7496 | [0.7260, 0.7743] | 0.7883 | [0.7667, 0.8062] | 232/12000 |
| desk_vs_wall | patch_proj | weighted | 0.7294 | [0.6964, 0.7566] | 0.8126 | [0.7854, 0.8358] | 232/12000 |
| desk_vs_table | dino_exact | unweighted | 0.8496 | [0.8335, 0.8701] | 0.9207 | [0.9005, 0.9399] | 232/1353 |
| desk_vs_table | dino_exact | balanced | 0.8375 | [0.8115, 0.8562] | 0.9212 | [0.8968, 0.9395] | 232/1353 |
| desk_vs_table | dino_exact | weighted | 0.8511 | [0.8319, 0.8693] | 0.9222 | [0.8971, 0.9395] | 232/1353 |
| desk_vs_table | encoder_pooled | unweighted | 0.8959 | [0.8716, 0.9129] | 0.9389 | [0.9144, 0.9616] | 232/1353 |
| desk_vs_table | encoder_pooled | balanced | 0.8618 | [0.8378, 0.8846] | 0.9311 | [0.9091, 0.9506] | 232/1353 |
| desk_vs_table | encoder_pooled | weighted | 0.8861 | [0.8653, 0.9056] | 0.9382 | [0.9175, 0.9585] | 232/1353 |
| desk_vs_table | patch_proj | unweighted | 0.8869 | [0.8627, 0.9090] | 0.9402 | [0.9178, 0.9557] | 232/1353 |
| desk_vs_table | patch_proj | balanced | 0.8919 | [0.8765, 0.9093] | 0.9462 | [0.9296, 0.9621] | 232/1353 |
| desk_vs_table | patch_proj | weighted | 0.8836 | [0.8621, 0.9050] | 0.9375 | [0.9190, 0.9532] | 232/1353 |
| sink_vs_cabinet | dino_exact | unweighted | 0.5000 | [0.5000, 0.5000] | 0.5746 | [0.5012, 0.6705] | 14/971 |
| sink_vs_cabinet | dino_exact | balanced | 0.3869 | [0.2840, 0.4795] | 0.2973 | [0.2054, 0.3948] | 14/971 |
| sink_vs_cabinet | dino_exact | weighted | 0.4954 | [0.4923, 0.4979] | 0.5718 | [0.5040, 0.6634] | 14/971 |
| sink_vs_cabinet | encoder_pooled | unweighted | 0.4974 | [0.4954, 0.4990] | 0.5544 | [0.4684, 0.6239] | 14/971 |
| sink_vs_cabinet | encoder_pooled | balanced | 0.5310 | [0.4392, 0.6082] | 0.6283 | [0.5180, 0.7464] | 14/971 |
| sink_vs_cabinet | encoder_pooled | weighted | 0.4428 | [0.4299, 0.4516] | 0.5385 | [0.4316, 0.6062] | 14/971 |
| sink_vs_cabinet | patch_proj | unweighted | 0.4995 | [0.4985, 0.5000] | 0.8262 | [0.6689, 0.9117] | 14/971 |
| sink_vs_cabinet | patch_proj | balanced | 0.5094 | [0.3940, 0.6140] | 0.4618 | [0.3886, 0.5627] | 14/971 |
| sink_vs_cabinet | patch_proj | weighted | 0.5896 | [0.4791, 0.6995] | 0.8286 | [0.6883, 0.9321] | 14/971 |
| door_vs_wall | dino_exact | unweighted | 0.4849 | [0.4782, 0.4917] | 0.4583 | [0.4390, 0.4788] | 922/12000 |
| door_vs_wall | dino_exact | balanced | 0.4909 | [0.4774, 0.5090] | 0.4979 | [0.4802, 0.5177] | 922/12000 |
| door_vs_wall | dino_exact | weighted | 0.4653 | [0.4557, 0.4732] | 0.4596 | [0.4448, 0.4763] | 922/12000 |
| door_vs_wall | encoder_pooled | unweighted | 0.4993 | [0.4880, 0.5075] | 0.5633 | [0.5459, 0.5780] | 922/12000 |
| door_vs_wall | encoder_pooled | balanced | 0.5055 | [0.4890, 0.5216] | 0.5164 | [0.4986, 0.5330] | 922/12000 |
| door_vs_wall | encoder_pooled | weighted | 0.4949 | [0.4828, 0.5079] | 0.5575 | [0.5402, 0.5746] | 922/12000 |
| door_vs_wall | patch_proj | unweighted | 0.5038 | [0.4947, 0.5126] | 0.5635 | [0.5444, 0.5809] | 922/12000 |
| door_vs_wall | patch_proj | balanced | 0.4718 | [0.4563, 0.4885] | 0.4735 | [0.4556, 0.4928] | 922/12000 |
| door_vs_wall | patch_proj | weighted | 0.5344 | [0.5205, 0.5467] | 0.6568 | [0.6429, 0.6701] | 922/12000 |
| shower_curtain_vs_wall | dino_exact | unweighted | 0.5000 | [0.5000, 0.5000] | 0.6394 | [0.5539, 0.6923] | 74/12000 |
| shower_curtain_vs_wall | dino_exact | balanced | 0.6898 | [0.6451, 0.7316] | 0.7843 | [0.7238, 0.8360] | 74/12000 |
| shower_curtain_vs_wall | dino_exact | weighted | 0.5133 | [0.4997, 0.5304] | 0.5463 | [0.4820, 0.6086] | 74/12000 |
| shower_curtain_vs_wall | encoder_pooled | unweighted | 0.5000 | [0.5000, 0.5000] | 0.9075 | [0.8794, 0.9327] | 74/12000 |
| shower_curtain_vs_wall | encoder_pooled | balanced | 0.7574 | [0.7531, 0.7611] | 0.9645 | [0.9485, 0.9757] | 74/12000 |
| shower_curtain_vs_wall | encoder_pooled | weighted | 0.5268 | [0.5064, 0.5572] | 0.8606 | [0.8257, 0.8926] | 74/12000 |
| shower_curtain_vs_wall | patch_proj | unweighted | 0.5000 | [0.5000, 0.5000] | 0.9190 | [0.8616, 0.9515] | 74/12000 |
| shower_curtain_vs_wall | patch_proj | balanced | 0.6505 | [0.6006, 0.6936] | 0.6943 | [0.6094, 0.7503] | 74/12000 |
| shower_curtain_vs_wall | patch_proj | weighted | 0.5933 | [0.5527, 0.6473] | 0.8267 | [0.7797, 0.8701] | 74/12000 |

## Interpretation Guide
- `dino_exact` is the frozen DINO target feature from `model.ENC2D_forward` on the exact same augmented image patches used for the Concerto rows.
- `encoder_pooled` is the Concerto 3D encoder feature pooled to those same patch ids through point-pixel correspondence.
- `patch_proj` is the Concerto enc2d patch projection of `encoder_pooled`.
- `balanced` trains the binary probe on a class-balanced train subset; validation metrics are still reported on all validation rows through balanced accuracy and AUC.
- Confidence intervals bootstrap validation rows with class-stratified resampling.
