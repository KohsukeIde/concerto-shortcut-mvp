# Proposal Recall Analysis

Gate for proposal-first / mask-lite readout: can simple local proposals cover weak-class points as high-purity candidates?

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Proposal sources: `voxel,pred_cc`
- Region voxel sizes: `4,8,16`
- Purity thresholds: `0.5,0.7,0.8,0.9`
- Class pairs: `picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall`
- Seen val batches: `312`

## Base Decoder Reference

- mIoU=0.7788, picture=0.4076, picture->wall=0.4378

## Picture Proposal Recall

| source | s | thr | recall | candidate point frac | candidate regions | mean purity | wall contam | wall-majority frac | pred-picture maj | pred-wall maj | best cover/scene |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pred_cc` | 4 | 0.5 | 0.5603 | 0.0111 | 247 | 0.5726 | 0.3965 | 0.4316 | 0.5266 | 0.4502 | 0.7095 |
| `pred_cc` | 4 | 0.7 | 0.5405 | 0.0105 | 230 | 0.5726 | 0.3965 | 0.4316 | 0.5266 | 0.4502 | 0.7095 |
| `pred_cc` | 4 | 0.8 | 0.5063 | 0.0097 | 215 | 0.5726 | 0.3965 | 0.4316 | 0.5266 | 0.4502 | 0.7095 |
| `pred_cc` | 4 | 0.9 | 0.3188 | 0.0059 | 148 | 0.5726 | 0.3965 | 0.4316 | 0.5266 | 0.4502 | 0.7095 |
| `pred_cc` | 8 | 0.5 | 0.5271 | 0.0108 | 181 | 0.5325 | 0.4265 | 0.4482 | 0.5011 | 0.4737 | 0.6858 |
| `pred_cc` | 8 | 0.7 | 0.5086 | 0.0103 | 165 | 0.5325 | 0.4265 | 0.4482 | 0.5011 | 0.4737 | 0.6858 |
| `pred_cc` | 8 | 0.8 | 0.3831 | 0.0074 | 123 | 0.5325 | 0.4265 | 0.4482 | 0.5011 | 0.4737 | 0.6858 |
| `pred_cc` | 8 | 0.9 | 0.2077 | 0.0038 | 74 | 0.5325 | 0.4265 | 0.4482 | 0.5011 | 0.4737 | 0.6858 |
| `pred_cc` | 16 | 0.5 | 0.4484 | 0.0103 | 139 | 0.4272 | 0.4989 | 0.5190 | 0.4229 | 0.5384 | 0.6427 |
| `pred_cc` | 16 | 0.7 | 0.3332 | 0.0069 | 102 | 0.4272 | 0.4989 | 0.5190 | 0.4229 | 0.5384 | 0.6427 |
| `pred_cc` | 16 | 0.8 | 0.2190 | 0.0042 | 63 | 0.4272 | 0.4989 | 0.5190 | 0.4229 | 0.5384 | 0.6427 |
| `pred_cc` | 16 | 0.9 | 0.1255 | 0.0023 | 34 | 0.4272 | 0.4989 | 0.5190 | 0.4229 | 0.5384 | 0.6427 |
| `voxel` | 4 | 0.5 | 0.9412 | 0.0176 | 15158 | 0.9182 | 0.0771 | 0.0613 | 0.5266 | 0.4502 | 0.0324 |
| `voxel` | 4 | 0.7 | 0.8797 | 0.0158 | 13927 | 0.9182 | 0.0771 | 0.0613 | 0.5266 | 0.4502 | 0.0324 |
| `voxel` | 4 | 0.8 | 0.8462 | 0.0150 | 13438 | 0.9182 | 0.0771 | 0.0613 | 0.5266 | 0.4502 | 0.0324 |
| `voxel` | 4 | 0.9 | 0.8055 | 0.0142 | 12914 | 0.9182 | 0.0771 | 0.0613 | 0.5266 | 0.4502 | 0.0324 |
| `voxel` | 8 | 0.5 | 0.8839 | 0.0173 | 3948 | 0.8533 | 0.1369 | 0.1098 | 0.5011 | 0.4737 | 0.0918 |
| `voxel` | 8 | 0.7 | 0.7845 | 0.0144 | 3398 | 0.8533 | 0.1369 | 0.1098 | 0.5011 | 0.4737 | 0.0918 |
| `voxel` | 8 | 0.8 | 0.7249 | 0.0130 | 3153 | 0.8533 | 0.1369 | 0.1098 | 0.5011 | 0.4737 | 0.0918 |
| `voxel` | 8 | 0.9 | 0.6677 | 0.0118 | 2935 | 0.8533 | 0.1369 | 0.1098 | 0.5011 | 0.4737 | 0.0918 |
| `voxel` | 16 | 0.5 | 0.7685 | 0.0162 | 989 | 0.7418 | 0.2310 | 0.1953 | 0.4229 | 0.5384 | 0.2338 |
| `voxel` | 16 | 0.7 | 0.6042 | 0.0114 | 758 | 0.7418 | 0.2310 | 0.1953 | 0.4229 | 0.5384 | 0.2338 |
| `voxel` | 16 | 0.8 | 0.5211 | 0.0094 | 661 | 0.7418 | 0.2310 | 0.1953 | 0.4229 | 0.5384 | 0.2338 |
| `voxel` | 16 | 0.9 | 0.4580 | 0.0081 | 594 | 0.7418 | 0.2310 | 0.1953 | 0.4229 | 0.5384 | 0.2338 |

## High-Purity Recall At Threshold 0.9

| class | counterpart | source | s | recall | candidate point frac | candidate regions | mean purity | counterpart contam | counterpart-majority frac |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `counter` | `cabinet` | `pred_cc` | 4 | 0.6671 | 0.0309 | 219 | 0.8213 | 0.0612 | 0.0388 |
| `counter` | `cabinet` | `pred_cc` | 8 | 0.3536 | 0.0165 | 90 | 0.7581 | 0.0988 | 0.0505 |
| `counter` | `cabinet` | `pred_cc` | 16 | 0.1256 | 0.0058 | 48 | 0.6281 | 0.1463 | 0.1145 |
| `counter` | `cabinet` | `voxel` | 4 | 0.8330 | 0.0368 | 15770 | 0.9317 | 0.0276 | 0.0202 |
| `counter` | `cabinet` | `voxel` | 8 | 0.6565 | 0.0291 | 3829 | 0.8631 | 0.0611 | 0.0349 |
| `counter` | `cabinet` | `voxel` | 16 | 0.4261 | 0.0189 | 859 | 0.7345 | 0.1112 | 0.0933 |
| `desk` | `table` | `pred_cc` | 4 | 0.8241 | 0.0492 | 502 | 0.8886 | 0.0047 | 0.0058 |
| `desk` | `table` | `pred_cc` | 8 | 0.6787 | 0.0410 | 272 | 0.8387 | 0.0047 | 0.0058 |
| `desk` | `table` | `pred_cc` | 16 | 0.3413 | 0.0209 | 119 | 0.7226 | 0.0069 | 0.0052 |
| `desk` | `table` | `voxel` | 4 | 0.9166 | 0.0530 | 49348 | 0.9635 | 0.0002 | 0.0001 |
| `desk` | `table` | `voxel` | 8 | 0.8199 | 0.0475 | 12713 | 0.9190 | 0.0003 | 0.0001 |
| `desk` | `table` | `voxel` | 16 | 0.6138 | 0.0357 | 2509 | 0.8185 | 0.0005 | 0.0000 |
| `door` | `wall` | `pred_cc` | 4 | 0.6909 | 0.0443 | 669 | 0.8651 | 0.1027 | 0.0550 |
| `door` | `wall` | `pred_cc` | 8 | 0.6121 | 0.0396 | 418 | 0.8367 | 0.1167 | 0.0632 |
| `door` | `wall` | `pred_cc` | 16 | 0.3321 | 0.0216 | 204 | 0.7606 | 0.1551 | 0.0864 |
| `door` | `wall` | `voxel` | 4 | 0.9103 | 0.0561 | 100030 | 0.9628 | 0.0260 | 0.0196 |
| `door` | `wall` | `voxel` | 8 | 0.8313 | 0.0513 | 25137 | 0.9291 | 0.0487 | 0.0343 |
| `door` | `wall` | `voxel` | 16 | 0.6641 | 0.0411 | 5837 | 0.8570 | 0.0908 | 0.0617 |
| `picture` | `wall` | `pred_cc` | 4 | 0.3188 | 0.0059 | 148 | 0.5726 | 0.3965 | 0.4316 |
| `picture` | `wall` | `pred_cc` | 8 | 0.2077 | 0.0038 | 74 | 0.5325 | 0.4265 | 0.4482 |
| `picture` | `wall` | `pred_cc` | 16 | 0.1255 | 0.0023 | 34 | 0.4272 | 0.4989 | 0.5190 |
| `picture` | `wall` | `voxel` | 4 | 0.8055 | 0.0142 | 12914 | 0.9182 | 0.0771 | 0.0613 |
| `picture` | `wall` | `voxel` | 8 | 0.6677 | 0.0118 | 2935 | 0.8533 | 0.1369 | 0.1098 |
| `picture` | `wall` | `voxel` | 16 | 0.4580 | 0.0081 | 594 | 0.7418 | 0.2310 | 0.1953 |
| `shower curtain` | `wall` | `pred_cc` | 4 | 0.9032 | 0.0481 | 100 | 0.9368 | 0.0354 | 0.0242 |
| `shower curtain` | `wall` | `pred_cc` | 8 | 0.8608 | 0.0463 | 57 | 0.9152 | 0.0402 | 0.0266 |
| `shower curtain` | `wall` | `pred_cc` | 16 | 0.5651 | 0.0311 | 30 | 0.8286 | 0.0860 | 0.0462 |
| `shower curtain` | `wall` | `voxel` | 4 | 0.9559 | 0.0500 | 8780 | 0.9817 | 0.0095 | 0.0058 |
| `shower curtain` | `wall` | `voxel` | 8 | 0.9029 | 0.0472 | 2078 | 0.9588 | 0.0196 | 0.0165 |
| `shower curtain` | `wall` | `voxel` | 16 | 0.7180 | 0.0377 | 471 | 0.8828 | 0.0605 | 0.0309 |
| `sink` | `cabinet` | `pred_cc` | 4 | 0.5131 | 0.0064 | 84 | 0.8056 | 0.0664 | 0.0329 |
| `sink` | `cabinet` | `pred_cc` | 8 | 0.4235 | 0.0054 | 44 | 0.7636 | 0.0744 | 0.0369 |
| `sink` | `cabinet` | `pred_cc` | 16 | 0.2490 | 0.0032 | 27 | 0.6492 | 0.1069 | 0.1110 |
| `sink` | `cabinet` | `voxel` | 4 | 0.8244 | 0.0101 | 6053 | 0.9261 | 0.0270 | 0.0206 |
| `sink` | `cabinet` | `voxel` | 8 | 0.6920 | 0.0085 | 1495 | 0.8633 | 0.0471 | 0.0418 |
| `sink` | `cabinet` | `voxel` | 16 | 0.4459 | 0.0055 | 270 | 0.7401 | 0.0881 | 0.0856 |

## Interpretation Gate

- PVD go: a weak class has high-purity recall at `thr>=0.8` with a reasonably small candidate point fraction.
- Fine-only go: `voxel s4` has high recall but `pred_cc` / coarse sizes collapse; proposal-first needs learned object-quality masks, not region averaging.
- No-go: even `voxel s4` has low high-purity recall; proposal-first is unlikely to recover the oracle headroom under this feature family.

## Files

- Detail CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/proposal_recall_analysis/proposal_recall_detail.csv`
