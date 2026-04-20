# Proposal Recall Analysis

Gate for proposal-first / mask-lite readout: can simple local proposals cover weak-class points as high-purity candidates?

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Proposal sources: `voxel,pred_cc`
- Region voxel sizes: `4,8,16`
- Purity thresholds: `0.5,0.7,0.8,0.9`
- Class pairs: `picture:wall,counter:cabinet,desk:table,sink:cabinet,door:wall,shower curtain:wall`
- Seen val batches: `4`

## Base Decoder Reference

- mIoU=0.4091, picture=0.6286, picture->wall=0.0260

## Picture Proposal Recall

| source | s | thr | recall | candidate point frac | candidate regions | mean purity | wall contam | wall-majority frac | pred-picture maj | pred-wall maj | best cover/scene |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pred_cc` | 4 | 0.5 | 0.8589 | 0.0181 | 9 | 0.8109 | 0.0728 | 0.0224 | 0.6889 | 0.0230 | 0.6818 |
| `pred_cc` | 4 | 0.7 | 0.8589 | 0.0181 | 9 | 0.8109 | 0.0728 | 0.0224 | 0.6889 | 0.0230 | 0.6818 |
| `pred_cc` | 4 | 0.8 | 0.8589 | 0.0181 | 9 | 0.8109 | 0.0728 | 0.0224 | 0.6889 | 0.0230 | 0.6818 |
| `pred_cc` | 4 | 0.9 | 0.8259 | 0.0173 | 8 | 0.8109 | 0.0728 | 0.0224 | 0.6889 | 0.0230 | 0.6818 |
| `pred_cc` | 8 | 0.5 | 0.8436 | 0.0175 | 4 | 0.8082 | 0.0771 | 0.0419 | 0.7013 | 0.0425 | 0.7013 |
| `pred_cc` | 8 | 0.7 | 0.8436 | 0.0175 | 4 | 0.8082 | 0.0771 | 0.0419 | 0.7013 | 0.0425 | 0.7013 |
| `pred_cc` | 8 | 0.8 | 0.8223 | 0.0169 | 3 | 0.8082 | 0.0771 | 0.0419 | 0.7013 | 0.0425 | 0.7013 |
| `pred_cc` | 8 | 0.9 | 0.8223 | 0.0169 | 3 | 0.8082 | 0.0771 | 0.0419 | 0.7013 | 0.0425 | 0.7013 |
| `pred_cc` | 16 | 0.5 | 0.6647 | 0.0147 | 3 | 0.5990 | 0.2344 | 0.1724 | 0.5449 | 0.1724 | 0.4870 |
| `pred_cc` | 16 | 0.7 | 0.6647 | 0.0147 | 3 | 0.5990 | 0.2344 | 0.1724 | 0.5449 | 0.1724 | 0.4870 |
| `pred_cc` | 16 | 0.8 | 0.6647 | 0.0147 | 3 | 0.5990 | 0.2344 | 0.1724 | 0.5449 | 0.1724 | 0.4870 |
| `pred_cc` | 16 | 0.9 | 0.1777 | 0.0035 | 2 | 0.5990 | 0.2344 | 0.1724 | 0.5449 | 0.1724 | 0.4870 |
| `voxel` | 4 | 0.5 | 0.9675 | 0.0199 | 169 | 0.9523 | 0.0373 | 0.0342 | 0.6889 | 0.0230 | 0.0118 |
| `voxel` | 4 | 0.7 | 0.9374 | 0.0188 | 163 | 0.9523 | 0.0373 | 0.0342 | 0.6889 | 0.0230 | 0.0118 |
| `voxel` | 4 | 0.8 | 0.9103 | 0.0181 | 159 | 0.9523 | 0.0373 | 0.0342 | 0.6889 | 0.0230 | 0.0118 |
| `voxel` | 4 | 0.9 | 0.8949 | 0.0177 | 156 | 0.9523 | 0.0373 | 0.0342 | 0.6889 | 0.0230 | 0.0118 |
| `voxel` | 8 | 0.5 | 0.9481 | 0.0197 | 48 | 0.9264 | 0.0600 | 0.0508 | 0.7013 | 0.0425 | 0.0390 |
| `voxel` | 8 | 0.7 | 0.9097 | 0.0186 | 45 | 0.9264 | 0.0600 | 0.0508 | 0.7013 | 0.0425 | 0.0390 |
| `voxel` | 8 | 0.8 | 0.8300 | 0.0165 | 42 | 0.9264 | 0.0600 | 0.0508 | 0.7013 | 0.0425 | 0.0390 |
| `voxel` | 8 | 0.9 | 0.8176 | 0.0162 | 41 | 0.9264 | 0.0600 | 0.0508 | 0.7013 | 0.0425 | 0.0390 |
| `voxel` | 16 | 0.5 | 0.9103 | 0.0227 | 15 | 0.8004 | 0.1267 | 0.0130 | 0.5449 | 0.1724 | 0.1104 |
| `voxel` | 16 | 0.7 | 0.6163 | 0.0124 | 12 | 0.8004 | 0.1267 | 0.0130 | 0.5449 | 0.1724 | 0.1104 |
| `voxel` | 16 | 0.8 | 0.5838 | 0.0116 | 11 | 0.8004 | 0.1267 | 0.0130 | 0.5449 | 0.1724 | 0.1104 |
| `voxel` | 16 | 0.9 | 0.5838 | 0.0116 | 11 | 0.8004 | 0.1267 | 0.0130 | 0.5449 | 0.1724 | 0.1104 |

## High-Purity Recall At Threshold 0.9

| class | counterpart | source | s | recall | candidate point frac | candidate regions | mean purity | counterpart contam | counterpart-majority frac |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `desk` | `table` | `pred_cc` | 4 | 0.4269 | 0.0469 | 7 | 0.8628 | 0.0000 | 0.0000 |
| `desk` | `table` | `pred_cc` | 8 | 0.4006 | 0.0453 | 5 | 0.7886 | 0.0001 | 0.0000 |
| `desk` | `table` | `pred_cc` | 16 | 0.4570 | 0.0557 | 1 | 0.6986 | 0.0001 | 0.0000 |
| `desk` | `table` | `voxel` | 4 | 0.9133 | 0.1003 | 1102 | 0.9624 | 0.0000 | 0.0000 |
| `desk` | `table` | `voxel` | 8 | 0.8027 | 0.0884 | 265 | 0.9086 | 0.0000 | 0.0000 |
| `desk` | `table` | `voxel` | 16 | 0.4315 | 0.0477 | 43 | 0.8119 | 0.0000 | 0.0000 |
| `door` | `wall` | `pred_cc` | 4 | 0.9166 | 0.0797 | 15 | 0.8881 | 0.0819 | 0.0599 |
| `door` | `wall` | `pred_cc` | 8 | 0.9003 | 0.0792 | 11 | 0.8598 | 0.0950 | 0.0708 |
| `door` | `wall` | `pred_cc` | 16 | 0.0778 | 0.0065 | 6 | 0.8097 | 0.0930 | 0.0521 |
| `door` | `wall` | `voxel` | 4 | 0.9178 | 0.0764 | 1991 | 0.9671 | 0.0205 | 0.0114 |
| `door` | `wall` | `voxel` | 8 | 0.8663 | 0.0721 | 535 | 0.9356 | 0.0399 | 0.0425 |
| `door` | `wall` | `voxel` | 16 | 0.7426 | 0.0620 | 124 | 0.8806 | 0.0546 | 0.0398 |
| `picture` | `wall` | `pred_cc` | 4 | 0.8259 | 0.0173 | 8 | 0.8109 | 0.0728 | 0.0224 |
| `picture` | `wall` | `pred_cc` | 8 | 0.8223 | 0.0169 | 3 | 0.8082 | 0.0771 | 0.0419 |
| `picture` | `wall` | `pred_cc` | 16 | 0.1777 | 0.0035 | 2 | 0.5990 | 0.2344 | 0.1724 |
| `picture` | `wall` | `voxel` | 4 | 0.8949 | 0.0177 | 156 | 0.9523 | 0.0373 | 0.0342 |
| `picture` | `wall` | `voxel` | 8 | 0.8176 | 0.0162 | 41 | 0.9264 | 0.0600 | 0.0508 |
| `picture` | `wall` | `voxel` | 16 | 0.5838 | 0.0116 | 11 | 0.8004 | 0.1267 | 0.0130 |

## Interpretation Gate

- PVD go: a weak class has high-purity recall at `thr>=0.8` with a reasonably small candidate point fraction.
- Fine-only go: `voxel s4` has high recall but `pred_cc` / coarse sizes collapse; proposal-first needs learned object-quality masks, not region averaging.
- No-go: even `voxel s4` has low high-purity recall; proposal-first is unlikely to recover the oracle headroom under this feature family.

## Files

- Detail CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/proposal_recall_smoke/proposal_recall_detail.csv`
