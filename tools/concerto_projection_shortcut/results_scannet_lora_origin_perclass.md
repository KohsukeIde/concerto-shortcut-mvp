# ScanNet Origin Plain LoRA Per-Class Control

## Summary

Plain same-checkpoint LoRA was run as a decisive encoder-side adaptation control after cached-feature post-hoc methods and CIDA failed to recover oracle/actionability headroom.

This is **not** the large-video checkpoint line. It uses the Concerto paper origin backbone `data/weights/concerto/concerto_base_origin.pth` through `DefaultLORASegmentorV2`.

## Jobs

| purpose | job id | resource | status | notes |
| --- | --- | --- | --- | --- |
| 1 epoch smoke | `133382.qjcm` | `rt_QF=1`, `00:20:00` | pass | checkpoint load, LoRA injection, train/val loop pass; job was stopped after unnecessary precise eval had already saved checkpoints |
| 100 epoch LoRA train | `133383.qjcm` | `rt_QF=2`, `01:00:00` | pass | 2 nodes / 8 H100, global batch 64, precise eval disabled |
| best-checkpoint classwise eval | `133384.qjcm` | `rt_QF=1`, `00:20:00` | pass | batch size 1 full ScanNet val |
| 1 epoch same-head no-LoRA smoke | `133385.qjcm` | `rt_QF=1`, `00:12:00` | pass | same config/head/schedule with `use_lora=False` |
| 100 epoch same-head no-LoRA train | `133386.qjcm` | `rt_QF=2`, `00:45:00` | pass | 2 nodes / 8 H100, global batch 64, precise eval disabled |
| same-head no-LoRA classwise eval | `133387.qjcm` | `rt_QF=1`, `00:20:00` | pass | batch size 1 full ScanNet val |

## Configuration

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-ft-lora-origin-e100.py`
- Same-head no-LoRA config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-lin-origin-e100.py`
- Backbone: `data/weights/concerto/concerto_base_origin.pth`
- Model: `DefaultLORASegmentorV2`, PTv3 base encoder mode, linear segmentation head
- Trainable parameters: segmentation head plus LoRA on encoder `qkv` (`228,864` LoRA params in smoke)
- LoRA: rank `8`, alpha `16`, dropout `0.1`
- Optimizer/scheduler follow the Pointcept issue guidance: AdamW `lr=0.001`, block/LoRA group `lr=0.0001`, OneCycleLR
- Precise evaluator disabled for this control; per-class/confusion is run separately with batch size 1.

## Overall

| model | mIoU | mAcc | allAcc | notes |
| --- | ---: | ---: | ---: | --- |
| origin decoder probe e100 | 0.7888 | 0.8813 | 0.9243 | reference from `results_scannet_decoder_probe_origin.md` |
| origin same-head no-LoRA linear e100 best | 0.7617 | 0.8646 | 0.9124 | same linear-head family as LoRA, `use_lora=False`; custom batch-size-1 eval |
| origin plain LoRA r8 e100 best | 0.7749 | 0.8766 | 0.9174 | best epoch `68`; custom batch-size-1 eval |

Same-head takeaway: plain LoRA is positive relative to the no-LoRA linear-head baseline (`+0.0132` mIoU, `+0.0120` mAcc, `+0.0050` allAcc). It remains below the stronger decoder-probe reference (`-0.0139` mIoU), so decoder-probe and LoRA numbers should not be conflated.

## Weak-Class Comparison

| class | decoder IoU | linear IoU | LoRA IoU | LoRA-linear | LoRA-decoder |
| --- | ---: | ---: | ---: | ---: | ---: |
| picture | 0.4217 | 0.4078 | 0.4303 | +0.0225 | +0.0087 |
| counter | 0.7044 | 0.6554 | 0.6522 | -0.0032 | -0.0522 |
| cabinet | 0.7318 | 0.6818 | 0.7104 | +0.0286 | -0.0214 |
| desk | 0.7096 | 0.6854 | 0.7026 | +0.0172 | -0.0070 |
| table | 0.7896 | 0.7709 | 0.7813 | +0.0105 | -0.0083 |
| sink | 0.7199 | 0.6821 | 0.6850 | +0.0029 | -0.0349 |
| shower curtain | 0.8055 | 0.7402 | 0.7624 | +0.0222 | -0.0431 |
| door | 0.7715 | 0.7330 | 0.7428 | +0.0098 | -0.0287 |
| wall | 0.8793 | 0.8624 | 0.8655 | +0.0031 | -0.0138 |

## Dominant Confusions

| target -> pred | decoder fraction | linear fraction | LoRA fraction | LoRA-linear | LoRA-decoder |
| --- | ---: | ---: | ---: | ---: | ---: |
| picture -> wall | 0.4310 | 0.4151 | 0.3867 | -0.0284 | -0.0443 |
| wall -> picture | 0.0052 | 0.0060 | 0.0070 | +0.0010 | +0.0019 |
| counter -> cabinet | 0.0896 | 0.0872 | 0.0909 | +0.0037 | +0.0013 |
| desk -> table | 0.0312 | 0.0404 | 0.0346 | -0.0058 | +0.0034 |
| sink -> cabinet | 0.0833 | 0.0829 | 0.0860 | +0.0031 | +0.0027 |
| sink -> counter | 0.0365 | 0.0711 | 0.0688 | -0.0023 | +0.0323 |
| shower curtain -> wall | 0.0366 | 0.1389 | 0.1297 | -0.0092 | +0.0931 |
| door -> wall | 0.0671 | 0.0834 | 0.0749 | -0.0085 | +0.0078 |

## Interpretation

Plain encoder-side LoRA is a useful positive control within the same linear-head family. It changes the failure mode in the expected direction for `picture`:

- Against the no-LoRA linear-head baseline, mIoU improves from `0.7617` to `0.7749`.
- `picture` IoU improves from `0.4078` to `0.4303`.
- `picture -> wall` falls from `0.4151` to `0.3867`.

The earlier decoder-probe comparison was head-capacity confounded: decoder probe uses a stronger trainable decoder, while this LoRA control uses a linear segmentation head. With the same linear-head baseline added, the correct reading is:

- Plain LoRA is positive relative to the same-head no-LoRA baseline.
- It is still below the origin decoder-probe reference (`0.7749` vs `0.7888` mIoU), so the decoder/LoRA family comparison remains unresolved.
- Encoder-side adaptation is the first family here that moves `picture -> wall` substantially while improving same-head aggregate mIoU.

Decision: do not use the decoder-probe gap alone to call plain LoRA damaging. The next job should compare within a matched family: either a more faithful official LoRA setup or a decoder-capacity-matched LoRA/control, then add class-safety only if the matched LoRA still damages specific weak classes.

## Artifacts

- Training log: `data/logs/abciq/scannet_semseg_133383.qjcm.log`
- Node logs: `data/runs/scannet_lora_origin/logs/multinode/133383.qjcm_scannet-lora-origin-r8-e100_20260420_011807/logs/`
- Best checkpoint: `exp/concerto/scannet-lora-origin-r8-e100/model/model_best.pth`
- Same-head no-LoRA checkpoint: `exp/concerto/scannet-lin-origin-e100/model/model_best.pth`
- Classwise outputs: `data/runs/scannet_lora_origin/classwise/`
