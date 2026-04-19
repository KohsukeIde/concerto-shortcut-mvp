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

## Configuration

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-ft-lora-origin-e100.py`
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
| origin plain LoRA r8 e100 best | 0.7749 | 0.8766 | 0.9174 | best epoch `68`; custom batch-size-1 eval |

## Weak-Class Comparison

| class | decoder IoU | LoRA IoU | delta | decoder acc | LoRA acc | delta acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| picture | 0.4217 | 0.4303 | +0.0087 | 0.5472 | 0.5968 | +0.0496 |
| counter | 0.7044 | 0.6522 | -0.0522 | 0.8209 | 0.8364 | +0.0156 |
| cabinet | 0.7318 | 0.7104 | -0.0214 | 0.8337 | 0.8307 | -0.0030 |
| desk | 0.7096 | 0.7026 | -0.0070 | 0.8676 | 0.8572 | -0.0104 |
| table | 0.7896 | 0.7813 | -0.0083 | 0.8491 | 0.8527 | +0.0036 |
| sink | 0.7199 | 0.6850 | -0.0349 | 0.8594 | 0.8207 | -0.0387 |
| shower curtain | 0.8055 | 0.7624 | -0.0431 | 0.8927 | 0.8214 | -0.0712 |
| door | 0.7715 | 0.7428 | -0.0287 | 0.9129 | 0.8996 | -0.0133 |
| wall | 0.8793 | 0.8655 | -0.0138 | 0.9356 | 0.9224 | -0.0132 |

## Dominant Confusions

| target -> pred | decoder fraction | LoRA fraction | delta |
| --- | ---: | ---: | ---: |
| picture -> wall | 0.4310 | 0.3867 | -0.0443 |
| counter -> cabinet | 0.0896 | 0.0909 | +0.0013 |
| desk -> table | 0.0312 | 0.0346 | +0.0034 |
| desk -> wall | 0.0283 | 0.0281 | -0.0001 |
| sink -> cabinet | 0.0833 | 0.0860 | +0.0027 |
| sink -> counter | 0.0365 | 0.0688 | +0.0323 |
| shower curtain -> wall | 0.0366 | 0.1297 | +0.0931 |
| door -> wall | 0.0671 | 0.0749 | +0.0078 |

## Interpretation

Plain encoder-side LoRA is a useful control, but not a positive method yet. It changes the failure mode in the expected direction for `picture`:

- `picture` IoU improves from `0.4217` to `0.4303`.
- `picture -> wall` falls from `0.4310` to `0.3867`.

However, the aggregate result is worse than the origin decoder-probe reference (`0.7749` vs `0.7888` mIoU), and several weak classes degrade (`counter`, `cabinet`, `sink`, `shower curtain`, `door`). This means plain encoder-side LoRA can move the `picture/wall` geometry, unlike CIDA, but it is not class-safe and does not solve the full multiclass decision surface.

Decision: treat this as evidence that encoder-side adaptation can affect the target confusion, but do not continue plain LoRA as-is. The next method must preserve aggregate and weak-class geometry while selectively repairing `picture/wall`, likely through a stronger full-distribution/class-safety constraint or a more faithful official LoRA/decoder setup.

## Artifacts

- Training log: `data/logs/abciq/scannet_semseg_133383.qjcm.log`
- Node logs: `data/runs/scannet_lora_origin/logs/multinode/133383.qjcm_scannet-lora-origin-r8-e100_20260420_011807/logs/`
- Best checkpoint: `exp/concerto/scannet-lora-origin-r8-e100/model/model_best.pth`
- Classwise outputs: `data/runs/scannet_lora_origin/classwise/`
