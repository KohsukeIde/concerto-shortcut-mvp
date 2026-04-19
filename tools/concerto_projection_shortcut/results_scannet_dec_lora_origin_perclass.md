# ScanNet Origin Decoder-Capacity-Matched LoRA Control

## Summary

This run tests the next matched-family question after the origin same-head
linear LoRA control:

> Does the LoRA gain survive when the head capacity is matched to the origin
> decoder-probe family?

The current answer is **no for this plain setup**. Decoder-capacity-matched LoRA
trains and evaluates correctly, but it does not beat the origin decoder-probe
baseline. It is a near tie on aggregate mIoU and slightly worse on the main
`picture -> wall` failure.

This is **not** the large-video checkpoint line. It uses the Concerto paper
origin backbone `data/weights/concerto/concerto_base_origin.pth`.

## Jobs

| purpose | job id | resource | status | notes |
| --- | --- | --- | --- | --- |
| config smoke | `133388.qjcm` | `rt_QF=1`, `00:20:00` | pass | checkpoint load, LoRA injection, decoder train/val loop pass |
| 100 epoch decoder+LoRA train | `133389.qjcm` | `rt_QF=2`, `01:00:00` | pass | 2 nodes / 8 H100, global batch 64, best built-in val mIoU `0.7859` |
| best-checkpoint classwise eval | `133390.qjcm` | `rt_QF=1`, `00:20:00` | pass | batch size 1 full ScanNet val |

## Configuration

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-lora-origin-e100.py`
- Decoder baseline config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec.py`
- Backbone: `data/weights/concerto/concerto_base_origin.pth`
- Model: `DefaultLORASegmentorV2`, decoder-probe family with `enc_mode=False`
  and `backbone_out_channels=64`
- Trainable parameters: decoder/head plus LoRA on encoder `qkv`
- LoRA: rank `8`, alpha `16`, dropout `0.1`
- Loss: plain decoder-probe semseg objective, no weak-class weighting, no
  pairwise auxiliary, no class-safety regularizer
- Precise evaluator disabled during training; per-class/confusion is run
  separately with batch size 1.

## Overall

| model | mIoU | mAcc | allAcc | notes |
| --- | ---: | ---: | ---: | --- |
| origin same-head no-LoRA linear e100 best | 0.7617 | 0.8646 | 0.9124 | linear-head reference from `results_scannet_lora_origin_perclass.md` |
| origin plain LoRA linear r8 e100 best | 0.7749 | 0.8766 | 0.9174 | matched linear-head positive control |
| origin decoder probe e100 | 0.7888 | 0.8813 | 0.9243 | decoder-capacity baseline from `results_scannet_decoder_probe_origin.md` |
| origin decoder+LoRA r8 e100 best | 0.7860 | 0.8694 | 0.9229 | this run |

The linear-head LoRA gain is real in the matched linear family
(`0.7617 -> 0.7749` mIoU). Under decoder-capacity matching, plain LoRA does not
improve the decoder baseline (`0.7888 -> 0.7860` mIoU). This suggests the
linear-family gain is largely absorbed by decoder capacity, or that this
decoder+LoRA interaction is weakly negative in the current setup.

## Weak-Class Comparison

| class | decoder IoU | decoder+LoRA IoU | delta |
| --- | ---: | ---: | ---: |
| picture | 0.4217 | 0.4204 | -0.0013 |
| counter | 0.7044 | 0.6633 | -0.0411 |
| cabinet | 0.7318 | 0.7223 | -0.0096 |
| desk | 0.7096 | 0.6988 | -0.0109 |
| table | 0.7896 | 0.7889 | -0.0008 |
| sink | 0.7199 | 0.7009 | -0.0190 |
| shower curtain | 0.8055 | 0.7939 | -0.0116 |
| door | 0.7715 | 0.7761 | +0.0046 |
| wall | 0.8793 | 0.8755 | -0.0038 |

## Dominant Confusions

| target -> pred | decoder fraction | decoder+LoRA fraction | delta |
| --- | ---: | ---: | ---: |
| picture -> wall | 0.4310 | 0.4387 | +0.0077 |
| wall -> picture | 0.0052 | 0.0049 | -0.0003 |
| counter -> cabinet | 0.0896 | 0.0981 | +0.0085 |
| desk -> table | 0.0312 | 0.0367 | +0.0055 |
| sink -> cabinet | 0.0833 | 0.1020 | +0.0187 |
| sink -> counter | 0.0365 | 0.0849 | +0.0484 |
| shower curtain -> wall | 0.0366 | 0.0945 | +0.0579 |
| door -> wall | 0.0671 | 0.0990 | +0.0319 |

## Interpretation

This completes the intended 2x2 comparison:

| head family | no LoRA | LoRA | LoRA effect |
| --- | ---: | ---: | ---: |
| linear head | 0.7617 | 0.7749 | +0.0132 |
| decoder head | 0.7888 | 0.7860 | -0.0028 |

The key update is:

- Plain encoder-side LoRA is a **matched-head positive control** in the linear
  family.
- The same plain LoRA idea does **not** produce a positive result once decoder
  capacity is matched.
- The main `picture -> wall` failure is not improved by decoder+LoRA
  (`0.4310 -> 0.4387`), and several weak-class confusions worsen.

Decision: do not launch class-safety or weak-class variants of this exact
decoder+LoRA setup without a new hypothesis. A reasonable next decision is
whether to test the official Concerto LoRA recipe if it differs materially from
this config, or to treat the current method line as diagnostic rather than a
positive method path.

## Artifacts

- Training log: `data/logs/abciq/scannet_semseg_133389.qjcm.log`
- Node logs:
  `data/runs/scannet_dec_lora_origin/logs/multinode/133389.qjcm_scannet-dec-lora-origin-r8-e100_20260420_030301/logs/`
- Best checkpoint:
  `exp/concerto/scannet-dec-lora-origin-r8-e100/model/model_best.pth`
- Classwise outputs:
  `data/runs/scannet_dec_lora_origin/classwise/`
