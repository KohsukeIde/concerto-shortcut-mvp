# ScanNet Origin LP-FT / Class-Safe LoRA Follow-Up

## Summary

This follow-up tests the next method family after prototype/kNN readout was
no-go and after matched-head plain LoRA proved positive in the linear-head
family. The question is whether a linear-probe warm start and class-safety
constraints improve the same origin LoRA line.

Result: **LP-FT plain is a tiny positive over the prior plain LoRA mIoU, but not
a decisive new method. Class-safe LoRA is no-go.**

## Jobs

| purpose | job id | resource | status | notes |
| --- | --- | --- | --- | --- |
| prototype-only readout | `133392.qjcm` | `rt_QF=1` | pass | no-go; see `results_prototype_readout.md` |
| kNN small readout | `133396.qjcm` | `rt_QF=1` | pass | no-go; see `results_knn_readout_small.md` |
| LP-FT plain train | `133397.qjcm` | `rt_QF=2`, `02:00:00` | pass | warm-start from `scannet-lin-origin-e100`, plain CE |
| LP-FT plain classwise eval | `133399.qjcm` | `rt_QF=1`, `00:30:00` | pass | first eval submission `133398.qjcm` failed from path-style config arg |
| LP-FT class-safe train | `133395.qjcm` | `rt_QF=2`, `02:00:00` | pass | weak CE + non-weak KL + distribution KL |
| LP-FT class-safe classwise eval | `133400.qjcm` | `rt_QF=1`, `00:30:00` | pass | best checkpoint eval |

## Configuration

Common:

- Backbone/source: `concerto_base_origin.pth`
- Model family: `DefaultLORASegmentorV2` / `DefaultClassSafeLORASegmentorV2`
- Head family: PTv3 base encoder-mode linear segmentation head
- LoRA: rank 8 qkv LoRA, same as prior plain LoRA control

LP-FT plain:

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-ft-lora-origin-e100.py`
- Init weight: `exp/concerto/scannet-lin-origin-e100/model/model_best.pth`
- Loss: standard 20-way CE only

LP-FT class-safe:

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0f-scannet-ft-lora-origin-lpft-classsafe-e100.py`
- Init/anchor weight: `exp/concerto/scannet-lin-origin-e100/model/model_best.pth`
- Loss: CE + weak CE + non-weak KL-to-anchor + batch distribution KL
- Weak classes: `picture`, `counter`, `sink`, `shower curtain`, `door`
- Default weights: weak CE `0.2`, non-weak KL `0.05`, distribution KL `0.02`, temperature `2.0`

## Overall

| model | mIoU | mAcc | allAcc | picture IoU | picture -> wall | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| same-head no-LoRA linear | 0.7617 | 0.8646 | 0.9124 | 0.4078 | 0.4151 | prior matched baseline |
| plain LoRA r8 | 0.7749 | 0.8766 | 0.9174 | 0.4303 | 0.3867 | prior matched positive control |
| LP-FT plain r8 | 0.7771 | 0.8718 | 0.9188 | 0.4275 | 0.4222 | small mIoU gain, worse picture/wall than prior plain LoRA |
| LP-FT class-safe r8 | 0.7706 | 0.8779 | 0.9148 | 0.4077 | 0.3554 | lowers picture->wall but hurts mIoU and picture IoU |

## Selected Class Metrics

| class | plain LoRA IoU | LP-FT plain IoU | class-safe IoU |
| --- | ---: | ---: | ---: |
| picture | 0.4303 | 0.4275 | 0.4077 |
| counter | 0.6522 | 0.6644 | 0.6356 |
| sink | 0.6850 | 0.6910 | 0.6761 |
| shower curtain | 0.7624 | 0.7526 | 0.7670 |
| door | 0.7428 | 0.7510 | 0.7196 |
| wall | 0.8655 | 0.8682 | 0.8609 |

## Interpretation

- Prototype and kNN readout did not recover oracle/actionability headroom.
  They slightly reduce `picture -> wall`, but picture IoU and mIoU barely move.
- LP-FT plain is the strongest follow-up in this batch by mIoU:
  `0.7749 -> 0.7771` relative to the prior same-head plain LoRA control.
  The gain is small and does not improve the central `picture -> wall` failure.
- Class-safe LoRA validates the expected trade-off:
  it reduces `picture -> wall` substantially (`0.4222/0.3867`-level references
  down to `0.3554`), but it over-adjusts the multiclass geometry and loses both
  mIoU and picture IoU. This is not a positive method.

Decision:

- Treat retrieval/prototype as no-go under this protocol.
- Treat class-safe LP-FT as no-go in its current weights.
- LP-FT warm start is a weak positive control, not yet paper-grade.
- Do not launch more broad method sweeps without a sharper hypothesis. If this
  line continues, the next ablation should be a very small LP-FT sweep around
  plain CE / milder class-safety, not another pair-emphasis or cached-feature
  reranker.

## Artifacts

- Prototype result: `tools/concerto_projection_shortcut/results_prototype_readout.md`
- kNN result: `tools/concerto_projection_shortcut/results_knn_readout_small.md`
- LP-FT plain classwise outputs: `data/runs/scannet_lora_lpft_origin/classwise/`
- Class-safe classwise outputs: `data/runs/scannet_lora_classsafe_origin/classwise/`
- LP-FT plain checkpoint: `exp/concerto/scannet-lora-origin-lpft-plain-e100/model/model_best.pth`
- Class-safe checkpoint: `exp/concerto/scannet-lora-origin-lpft-classsafe-e100/model/model_best.pth`
