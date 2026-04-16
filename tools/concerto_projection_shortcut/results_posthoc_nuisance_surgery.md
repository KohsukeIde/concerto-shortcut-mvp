# Frozen Posthoc Nuisance Surgery Results

Updated: 2026-04-16 23:50 JST

## Bottom Line

The first frozen-feature post-training pilot completed on the same-stage
original e025 checkpoint. It tests whether a post-training editor can remove
coordinate / height nuisance directions while keeping the ScanNet proxy linear
probe near the original e025 reference.

Result:
- Original e025 same-stage ScanNet proxy reference: `0.5531` last / `0.5531`
  best mIoU.
- SPLICE-3D and HLNS stay within `0.0022` mIoU of that reference.
- This is not a win, but it is a useful cheap fallback: the Concerto backbone is
  frozen, only the editor and linear head are fitted, and no scratch pretraining
  is required.

## Implementation

Added frozen post-training surgery components:
- `pointcept/models/posthoc_nuisance_surgery.py`
  - `PosthocEditedSegmentorV2`
  - `Splice3DEditor`
  - `HLNSEditor`
- `tools/concerto_projection_shortcut/extract_frozen_backbone_features.py`
- `tools/concerto_projection_shortcut/fit_splice3d_frozen.py`
- `tools/concerto_projection_shortcut/fit_hlns_frozen.py`
- `tools/concerto_projection_shortcut/run_posthoc_surgery_chain.sh`
- `tools/concerto_projection_shortcut/run_posthoc_surgery_suite.sh`
- `tools/concerto_projection_shortcut/submit_posthoc_surgery_abciq_qf.sh`
- `configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin-splice3d-frozen.py`
- `configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin-hlns-frozen.py`

The editor is inserted after the frozen Concerto backbone and before the linear
segmentation head. The pretraining path is untouched.

The current HLNS implementation is a channel-group proxy, not true attention
head surgery, because the current ScanNet linear-probe path exposes the final
`point.feat` but not per-attention-head activations.

## Pilot Run

Run:
- job: `132608.qjcm`
- resource: ABCI-Q `rt_QF=1`
- checkpoint:
  `exp/concerto/arkit-full-original-long-e025-qf32-continue/model/model_last.pth`
- result root:
  `data/runs/posthoc_surgery_e025pilot/original-long-e025-qf32`
- feature cache:
  `data/runs/posthoc_surgery_e025pilot/original-long-e025-qf32/cache/semseg-ptv3-base-v1m1-0a-scannet-lin-proxy_r1024_mt512_mv128`
- feature rows:
  - train: 524288
  - val: 131072
  - dim: 1232

ScanNet proxy linear results:

| method | nuisance | last mIoU | best mIoU | mAcc | allAcc | delta vs original e025 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original e025 reference | - | 0.5531 | 0.5531 | 0.6832 | 0.8070 | 0.0000 |
| SPLICE-3D | height+xyz | 0.5509 | 0.5509 | 0.6816 | 0.8065 | -0.0022 |
| SPLICE-3D | height | 0.5510 | 0.5510 | 0.6811 | 0.8064 | -0.0021 |
| HLNS channel-group proxy | height+xyz | 0.5515 | 0.5515 | 0.6834 | 0.8065 | -0.0016 |

Editor fit diagnostics:

| method | nuisance | harm rank / selected groups | nuisance energy before val | nuisance energy after val | proxy val acc before | proxy val acc after | edit energy val |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SPLICE-3D | height+xyz | rank 3 | 0.000939 | ~0 | 0.807097 | 0.806967 | - |
| SPLICE-3D | height | rank 1 | 0.000737 | ~0 | 0.807097 | 0.807007 | - |
| HLNS channel-group proxy | height+xyz | groups 0,1,2,3 | - | - | 0.807097 | 0.807127 | 0.003607 |

## Readout

- SPLICE-3D behaves as intended in the pilot: nuisance energy is driven near
  zero and the proxy classifier is essentially unchanged.
- Height-only SPLICE is slightly simpler and performs almost identically to
  height+xyz.
- HLNS initially looked worse at early eval but recovered to the best final
  mIoU among the three pilot arms. Since it is only a channel-group proxy, it
  should stay analysis-first until we can inspect whether nuisance/task
  directions are genuinely localized.
- None of the posthoc arms beats the same-stage original e025 reference. The
  useful result is that frozen post-training can keep downstream performance
  near-tied without spending scratch-pretraining resources.

Recommended next step:
- Keep the e050 original/v1b continuation as the mainline decision.
- In parallel, run the best posthoc editors on the e050 original and e050 v1b
  checkpoints after their follow-ups finish, using the same cached ScanNet
  extraction strategy.
