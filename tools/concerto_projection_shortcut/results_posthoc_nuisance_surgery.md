# Frozen Posthoc Nuisance Surgery Results

Updated: 2026-04-17 15:20 JST

## Bottom Line

The first frozen-feature post-training pilot completed on the same-stage
original e025 checkpoint. It tests whether a post-training editor can remove
coordinate / height nuisance directions while keeping the ScanNet proxy linear
probe near the original e025 reference.

Result:
- Original e025 same-stage ScanNet proxy reference: `0.5531` last / `0.5531`
  best mIoU.
- SPLICE-3D and HLNS stay within `0.0022` mIoU of that reference.
- The e025 stress downstream gate does not pass: clean is preserved, but no
  SPLICE-3D or Residual Recycling arm improves any stress condition by the
  required `+0.005` mIoU.
- This is not a win, but it is a useful cheap fallback: the Concerto backbone is
  frozen, only the editor and linear head are fitted, and no scratch pretraining
  is required.

## Implementation

Added frozen post-training surgery components:
- `pointcept/models/posthoc_nuisance_surgery.py`
  - `PosthocEditedSegmentorV2`
  - `Splice3DEditor`
  - `HLNSEditor`
  - `ResidualRecyclingEditor`
- `tools/concerto_projection_shortcut/extract_frozen_backbone_features.py`
- `tools/concerto_projection_shortcut/fit_splice3d_frozen.py`
- `tools/concerto_projection_shortcut/fit_hlns_frozen.py`
- `tools/concerto_projection_shortcut/fit_residual_recycling_frozen.py`
- `tools/concerto_projection_shortcut/run_posthoc_surgery_chain.sh`
- `tools/concerto_projection_shortcut/run_posthoc_surgery_suite.sh`
- `tools/concerto_projection_shortcut/submit_posthoc_surgery_abciq_qf.sh`
- `tools/concerto_projection_shortcut/eval_scannet_semseg_stress.py`
- `tools/concerto_projection_shortcut/run_posthoc_stress_suite.sh`
- `tools/concerto_projection_shortcut/submit_posthoc_stress_abciq_qf.sh`
- `configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin-splice3d-frozen.py`
- `configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin-hlns-frozen.py`
- `configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin-recycle-frozen.py`

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
| Residual Recycling | height+xyz, coord9 | 0.5506 | 0.5506 | 0.6824 | 0.8062 | -0.0025 |

Editor fit diagnostics:

| method | nuisance | harm rank / selected groups | nuisance energy before val | nuisance energy after val | proxy val acc before | proxy val acc after | edit energy val |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SPLICE-3D | height+xyz | rank 3 | 0.000939 | ~0 | 0.807097 | 0.806967 | - |
| SPLICE-3D | height | rank 1 | 0.000737 | ~0 | 0.807097 | 0.807007 | - |
| HLNS channel-group proxy | height+xyz | groups 0,1,2,3 | - | - | 0.807097 | 0.807127 | 0.003607 |
| Residual Recycling | height+xyz, coord9 | rank 3 | 0.000939 | 0.000000068 | 0.807097 | 0.807037 | - |

Residual Recycling details:
- job: `132780.qjcm`
- result root:
  `data/runs/posthoc_surgery_e025pilot/original-long-e025-qf32/recycle_height_xyz_coord9_g1.0_r1.0`
- linear probe exp:
  `data/runs/posthoc_surgery_e025pilot/exp/posthoc-original-long-e025-qf32-recycle_height_xyz_coord9_g1.0_r1.0-lin`
- linear probe result: `0.5506 / 0.5506` mIoU.
- The fitted residual injection is very small:
  `train_residual_mse_deleted=0.01470065`,
  `train_residual_mse_recycled_with_deleted_head=0.01469802`.
  This suggests the simple coord9 residual write-back is not finding much
  recoverable task signal.

## Stress Downstream Gate

Run:
- SPLICE-3D / original stress job: `132775.qjcm`
- Residual Recycling stress job: `132789.qjcm`
- roots:
  - `data/runs/posthoc_stress_e025pilot`
  - `data/runs/posthoc_stress_e025pilot_recycle`

Pass criterion:
- clean mIoU within `-0.005` of original e025, and
- at least one stress condition improves by `+0.005` mIoU.

Results:

| method | clean | delta | local_surface_destroy | delta | z_flip | delta | xy_swap | delta | roll_90_x | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0.5531 | 0.0000 | 0.0359 | 0.0000 | 0.0988 | 0.0000 | 0.5531 | 0.0000 | 0.0435 | 0.0000 |
| SPLICE-3D height | 0.5515 | -0.0015 | 0.0358 | -0.0001 | 0.1015 | +0.0027 | 0.5515 | -0.0017 | 0.0438 | +0.0003 |
| SPLICE-3D height+xyz | 0.5511 | -0.0020 | 0.0374 | +0.0015 | 0.1017 | +0.0029 | 0.5511 | -0.0020 | 0.0433 | -0.0002 |
| Residual Recycling coord9 | 0.5509 | -0.0021 | 0.0378 | +0.0019 | 0.1011 | +0.0023 | 0.5502 | -0.0029 | 0.0444 | +0.0009 |

Decision:
- Stage 1 does not pass. All posthoc arms preserve clean mIoU within the
  tolerance, but the stress gains are too small.
- SPLICE-3D remains a useful cheap diagnostic/fallback, but the current
  post-training route is not yet a mainline NeurIPS claim.
- Residual Recycling in this simple global coord9 form should not be scaled up
  without a stronger new hypothesis, because the learned write-back is tiny and
  does not change the stress conclusion.

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
- Do not spend more points on broad posthoc sweeps yet.
- If continuing posthoc, the next hypothesis should change the residual channel,
  not just tune SPLICE gamma: e.g. class-conditional residual recycling,
  boundary/local-geometry descriptors, or an analysis-first head/channel
  localization test.
