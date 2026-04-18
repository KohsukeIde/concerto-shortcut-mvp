# Concerto Shortcut Current Status

This file is the single entry point for the current state of the shortcut
investigation.

## Current Bottom Line

- Strongest supported claim:
  - The released Concerto `enc2d` objective admits a strong coordinate shortcut.
- Scope of that claim:
  - Objective-level evidence on `ARKitScenes`.
  - The ScanNet continuation proxy shows measurable but limited downstream
    relevance.
  - It does not support the stronger claim that most of Concerto downstream gain
    is explained by the coordinate shortcut.
- Current action:
  - The minimal full-removal fix attempt, `projres_v1a`, completed its ABCI-Q
    gate and is no-go on ScanNet linear.
  - The factorized partial-removal follow-up, `projres_v1b`, also completed its
    ABCI-Q smoke, continuation, stress, and ScanNet linear gates.
  - The selective-prior follow-up, `projres_v1c`, completed prior fitting,
    smoke, continuation, stress, and ScanNet linear gates.
  - Result: v1b improves over v1a and `no-enc2d-renorm`, v1c does not improve
    over v1b, and both remain below the original continuation; no strong-go for
    fine-tuning.
  - A point-conscious long-horizon e025 same-stage check completed for original
    and v1b `combo-b075-a001`. v1b essentially ties the same-stage original
    reference but does not beat it: 0.5526 vs 0.5531 ScanNet proxy mIoU.
  - A frozen-feature post-training pilot completed for SPLICE-3D / HLNS on the
    original e025 checkpoint. All three posthoc arms are near the original e025
    ScanNet proxy mIoU without updating the Concerto backbone.
  - The posthoc e025 stress downstream gate also completed. SPLICE-3D and
    Residual Recycling preserve clean mIoU within `-0.005`, but none improves a
    stress condition by `+0.005`; Stage 1 is no-go as a mainline claim.
  - The Step 1 local-geometry smoke for UGNR completed. `geom_local9` concat
    and residual-expert variants do not improve the offline frozen-cache
    baseline by the required `+0.003` mIoU, so UGNR should not be launched with
    this descriptor.
  - The attempted fresh same-stage e050 original/v1b runs did not produce valid
    e050 checkpoints. Both hit walltime at epoch 6 after delayed node startup.
  - Step 0 official-checkpoint causal battery completed on the released
    `pretrain-concerto-v1m1-2-large-video.pth` checkpoint for ARKit val and
    Concerto ScanNet val. Target swaps strongly increase enc2d loss on both
    datasets, so the signal is not an ARKit-continued artifact. This checkpoint
    is the large-video variant, so it is now treated as cross-variant evidence,
    not as the final Concerto paper main-variant diagnostic.
  - Main-variant Step 0/0.5 completed with `concerto_base_origin.pth` as a
    frozen backbone plus short-refit enc2d alignment projection on the six
    indoor Concerto datasets. Target swaps remain clearly positive on ARKit and
    ScanNet. The coord-MLP rival is weak on ARKit and partial on ScanNet, so it
    is usable only as a lower-bound rival, not as a full shortcut proxy.
  - A target-corruption distance diagnostic completed for the main-variant
    ARKit/ScanNet battery. ARKit cross-scene targets are not closer to the
    original targets than cross-image targets, so the lower ARKit cross-scene
    loss is not explained by `cos(t_original, t_corrupted)` alone.
  - SR-LoRA v5 Phase A completed its code smoke, 256-iteration pilot, 4-arm
    training matrix, and representative downstream follow-ups. Training is
    stable, but downstream is no-go. `m=0.1` gives only `+0.0009/+0.0015`
    ScanNet linear last/best mIoU and no stress gain; `m=0.2` increases margin
    pressure but gives `-0.0003/-0.0003` linear last/best mIoU and no stress
    gain. Do not launch the remaining matrix follow-ups without a new
    hypothesis.
  - Data and run outputs should live under repo-local `data/`.
  - Existing ScanNet is used through a symlink, not copied.
  - Do not run the optional fine-tune, e075/e100, or broad posthoc sweeps
    without a new decision criterion or hypothesis.

## Documentation Policy

- `CURRENT_STATUS.md` is the canonical high-level status document.
- When a stage finishes, the corresponding `results_*.md` / `results_*.csv`
  file should be updated and linked here.
- If a stage is still running, its current state is tracked here with the
  primary log path.

## Best Documents To Read First

1. Objective-level conclusion:
   - [results_arkit_full_causal.md](./results_arkit_full_causal.md)
2. Geometry-vs-coordinate stress result:
   - [results_arkit_full_stress_corrected.md](./results_arkit_full_stress_corrected.md)
3. Official checkpoint large-video ARKit/ScanNet causal battery:
   - [results_official_causal_battery.md](./results_official_causal_battery.md)
4. ScanNet continuation proxy:
   - [results_scannet_proxy_lin.md](./results_scannet_proxy_lin.md)
5. ProjRes v1 gate:
   - [results_projres_v1.md](./results_projres_v1.md)
6. Frozen post-training nuisance surgery:
   - [results_posthoc_nuisance_surgery.md](./results_posthoc_nuisance_surgery.md)
7. SR-LoRA v5 Phase A:
   - [results_sr_lora_phasea.md](./results_sr_lora_phasea.md)
8. Coordinate projection residual handoff:
   - [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md)
9. Short narrative summary:
   - [results_interim_summary_2026-04-06.md](./results_interim_summary_2026-04-06.md)
10. Reproduction / runner overview:
   - [README.md](./README.md)

## Official Large-Video Checkpoint Causal Battery

Source:
- [results_official_causal_battery.md](./results_official_causal_battery.md)

Setup:
- Weight: `data/weights/concerto/pretrain-concerto-v1m1-2-large-video.pth`.
- Scope: released full pretraining checkpoint for the large-video variant.
  These numbers remain useful as cross-variant evidence, but the main Concerto
  paper variant now needs the frozen-backbone head-refit diagnostic below.
- ARKit root: `data/arkitscenes_absmeta`.
- ScanNet root: `data/concerto_scannet_imagepoint_absmeta`.
- Smoke size: 32 batches per dataset and mode.

Key numbers:

| dataset | mode | enc2d loss | delta vs baseline |
| --- | --- | ---: | ---: |
| ARKit val | baseline | 2.737827 | 0.000000 |
| ARKit val | global target permutation | 3.324298 | +0.586471 |
| ARKit val | cross-image target swap | 3.371361 | +0.633534 |
| ARKit val | cross-scene target swap | 3.324546 | +0.586719 |
| ScanNet val | baseline | 3.416545 | 0.000000 |
| ScanNet val | global target permutation | 5.466386 | +2.049841 |
| ScanNet val | cross-image target swap | 5.502719 | +2.086174 |
| ScanNet val | cross-scene target swap | 5.467065 | +2.050520 |

Interpretation:
- The released large-video full pretraining checkpoint remains sensitive to
  target swaps on ARKit validation.
- The same signal is stronger on Concerto-preprocessed ScanNet validation.
- This supports the read that the issue is not only an ARKit-continued artifact.
- `concerto_base_origin.pth` is the Concerto paper main-variant backbone weight
  but does not contain enc2d / patch projection heads. The main-variant line
  therefore uses frozen-backbone head-refit rather than exact unreleased enc2d
  head recovery.

## Main-Variant Step 0/0.5 Plan

Current implementation:
- Data prep:
  - `setup_concerto_six_imagepoint.sh`
  - `prepare_concerto_imagepoint_splits.py`
  - `verify_concerto_six_datasets.py`
  - `download_hf_dataset_shard.py`
  - `submit_hf_dataset_download_shard_abciq_qc.sh`
  - `launch_structured3d_parallel_download.sh`
- Main-variant diagnostics:
  - `fit_main_variant_enc2d_head.py`
  - `fit_main_variant_coord_mlp_rival.py`
  - `run_main_variant_step05.sh`
  - `submit_main_variant_step05_abciq_qf.sh`

Dataset scope:
- ScanNet, ScanNet++, Structured3D, S3DIS, ARKitScenes, HM3D.
- RE10K is excluded from this mainline because it belongs to the large-video
  variant.

Current data state:
- Ready: ARKitScenes, ScanNet, ScanNet++, S3DIS, HM3D, and Structured3D
  Concerto image-point absmeta roots under repo-local `data/`.
- The final six-dataset Step 0/0.5 job `133050.qjcm` finished with
  `Exit_status=0`, `rt_QF=1`, tag `main-origin-six-step05`.
- Target-corruption distance job `133090.qjcm` finished with `Exit_status=0`.

Acceptance:
- Main-variant causal battery:
  `data/runs/main_variant_enc2d_headfit/main-origin-six-step05/results_main_variant_causal_battery.md`.
- Main-variant coord-MLP rival:
  `data/runs/main_variant_coord_mlp_rival/main-origin-six-step05/results_official_coord_mlp_rival.md`.
- Main-variant target-corruption distance:
  `data/runs/main_variant_target_corruption_distance/main-origin-six-step05/results_target_corruption_distance.md`.
- SR-LoRA Phase A:
  [results_sr_lora_phasea.md](./results_sr_lora_phasea.md) and
  [results_sr_lora_phasea.csv](./results_sr_lora_phasea.csv).
  - Smoke: `133093.qjcm`, `sr-lora-v5-smoke3-r4-d03`.
  - Pilot: `133095.qjcm`, `sr-lora-v5-pilot-r4-d03-i256-qf4`.
  - Matrix: `133097.qjcm` to `133100.qjcm`.
  - Corrected `m=0.1` follow-up: `133102.qjcm`.
  - `m=0.2` pilot/follow-up: `133104.qjcm`, `133105.qjcm`.
  - Failed first follow-up: `133101.qjcm`, base-config/large-checkpoint shape
    mismatch; corrected by large ScanNet proxy config.
  - Decision: coord-only SR-LoRA v5 is training-stable but downstream no-go.

## ARKit Full Causal Branch

Source:
- [results_arkit_full_causal.md](./results_arkit_full_causal.md)

Key numbers:

| experiment | enc2d last | delta vs baseline |
| --- | ---: | ---: |
| baseline | 5.9470 | 0.0000 |
| coord_mlp | 6.4204 | +0.4734 |
| global_target_permutation | 6.4563 | +0.5093 |
| cross_scene_target_swap | 6.4526 | +0.5056 |
| cross_image_target_swap | 6.4873 | +0.5403 |
| coord_residual_target | 7.2936 | +1.3466 |

Interpretation:
- `coord_mlp` remains surprisingly competitive.
- Strong correspondence corruption hurts, but not catastrophically.
- This supports a `correspondence-induced coordinate shortcut` /
  `scene-coordinate cache shortcut` reading.
- The current `coord_residual_target` implementation is not yet a successful fix.

## Corrected Stress Test

Source:
- [results_arkit_full_stress_corrected.md](./results_arkit_full_stress_corrected.md)

Key numbers:

| checkpoint | clean | local_surface_destroy | z_flip | xy_swap | roll_90_x |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.453537 | 3.897785 | 3.980001 | 3.450245 | 3.845464 |
| coord_mlp | 3.536469 | 3.544549 | 3.620394 | 3.537728 | 3.601248 |
| coord_residual_target | 3.628110 | 4.223210 | 4.181930 | 3.624020 | 4.302245 |

Interpretation:
- The original branch is more sensitive to geometry destruction and scene-frame
  transforms.
- `coord_mlp` is much flatter under these perturbations.
- This is consistent with weaker dependence on local geometry.

## ScanNet Downstream Status

Status:
- The original / coord_mlp / no-enc2d / no-enc2d-renorm continuation proxy is
  finished enough for the current decision.
- The safest readout is "downstream effect is real but limited."
- The next gate is not another critique arm; it is whether the projection
  residual fix can match or beat original Concerto on ScanNet linear.

What was confirmed:
- The dataset path and weights are usable.
- A single-GPU safe smoke run reaches actual training:
  - historical path: `exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log`
- The official ScanNet linear gate now completes on the safe single-GPU path:
  - historical path: `exp/concerto/scannet-proxy-official-origin-lin/train.log`
  - final `Val result: mIoU/mAcc/allAcc = 0.1752 / 0.2467 / 0.6167`
  - checkpoints:
    - `exp/concerto/scannet-proxy-official-origin-lin/model/model_best.pth`
    - `exp/concerto/scannet-proxy-official-origin-lin/model/model_last.pth`

Previous failure mode:
- The original 2-GPU official gate crashed repeatedly in the distributed spawn path:
  - historical path: `tools/concerto_projection_shortcut/logs/scannet_gate.launch.log`
- The previous `no-enc2d-renorm` full post-train test aborted while writing
  `.npy` outputs because of local disk pressure. Validation metrics are still
  usable.

Current interpretation:
- The blocker is not the basic dataset path.
- The old blocker was the multi-GPU `mp.spawn` path on this machine.
- For current work, use the validation-only ScanNet linear config to avoid the
  full-test disk failure path.
- `projres_v1b` shows that full coordinate removal was too blunt; partial
  target residualization around `beta=0.75` recovers meaningful downstream
  performance, but still does not beat original.
- `projres_v1c` shows that swapping to lower-capacity / height-biased static
  priors does not close the gap; `mlp_z` is best within v1c but remains below
  the v1b best.
- The long-horizon e025 same-stage check shows v1b can match the original
  downstream proxy much more closely than the 5-epoch gate suggested:
  - same-stage original: 0.5531 / 0.5531 mIoU
  - v1b `combo-b075-a001`: 0.5526 / 0.5526 mIoU
  - this is an effective tie, not a +0.01 fix-and-beat-original result
- Frozen post-training nuisance surgery on the original e025 checkpoint is also
  near-tie:
  - SPLICE-3D `height+xyz`: 0.5509 / 0.5509 mIoU
  - SPLICE-3D `height`: 0.5510 / 0.5510 mIoU
  - HLNS channel-group proxy `height+xyz`: 0.5515 / 0.5515 mIoU
  - Residual Recycling `height+xyz`, `coord9`: 0.5506 / 0.5506 mIoU
  - this is not a win, but it is a cheap post-training path that leaves the
    backbone frozen and does not materially damage the ScanNet proxy.
- The e025 posthoc stress downstream gate is no-go:
  - SPLICE-3D height max stress gain: `+0.0027` on `z_flip`
  - SPLICE-3D height+xyz max stress gain: `+0.0029` on `z_flip`
  - Residual Recycling coord9 max stress gain: `+0.0023` on `z_flip`
  - pass threshold was `clean >= original - 0.005` and any stress
    `>= original + 0.005`; clean passes, stress does not.
- The Step 1 local-geometry smoke is no-go for the current `geom_local9`
  descriptor:
  - offline frozen-cache original baseline: `0.49433` mIoU
  - `concat=[x, phi]`: `0.49435`, delta `+0.00002`
  - `residual_expert_global=W0x+Aphi`: `0.49427`, delta `-0.00006`
  - pass threshold was `+0.003`; this does not support launching UGNR with the
    tested local geometry descriptor.
- On ABCI-Q, keep using the validated `torchrun` / `pbsdsh` path described in
  [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md).

## Active Downstream Jobs

Running now:
- None observed at the time of this update.

Recently completed:
- `132837.qjcm`: Step 1 local-geometry smoke fit-only rerun after robust
  float64 ridge solve, ABCI-Q `rt_QF=1`, completed.
  - result root:
    `data/runs/step1_geometry_smoke/arkit-full-original-long-e025-qf32-continue/geom_smoke`
  - result: `concat` delta `+0.00002`, `residual_expert_global` delta
    `-0.00006`; no-go by the `+0.003` geometry-signal criterion.
- `132835.qjcm`: Step 1 local-geometry main cache extraction, ABCI-Q
  `rt_QF=1`, created the 524288 train / 131072 val row `geom_local9` cache but
  hit a singular float32 normal-equation solve in the fit stage.
- `132832.qjcm` and `132833.qjcm`: Step 1 tiny path smoke and calibration,
  ABCI-Q `rt_QF=1`, both completed.
- `132789.qjcm`: Residual Recycling e025 stress downstream, ABCI-Q `rt_QF=1`,
  completed.
  - result root:
    `data/runs/posthoc_stress_e025pilot_recycle`
  - result: clean `0.5509`, max stress gain `+0.0023` on `z_flip`; no-go by
    the `+0.005` stress criterion.
- `132780.qjcm`: Residual Recycling e025 posthoc linear probe, ABCI-Q
  `rt_QF=1`, completed.
  - result root:
    `data/runs/posthoc_surgery_e025pilot/original-long-e025-qf32/recycle_height_xyz_coord9_g1.0_r1.0`
  - result: `0.5506 / 0.5506` ScanNet proxy mIoU.
- `132775.qjcm`: SPLICE-3D e025 stress downstream, ABCI-Q `rt_QF=1`,
  completed.
  - result root:
    `data/runs/posthoc_stress_e025pilot`
  - result: clean preserved, max stress gains below `+0.005`; no-go for Stage 1.
- `132600.qjcm` and `132602.qjcm`: attempted fresh e050 same-stage
  continuations on ABCI-Q `rt_QF=8`, both failed by walltime.
  - `132600.qjcm`: original e050, `Exit_status=-29`, stopped at epoch 6.
  - `132602.qjcm`: v1b e050, `Exit_status=-29`, stopped at epoch 6.
  - No valid e050 same-stage checkpoint should be read from these runs.
- `132601.qjcm` and `132603.qjcm`: dependent follow-ups for the failed e050
  attempts did not produce valid e050 follow-up results.
- `132608.qjcm`: frozen post-training nuisance surgery e025 pilot, ABCI-Q
  `rt_QF=1`, completed.
  - result root:
    `data/runs/posthoc_surgery_e025pilot/original-long-e025-qf32`
  - results:
    SPLICE-3D `height+xyz` 0.5509 / 0.5509, SPLICE-3D `height` 0.5510 /
    0.5510, HLNS `height+xyz` 0.5515 / 0.5515 ScanNet proxy mIoU.
- `132455.qjcm` and `132457.qjcm`: long-horizon e025 continuations on ABCI-Q
  `rt_QF=8`, both `Exit_status=0`.
  - `132455.qjcm`: original reference, walltime `03:11:21`
  - `132457.qjcm`: v1b `combo-b075-a001`, walltime `03:11:46`
  - requested walltime was `03:50:00`, so the setting was sufficient without
    excessive slack.
  - checkpoints:
    `exp/concerto/arkit-full-original-long-e025-qf32-continue/model/model_last.pth`
    and
    `exp/concerto/arkit-full-projres-v1b-combo-b075-a001-long-e025-qf32-continue/model/model_last.pth`
- `132456.qjcm` and `132458.qjcm`: dependent ScanNet proxy follow-ups on
  ABCI-Q `rt_QF=1`, both `Exit_status=0`.
  - walltimes: `00:50:15` and `00:49:57`
  - requested walltime was `01:05:00`
  - results:
    original e025 `0.5531 / 0.5531`, v1b e025 `0.5526 / 0.5526`
  - summary root:
    `data/runs/projres_long/summaries/long-e025-qf32`
- `132277.qjcm`: ProjRes v1c z-prior fit on ABCI-Q `rt_QF=1`,
  `Exit_status=0`.
  - cache reused:
    `data/runs/projres_v1/priors/cache`
  - fitted priors:
    `linear_z`, `mlp_z`
  - selected by cosine loss:
    `mlp_z`
- `132278.qjcm` to `132283.qjcm`: ProjRes v1c 6-arm prior-family smoke matrix
  on ABCI-Q `rt_QF=1`.
  - summary root:
    `data/runs/projres_v1c/summaries/h10016-qf1-v1c-prior256`
  - logs reached 190 to 193 steps before the 35 minute walltime; partial smoke
    summaries were generated with a 128-step minimum.
  - selected top arms:
    `linz-b075-a000`, `mlpz-b075-a001`, `linxyz-b075-a001`
- `132284.qjcm` to `132286.qjcm`: ProjRes v1c 5-epoch continuations, each on
  ABCI-Q `rt_QF=4` (4 nodes / 16 H100 GPUs), all `Exit_status=0`.
  - concurrent allocation: 12 nodes / 48 H100 GPUs
  - walltimes: about 47 minutes
  - checkpoints:
    `exp/concerto/arkit-full-projres-v1c-linz-b075-a000-h10016x3-qf16-v1c-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1c-mlpz-b075-a001-h10016x3-qf16-v1c-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1c-linxyz-b075-a001-h10016x3-qf16-v1c-continue/model/model_last.pth`
- `132287.qjcm` to `132289.qjcm`: ProjRes v1c follow-up stress + ScanNet
  linear gates on ABCI-Q `rt_QF=1`, all `Exit_status=0`.
  - summary root:
    `data/runs/projres_v1c/summaries/h10016x3-qf16-v1c`
  - result: no strong-go for all three arms
- `132208.qjcm`: ProjRes v1b metric sanity on ABCI-Q `rt_QF=1`,
  `Exit_status=0`.
  - setting: v1a-equivalent `beta=1.0`, `alpha=0.05`, 16 train steps
  - result: `coord_projection_loss_check=0.0`
- `132209.qjcm` to `132219.qjcm`: ProjRes v1b 11-arm smoke matrix on ABCI-Q
  `rt_QF=1`, all `Exit_status=0`.
  - summary root:
    `data/runs/projres_v1b/summaries/h10016-qf1-v1b-pre256`
  - selected top arms:
    `combo-b075-a001`, `penalty-b000-a002`, `resonly-b075-a000`,
    `combo-b050-a002`
- `132220.qjcm` to `132223.qjcm`: ProjRes v1b 5-epoch continuations, each on
  ABCI-Q `rt_QF=4` (4 nodes / 16 H100 GPUs), all `Exit_status=0`.
  - concurrent allocation: 16 nodes / 64 H100 GPUs
  - walltimes: about 47 minutes
  - checkpoints:
    `exp/concerto/arkit-full-projres-v1b-combo-b075-a001-h10016x4-qf16-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1b-penalty-b000-a002-h10016x4-qf16-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1b-resonly-b075-a000-h10016x4-qf16-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1b-combo-b050-a002-h10016x4-qf16-continue/model/model_last.pth`
- `132255.qjcm` to `132258.qjcm`: ProjRes v1b follow-up stress + ScanNet
  linear gates on ABCI-Q `rt_QF=1`, all `Exit_status=0`.
  - summary root:
    `data/runs/projres_v1b/summaries/h10016x4-qf16`
  - result: no strong-go for all four arms
- `132196.qjcm`: ProjRes v1 5-epoch continuation on ABCI-Q `rt_QF=8`
  (8 nodes / 32 H100 GPUs), `Exit_status=0`, walltime `00:39:37`.
  - experiment:
    `arkit-full-projres-v1a-alpha005-h10032-qf32-continue`
  - checkpoint:
    `exp/concerto/arkit-full-projres-v1a-alpha005-h10032-qf32-continue/model/model_last.pth`
  - main log:
    `data/logs/abciq/projres_v1_continue_qf16_132196.qjcm.log`
  - rank logs:
    `data/runs/projres_v1/logs/multinode/132196.qjcm_arkit-full-projres-v1a-alpha005-h10032-qf32-continue_20260415_233258/logs/`
- `132198.qjcm`: ProjRes v1 follow-up on ABCI-Q `rt_QF=1`,
  `Exit_status=0`, walltime `00:50:06`.
  - script:
    `tools/concerto_projection_shortcut/submit_projres_v1_followup_abciq_qf.sh`
  - log:
    `data/logs/abciq/projres_v1_followup_132198.qjcm.log`
  - outputs:
    `data/runs/projres_v1/summaries/h10032-qf32/arkit-full-projres-v1a-alpha005-h10032-qf32-continue_stress.csv`
    and
    `data/runs/projres_v1/summaries/h10032-qf32/scannet-proxy-projres-v1a-alpha005-h10032-qf32-lin_gate.json`

ProjRes v1 gate result:
- selected alpha: `0.05`
- final continuation metrics:
  - `loss=8.0899`, `enc2d_loss=8.0655`,
    `coord_residual_enc2d_loss=6.3309`,
    `coord_alignment_loss=0.0200`,
    `coord_pred_energy=0.0020`, `coord_residual_norm=0.7308`
- ARKit stress, enc2d loss mean over 20 batches:
  - clean `8.022868`
  - local surface destroy `9.115467`
  - z flip `8.941781`
  - xy swap `8.022733`
  - roll 90 x `9.353544`
- ScanNet linear gate:
  - ProjRes v1a last/best mIoU: `0.3627` / `0.3627`
  - original continuation last/best mIoU: `0.4794` / `0.4552`
  - no-enc2d-renorm last/best mIoU: `0.3794` / `0.3802`
  - deltas vs original: `-0.1167` last, `-0.0925` best
  - deltas vs no-enc2d-renorm: `-0.0167` last, `-0.0175` best
  - decision: `strong_go=false`, `linear_gate_not_strong_go`
- Summary:
  - [results_projres_v1.md](./results_projres_v1.md)

ProjRes v1b gate result:
- best arm: `combo-b075-a001`
- beta / alpha: `0.75` / `0.01`
- final continuation metrics:
  - `loss=7.8700`, `enc2d_loss=7.6640`,
    `coord_residual_enc2d_loss=7.2912`,
    `coord_alignment_loss=1.7203`,
    `coord_removed_energy=0.0734`,
    `coord_pred_energy=0.1720`, `coord_residual_norm=0.8921`,
    `coord_projection_loss_check=0.0000`
- ARKit stress, enc2d loss mean over 20 batches:
  - clean `7.649344`
  - local surface destroy `8.862813`
  - z flip `8.860726`
  - xy swap `7.673076`
  - roll 90 x `9.093753`
- ScanNet linear gate:
  - best v1b last/best mIoU: `0.4220` / `0.4220`
  - original continuation last/best mIoU: `0.4794` / `0.4552`
  - no-enc2d-renorm last/best mIoU: `0.3794` / `0.3802`
  - deltas vs original: `-0.0574` last, `-0.0332` best
  - deltas vs no-enc2d-renorm: `+0.0426` last, `+0.0418` best
  - decision: `strong_go=false`, `linear_gate_not_strong_go`
- Summary:
  - [results_projres_v1.md](./results_projres_v1.md)

ProjRes v1c gate result:
- hypothesis:
  - keep `beta=0.75` and test lower-capacity / height-biased priors instead of
    widening the beta/alpha grid.
- fitted z-priors:
  - `linear_z`: cosine loss `0.735445`, target energy `0.080087`, residual norm
    `0.958630`
  - `mlp_z`: cosine loss `0.643186`, target energy `0.136156`, residual norm
    `0.928692`
- continued arms:
  - `linz-b075-a000`: `linear_z`, `beta=0.75`, `alpha=0.00`
  - `mlpz-b075-a001`: `mlp_z`, `beta=0.75`, `alpha=0.01`
  - `linxyz-b075-a001`: `linear_xyz`, `beta=0.75`, `alpha=0.01`
- best v1c arm:
  - `mlpz-b075-a001`
- best v1c final continuation metrics:
  - `loss=7.8765`, `enc2d_loss=7.6774`,
    `coord_residual_enc2d_loss=7.2917`,
    `coord_alignment_loss=1.6903`,
    `coord_removed_energy=0.0736`,
    `coord_pred_energy=0.1690`, `coord_residual_norm=0.8903`,
    `coord_projection_loss_check=0.0000`
- best v1c ScanNet linear gate:
  - last/best mIoU: `0.4186` / `0.4186`
  - deltas vs original: `-0.0608` last, `-0.0366` best
  - deltas vs no-enc2d-renorm: `+0.0392` last, `+0.0384` best
  - decision: `strong_go=false`, `linear_gate_not_strong_go`
- Summary:
  - [results_projres_v1.md](./results_projres_v1.md)

Latest ABCI-Q launcher status, 2026-04-16 JST:
- The useful hint from
  `/groups/qgah50055/ide/3d-sans-3dscans/Pointcept/configs/*.sh` is that
  ABCI-Q Pointcept jobs use `python -m torch.distributed.run` with
  `tools/ddp_train.py`, not the original Pointcept `mp.spawn` launcher.
- This checkout now has the same launcher option:
  - [tools/ddp_train.py](../../tools/ddp_train.py)
  - [scripts/train.sh](../../scripts/train.sh) with
    `POINTCEPT_TRAIN_LAUNCHER=torchrun`
- A lightweight batch-only diagnostic was added for ARKit DDP checks:
  - [debug_arkit_ddp_batches.py](./debug_arkit_ddp_batches.py)
  - [submit_debug_arkit_ddp_batches_abciq_qf.sh](./submit_debug_arkit_ddp_batches_abciq_qf.sh)
- Batch-only diagnostic job `132175.qjcm` completed successfully:
  - walltime: `00:01:12`
  - result: `Exit_status = 0`
  - log: `data/logs/abciq/debug_arkit_ddp_batches_132175.qjcm.log`
  - result: all ranks completed 16 batches through DataLoader, CUDA copy, and
    all-reduce. The current stall is therefore not reproduced by DataLoader
    alone.
- `POINTCEPT_TRACE_STEPS=1` now prints per-rank full-training step trace around
  `next(DataLoader)`, `run_step`, `forward`, `backward`, and `after_step`.
- Full-training trace results:
  - `132176.qjcm`: 8-step `alpha=0.05` run completed with `Exit_status = 0`
    in `00:02:09`.
  - `132177.qjcm`: 16-step run was stopped at `00:03:48`; all ranks fetched
    the 9th batch (`iter=8`) and reached `before_run_step`, but no rank reached
    `after_run_step`.
  - `132178.qjcm`: run-step trace was stopped at `00:02:54`; all ranks reached
    `run_step_before_forward` on the first iteration, rank 2 reached
    `run_step_after_forward` / `run_step_before_backward`, and ranks 0/1/3 did
    not return from forward. This established the pre-fix failure as a
    model-forward rank divergence/hang, not a pure DataLoader issue.
- On ABCI-Q H100, the current stable single-node setting is:
  - `POINTCEPT_TRAIN_LAUNCHER=torchrun`
  - `NCCL_STABLE_MODE=1`
  - `NCCL_P2P_DISABLE=1`
  - `NCCL_NET_GDR_LEVEL=0`
  - `CONCERTO_NUM_WORKER=1`
- A 4-GPU torchrun smoke with this stable NCCL mode completed:
  - job: `132168.qjcm`
  - walltime: `00:01:44`
  - result: `Exit_status = 0`
  - log: `data/logs/abciq/projres_v1_smoke_qf1_132168.qjcm.log`
  - output:
    `data/runs/projres_v1/summaries/h10016-qf4gtorchstable/selected_smoke.json`
- The earlier longer-smoke stalls were isolated to the distributed metric
  reduction at the end of `Concerto.forward`, not to DataLoader, optimizer, or
  H100 memory pressure:
  - pre-fix stopped/stalled jobs: `132169`, `132170`, `132172`, `132173`,
    `132184`, `132185`, `132187`, `132189`
  - symptom: one rank returned from forward while other ranks were still inside
    forward-side collectives.
  - fix: use a fixed distributed result key order, fill missing coord metric
    scalars with zero, reduce detached metric copies, and keep
    `loss_for_backward` separate from reduced logging metrics.
  - this also removes the PyTorch `c10d::allreduce_` autograd warning.
- Post-fix 4-GPU H100 smoke results:
  - `132190.qjcm`: 16 steps, flash on, `Exit_status = 0`, walltime `00:02:27`.
  - `132191.qjcm`: 64 steps, flash on, `Exit_status = 0`, walltime `00:04:57`.
  - `132192.qjcm`: 16 steps after detached metric reduction, `Exit_status = 0`,
    walltime `00:02:24`, no `c10d::allreduce_` autograd warning.
- Multi-node continuation validation:
  - `132194.qjcm`: 4 nodes / 16 H100 GPUs, short continuation validation,
    `Exit_status = 0`, walltime `00:03:58`.
  - `132195.qjcm`: 8 nodes / 32 H100 GPUs, 1 epoch x 16-step continuation
    validation, `Exit_status = 0`, walltime `00:02:08`.
  - The H100 continuation config now accepts `CONCERTO_EPOCH` for bounded
    validation jobs, and the `pbsdsh` launcher explicitly forwards
    `CONCERTO_*` env values to every node.
- Current smoke-only selected alpha artifact:
  - `data/runs/projres_v1/summaries/h10016-qf1fixed64/selected_smoke.json`
  - selected `alpha=0.05`
  - This is a 64-step single-node smoke artifact. It has now been validated by
    short 4-node and 8-node continuation runs.

Completed setup jobs:
- ABCI-Q env setup job `132080.qjcm` completed with `Exit_status = 0`.
  - log: `data/logs/abciq/env_setup_132080.qjcm.log`
  - result: created `data/venv/pointcept-concerto-py311-cu124` and validated
    `torch`, `torch_scatter`, `spconv`, `pointops`, `transformers`, and
    `pointcept` imports on an `rt_QF=1` GPU allocation.
- ABCI-Q dry-run job `132093.qjcm` completed successfully.
  - log: `data/logs/abciq/projres_v1_132093.qjcm.log`
  - result: GPU preflight passed and `DRY_RUN=1` printed only repo-local
    `data/...` paths.

Prepared data:
- `data/scannet` is a symlink to
  `/groups/qgah50055/ide/3d-sans-3dscans/scannet`.
- ARKit compressed snapshot exists under `data/concerto_arkitscenes_compressed`.
- ARKit extracted data exists under `data/arkitscenes`.
- ARKit absolute metadata exists under `data/arkitscenes_absmeta`.
- DINOv2 cache exists under `data/hf-home`.
- Concerto official weights exist under `data/weights/concerto`.

Stopped job:
- The local `projres_v1` chain was stopped during Stage 1 prior cache
  extraction.
- Log:
  - `/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1/logs/run_projres_v1_chain_20260415_131727_setsid.log`
- It produced logs only. No selected prior, smoke checkpoint, continuation
  checkpoint, stress csv, or ScanNet linear result was produced.

Expected next stage:
1. Finish the active e050 same-stage check for original and
   `projres_v1b combo-b075-a001`.
2. Do not launch the optional fine-tune from either arm under the current gate.
3. Do not automatically extend to e075/e100 until the e050 ScanNet follow-up is
   read out. If e050 is still a near tie or better with improved shortcut
   metrics, then stage the next checkpoint using the fixed target-epoch /
   stop-epoch resume path.
4. Keep the posthoc nuisance-surgery line as the cheap fallback while the
   expensive continuation jobs run. The e025 pilot shows frozen post-training
   stays within `0.0022` mIoU of the original e025 reference.
5. Keep the fixed DDP metric reduction and ABCI-Q `torchrun` path; those
   infrastructure changes are validated.

## Useful Logs And Artifacts

- ARKit causal summary:
  - [results_arkit_full_causal.csv](./results_arkit_full_causal.csv)
  - [results_arkit_full_causal.md](./results_arkit_full_causal.md)
- ARKit stress summary:
  - [results_arkit_full_stress_corrected.csv](./results_arkit_full_stress_corrected.csv)
  - [results_arkit_full_stress_corrected.md](./results_arkit_full_stress_corrected.md)
- Interim write-up:
  - [results_interim_summary_2026-04-06.md](./results_interim_summary_2026-04-06.md)
- ScanNet gate crash log:
  - historical path: `tools/concerto_projection_shortcut/logs/scannet_gate.launch.log`
- ScanNet gate success log:
  - historical path: `exp/concerto/scannet-proxy-official-origin-lin/train.log`
- ProjRes v1 result:
  - [results_projres_v1.md](./results_projres_v1.md)
- ScanNet gate result note:
  - [results_scannet_gate_2026-04-09.md](./results_scannet_gate_2026-04-09.md)
- ScanNet safe smoke log:
  - historical path: `exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log`
- Pipeline status:
  - [scannet_pipeline_status.md](./scannet_pipeline_status.md)
- Projection residual handoff:
  - [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md)

## Immediate Next Step

1. Continue monitoring the active e050 fresh same-stage jobs:
   - original: `132600.qjcm`
   - v1b `combo-b075-a001`: `132602.qjcm`
   - held follow-ups: `132601.qjcm`, `132603.qjcm`
2. Read out the dependent ScanNet follow-ups before deciding whether to stage
   e075/e100 or pivot to the posthoc fallback.
3. Use the exact-resume launcher only for future staged runs that kept the same
   target scheduler from the first stage; old target-e025 checkpoints are not
   exact e050 resumes.
4. Keep monitoring through ABCI-compatible `qstat` when jobs are active:
   - `qstat | awk -v u="$USER" 'NR==1 || NR==2 || $0 ~ u {print}'`
5. Keep the current completed artifacts:
   - `data/runs/projres_v1/summaries/h10032-qf32`
   - `data/runs/projres_v1b/summaries/h10016-qf1-v1b-pre256`
   - `data/runs/projres_v1b/summaries/h10016x4-qf16`
   - `data/runs/projres_v1c/summaries/h10016x3-qf16-v1c`
   - `data/runs/projres_long/summaries/long-e025-qf32`
   - `data/runs/posthoc_surgery_e025pilot`
