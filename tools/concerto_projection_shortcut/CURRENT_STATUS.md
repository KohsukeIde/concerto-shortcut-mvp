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
  - Data and run outputs should live under repo-local `data/`.
  - Existing ScanNet is used through a symlink, not copied.
  - Do not run the optional fine-tune for v1a/v1b/v1c without a new hypothesis.

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
3. ScanNet continuation proxy:
   - [results_scannet_proxy_lin.md](./results_scannet_proxy_lin.md)
4. ProjRes v1 gate:
   - [results_projres_v1.md](./results_projres_v1.md)
5. Coordinate projection residual handoff:
   - [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md)
6. Short narrative summary:
   - [results_interim_summary_2026-04-06.md](./results_interim_summary_2026-04-06.md)
7. Reproduction / runner overview:
   - [README.md](./README.md)

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
- On ABCI-Q, keep using the validated `torchrun` / `pbsdsh` path described in
  [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md).

## Active Downstream Jobs

Running now:
- No `projres_v1` / `projres_v1b` / `projres_v1c` ABCI-Q job is currently
  running.

Recently completed:
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
1. Treat `projres_v1a` and `projres_v1b` as complete for the current gate.
2. Do not launch the optional fine-tune from either arm under the current gate.
3. Use the v1b result as the next design constraint: partial target
   residualization helps, but still removes or distorts useful signal enough to
   stay below original continuation.
4. Design a new arm before spending more ABCI-Q points. The most plausible
   direction is not stronger removal, but a more selective objective around the
   useful `beta=0.75` region.
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

1. Decide the next method variant from the v1b result. Current evidence says
   "partial removal helps, full removal hurts, original still wins."
2. Keep monitoring through ABCI-compatible `qstat` when jobs are active:
   - `qstat | awk -v u="$USER" 'NR==1 || NR==2 || $0 ~ u {print}'`
3. Keep the current completed artifacts:
   - `data/runs/projres_v1/summaries/h10032-qf32`
   - `data/runs/projres_v1b/summaries/h10016-qf1-v1b-pre256`
   - `data/runs/projres_v1b/summaries/h10016x4-qf16`
