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
  - Move from critique to a minimal fix: coordinate projection residualization.
  - Prepare and run the `projres_v1` chain on ABCI-Q with `venv`, not conda.
  - Data and run outputs should live under repo-local `data/`.
  - Existing ScanNet is used through a symlink, not copied.

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
4. Coordinate projection residual handoff:
   - [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md)
5. Short narrative summary:
   - [results_interim_summary_2026-04-06.md](./results_interim_summary_2026-04-06.md)
6. Reproduction / runner overview:
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
- On ABCI-Q, run the `projres_v1` fix chain described in
  [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md).

## Active Downstream Jobs

Running now:
- No `projres_v1` job is currently running.

Latest ABCI-Q launcher status, 2026-04-15 JST:
- The useful hint from
  `/groups/qgah50055/ide/3d-sans-3dscans/Pointcept/configs/*.sh` is that
  ABCI-Q Pointcept jobs use `python -m torch.distributed.run` with
  `tools/ddp_train.py`, not the original Pointcept `mp.spawn` launcher.
- This checkout now has the same launcher option:
  - [tools/ddp_train.py](../../tools/ddp_train.py)
  - [scripts/train.sh](../../scripts/train.sh) with
    `POINTCEPT_TRAIN_LAUNCHER=torchrun`
- A lightweight batch-only diagnostic was added for the current blocker:
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
    not return from forward. Treat the current blocker as a model-forward rank
    divergence/hang, not a pure DataLoader issue.
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
- Longer ARKit smoke is still unstable and must be fixed before launching full
  continuation:
  - `132169.qjcm`: 64-step attempt stalled after partial progress.
  - `132170.qjcm`: 64-step attempt stalled before the first train batch.
  - `132172.qjcm`: 64-step attempt stalled after 8 train steps.
  - `132173.qjcm`: 8-step two-alpha attempt completed `alpha=0.05`, then
    `alpha=0.10` stalled before the first train batch and the job was stopped
    at short walltime.
- Current smoke-only selected alpha artifact:
  - `data/runs/projres_v1/summaries/h10016-qf1/selected_smoke.json`
  - selected `alpha=0.05`
  - This is only an 8-step smoke artifact; it is not a go signal for the
    5-epoch continuation.
- Do not launch the 4-node / 16-GPU continuation yet. First isolate the ARKit
  model-forward rank divergence on single-node 4-GPU.

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
1. Treat the prior stage as complete and resume from
   `data/runs/projres_v1/priors/selected_prior.json`.
2. Diagnose the model-forward rank divergence inside the Concerto forward path.
3. Keep `torchrun` and stable NCCL mode enabled for ABCI-Q jobs.
4. Run the 5-epoch continuation only after a longer 4-GPU smoke is repeatable.
5. Report the selected alpha, stress result, and ScanNet linear gate result.

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
- ScanNet gate result note:
  - [results_scannet_gate_2026-04-09.md](./results_scannet_gate_2026-04-09.md)
- ScanNet safe smoke log:
  - historical path: `exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log`
- Pipeline status:
  - [scannet_pipeline_status.md](./scannet_pipeline_status.md)
- Projection residual handoff:
  - [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md)

## Immediate Next Step

1. Add a narrower forward trace around Concerto student/teacher forward,
   DINO/image branches, and projection residual loss.
2. Keep monitoring through ABCI-compatible `qstat`:
   - `qstat | awk -v u="$USER" 'NR==1 || NR==2 || $0 ~ u {print}'`
3. Watch for:
   - `data/runs/projres_v1/priors/selected_prior.json`
   - `data/runs/projres_v1/summaries/selected_smoke.json`
   - stress CSV and ScanNet linear gate JSON under `data/runs/projres_v1/summaries`
