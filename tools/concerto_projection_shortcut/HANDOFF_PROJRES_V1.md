# ProjRes v1 ABCI-Q Handoff

Updated: 2026-04-15 JST

This document is the handoff entry point for running the coordinate projection
residual fix experiment on ABCI-Q.

## Current Decision

- Do not continue the old `/mnt/urashima`/conda run.
- Run the next gate on ABCI-Q with a repo-local `venv`.
- Use `qsub` and `rt_QF=1` for GPU work. On this system, `rt_QF=1` allocates
  4 GPUs.
- Keep data, caches, weights, and run mirrors under
  `/groups/qgah50055/ide/concerto-shortcut-mvp/data`.
- Use existing ScanNet through a symlink instead of copying it.
- The old `projres_v1` chain was stopped manually during Stage 1 before any
  prior cache or checkpoint was written.
- Do not launch the 4-node / 16-GPU continuation yet. A 4-GPU ABCI-Q smoke now
  starts correctly with `torchrun`, but longer ARKit smoke still stalls after a
  small number of batches.
- The next scientific gate is not another critique arm. It is whether the
  proposed fix can match or beat original Concerto in the ScanNet linear gate.

## Scientific State

Strong claim already supported:
- The released Concerto `enc2d` objective has a strong scene-coordinate
  shortcut on ARKitScenes.

Safe downstream interpretation:
- ScanNet continuation proxy shows the shortcut has measurable downstream
  relevance.
- It does not support the stronger claim that most of Concerto downstream gain
  is explained by the coordinate shortcut.

Relevant finished results:
- [results_interim_summary_2026-04-06.md](./results_interim_summary_2026-04-06.md)
- [results_arkit_full_causal.md](./results_arkit_full_causal.md)
- [results_arkit_full_stress_corrected.md](./results_arkit_full_stress_corrected.md)
- [results_scannet_proxy_lin.md](./results_scannet_proxy_lin.md)

Key ScanNet proxy numbers:

| arm | last val mIoU | best val mIoU | status |
| --- | ---: | ---: | --- |
| original continuation | 0.4794 | 0.4552 | finished |
| coord_mlp continuation | 0.4064 | 0.3829 | finished |
| no-enc2d continuation | 0.4010 | 0.3765 | finished |
| no-enc2d-renorm continuation | 0.3794 | 0.3802 | validation finished, full test aborted due disk |

Readout:
- `coord_mlp` is only slightly above `no-enc2d`.
- Objective-level shortcut remains strong.
- Downstream "mostly coordinate shortcut" is not supported by this continuation
  proxy.
- Next paper route is fix-and-beat-original.

## Implemented Fix

Mode added:
- `shortcut_probe.mode = "coord_projection_residual"`

Implementation:
- [pointcept/models/concerto/concerto_v1m1_base.py](../../pointcept/models/concerto/concerto_v1m1_base.py)

Configs:
- [configs/concerto/pretrain-concerto-v1m1-0-arkit-full-projres-v1a-continue.py](../../configs/concerto/pretrain-concerto-v1m1-0-arkit-full-projres-v1a-continue.py)
- [configs/concerto/pretrain-concerto-v1m1-0-arkit-full-projres-v1a-smoke.py](../../configs/concerto/pretrain-concerto-v1m1-0-arkit-full-projres-v1a-smoke.py)
- [configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly.py](../../configs/concerto/semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly.py)

Prior fitting:
- [fit_coord_prior.py](./fit_coord_prior.py)

Main chain:
- [run_projres_v1_chain.sh](./run_projres_v1_chain.sh)

Loss definition:

```text
u = normalize(stopgrad(g(c)))
t_res = t0 - dot(t0, u) * u
loss = 1 - cos(y0, t_res) + alpha * cos(y0, u)^2
```

Logged shortcut metrics:
- `coord_residual_enc2d_loss`
- `coord_alignment_loss`
- `coord_target_energy`
- `coord_pred_energy`
- `coord_residual_norm`

Interpretation of metrics:
- `coord_target_energy`: how much of the target lies along the learned coordinate
  direction.
- `coord_pred_energy`: how much the prediction is using that direction.
- `coord_residual_norm`: guard against deleting too much target signal.
- `coord_alignment_loss`: penalty term for prediction alignment to `u`.

## Chain Stages

The chain is auto-stop gated.

1. Preflight import, batch, and forward.
2. Fit `linear_xyz_prior` and `mlp_xyz_prior` on ARKit train/val caches.
3. Select MLP only if validation cosine loss improves by at least `0.02`.
4. Run 1-epoch smoke for `alpha=0.05` and `alpha=0.10`.
5. Continue only the best passing smoke arm for 5 epochs.
6. Run corrected ARKit stress on the selected checkpoint.
7. Run ScanNet linear validation-only gate.
8. Run ScanNet fine-tune only if the linear gate is strong.

Smoke pass conditions:
- finite loss
- `coord_pred_energy` decreases
- `coord_residual_norm >= 0.70`
- no `enc2d_loss` collapse or explosion

Linear gate conditions:
- strong go: fix beats original by at least `+0.01` mIoU on last or best val
- weak go: fix is comparable and shortcut metrics clearly improve
- no-go: fix is below original and near `no-enc2d-renorm`

## Current Run Status

ABCI-Q status, 2026-04-15 JST:
- No `projres_v1` job is currently running.
- ABCI-Q `rt_QF=1` exposes 4 H100 80GB GPUs in this checkout's jobs.
- The prior stage is complete and should be reused:
  - selected prior: `mlp`
  - selected prior summary:
    `data/runs/projres_v1/priors/selected_prior.json`
  - selected prior checkpoint:
    `data/runs/projres_v1/priors/mlp/model_last.pth`
- The useful ABCI-Q launcher pattern came from
  `/groups/qgah50055/ide/3d-sans-3dscans/Pointcept/configs/*.sh`:
  `python -m torch.distributed.run --nproc_per_node=... tools/ddp_train.py`.
- This checkout now supports that path through:
  - [tools/ddp_train.py](../../tools/ddp_train.py)
  - [scripts/train.sh](../../scripts/train.sh)
  - `POINTCEPT_TRAIN_LAUNCHER=torchrun`
- A batch-only diagnostic is available for the current stall:
  - [debug_arkit_ddp_batches.py](./debug_arkit_ddp_batches.py)
  - [submit_debug_arkit_ddp_batches_abciq_qf.sh](./submit_debug_arkit_ddp_batches_abciq_qf.sh)
- Batch-only diagnostic `132175.qjcm` completed 16 batches on all ranks through
  DataLoader, CUDA copy, and all-reduce in `00:01:12` with `Exit_status = 0`.
  The remaining stall is not reproduced by DataLoader alone.
- `POINTCEPT_TRACE_STEPS=1` is available for the next full-training diagnostic.
  It prints per-rank trace lines around batch fetch, CUDA copy, forward,
  backward, `run_step`, and `after_step`.
- Full-training diagnostics:
  - `132176.qjcm`: 8-step `alpha=0.05` run completed with `Exit_status = 0`
    in `00:02:09`.
  - `132177.qjcm`: 16-step run was stopped at `00:03:48`; all ranks fetched
    the 9th batch (`iter=8`) and reached `before_run_step`, but no rank reached
    `after_run_step`.
  - `132178.qjcm`: run-step trace was stopped at `00:02:54`; all ranks reached
    `run_step_before_forward` on the first iteration, rank 2 reached
    `run_step_after_forward` / `run_step_before_backward`, and ranks 0/1/3 did
    not return from forward.
- Current blocker: model-forward rank divergence/hang. It is not reproduced by
  DataLoader alone, and it is not an optimizer-step issue.
- The original Pointcept `mp.spawn` path should not be used for ABCI-Q
  multi-GPU debugging. On this machine it repeatedly stalled.
- Stable single-node 4-GPU smoke requires the current NCCL guard rails:
  - `NCCL_STABLE_MODE=1`
  - `NCCL_P2P_DISABLE=1`
  - `NCCL_NET_GDR_LEVEL=0`
  - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
  - `CONCERTO_NUM_WORKER=1`
- Confirmed short success:
  - job: `132168.qjcm`
  - walltime: `00:01:44`
  - result: `Exit_status = 0`
  - log: `data/logs/abciq/projres_v1_smoke_qf1_132168.qjcm.log`
  - output:
    `data/runs/projres_v1/summaries/h10016-qf4gtorchstable/selected_smoke.json`
- Current partial smoke artifact:
  - job: `132173.qjcm`
  - walltime: `00:08:39`
  - result: stopped/terminated at short walltime while `alpha=0.10` was stuck
  - selected artifact:
    `data/runs/projres_v1/summaries/h10016-qf1/selected_smoke.json`
  - selected `alpha=0.05` based on an 8-step smoke only
- Failed/stopped longer smoke attempts:
  - `132169.qjcm`: 64-step attempt stalled after partial progress.
  - `132170.qjcm`: 64-step attempt stalled before the first train batch.
  - `132172.qjcm`: 64-step attempt stalled after 8 train steps.
- Unfinished stages: reliable longer smoke, 5-epoch continuation, ARKit stress,
  ScanNet linear gate, and optional fine-tune.

Completed setup validation:
- env setup job `132080.qjcm` completed with `Exit_status = 0`
- dry-run job `132093.qjcm` completed successfully with `RUN_FAST_GATE=0`

Stopped run:
- log: `/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1/logs/run_projres_v1_chain_20260415_131727_setsid.log`
- stage reached: prior fit cache extraction
- output written: logs only
- no useful prior cache, selected prior, smoke checkpoint, continuation
  checkpoint, stress csv, or linear result was produced

Reason for stopping:
- current prior cache extraction is too slow on this server
- the replacement path is ABCI-Q single-node first, then multi-node only if the
  single-node fast gate behaves correctly

## ABCI-Q Data And Environment

Default layout:

```text
/groups/qgah50055/ide/concerto-shortcut-mvp/data
```

Shared defaults are in [device_defaults.sh](./device_defaults.sh):

- `POINTCEPT_DATA_ROOT=${REPO_ROOT}/data`
- `VENV_DIR=${POINTCEPT_DATA_ROOT}/venv/pointcept-concerto-py311-cu124`
- `ARKIT_COMPRESSED_DIR=${POINTCEPT_DATA_ROOT}/concerto_arkitscenes_compressed`
- `ARKIT_FULL_SOURCE_ROOT=${POINTCEPT_DATA_ROOT}/arkitscenes`
- `ARKIT_FULL_META_ROOT=${POINTCEPT_DATA_ROOT}/arkitscenes_absmeta`
- `SCANNET_EXTRACT_DIR=${POINTCEPT_DATA_ROOT}/scannet`
- `WEIGHT_DIR=${POINTCEPT_DATA_ROOT}/weights/concerto`
- `EXP_MIRROR_ROOT=${POINTCEPT_DATA_ROOT}/runs/projres_v1`
- `HF_HOME=${POINTCEPT_DATA_ROOT}/hf-home`
- `TORCH_HOME=${POINTCEPT_DATA_ROOT}/torch-home`

ABCI modules:

```bash
module load python/3.11/3.11.14
module load cuda/12.6/12.6.2
```

Minimum required inputs:
- ARKit compressed snapshot:
  `data/concerto_arkitscenes_compressed`
- ARKit extracted root:
  `data/arkitscenes`
- ARKit absolute metadata:
  `data/arkitscenes_absmeta`
- ScanNet processed root:
  `data/scannet -> /groups/qgah50055/ide/3d-sans-3dscans/scannet`
- official Concerto weight:
  `data/weights/concerto/concerto_base_origin.pth`
- DINOv2 cache:
  `data/hf-home`, especially `facebook/dinov2-with-registers-giant`
- venv:
  `data/venv/pointcept-concerto-py311-cu124`

Current prepared state on this checkout:
- ScanNet symlink exists and resolves to the existing shared ScanNet data.
- `concerto_base.pth` and `concerto_base_origin.pth` exist under
  `data/weights/concerto`.
- ARKit compressed data exists under `data/concerto_arkitscenes_compressed`.
- ARKit extracted data exists under `data/arkitscenes`.
- ARKit absolute metadata exists under `data/arkitscenes_absmeta`.
- DINOv2 model cache exists under `data/hf-home`.
- ABCI-Q venv exists under `data/venv/pointcept-concerto-py311-cu124`.

Create and validate the venv on a GPU node:

```bash
qsub tools/concerto_projection_shortcut/submit_abciq_env_setup.sh
```

Prepare data assets:

```bash
bash tools/concerto_projection_shortcut/setup_abciq_assets.sh
```

This downloads:
- `Pointcept/concerto_arkitscenes_compressed` under
  `data/concerto_arkitscenes_compressed`
- extracted ARKit under `data/arkitscenes`
- absolute metadata under `data/arkitscenes_absmeta`
- DINOv2 model cache under `data/hf-home`

It does not download `Pointcept/concerto_scannet_compressed` by default.

## ABCI-Q Launch Commands

Login-node checks after data prep:

```bash
bash tools/concerto_projection_shortcut/check_setup_status.sh
test -d data/scannet/train
test -d data/scannet/val
test -f data/arkitscenes_absmeta/splits/Training.json
test -f data/arkitscenes_absmeta/splits/Validation.json
test -f data/weights/concerto/concerto_base_origin.pth
```

Dry-run only on `rt_QF=1`:

```bash
qsub -v RUN_FAST_GATE=0 tools/concerto_projection_shortcut/submit_projres_v1_abciq_qf.sh
```

Single-node fast gate on `rt_QF=1`:

```bash
qsub tools/concerto_projection_shortcut/submit_projres_v1_abciq_qf.sh
```

Single-node smoke diagnostic on `rt_QF=1`:

```bash
qsub -v 'GPU_IDS_CSV=0\,1\,2\,3,CONCERTO_MAX_TRAIN_ITER=8,ALPHAS_CSV=0.05' \
  tools/concerto_projection_shortcut/submit_projres_v1_smoke_abciq_qf.sh
```

Batch-only ARKit DDP diagnostic on `rt_QF=1`:

```bash
qsub -v 'GPU_IDS_CSV=0\,1\,2\,3,DEBUG_MAX_BATCHES=16,DEBUG_BATCH_SIZE=8,DEBUG_NUM_WORKER=1' \
  tools/concerto_projection_shortcut/submit_debug_arkit_ddp_batches_abciq_qf.sh
```

Short full-training trace on `rt_QF=1`:

```bash
qsub -l walltime=00:08:00 \
  -v 'GPU_IDS_CSV=0\,1\,2\,3,CONCERTO_MAX_TRAIN_ITER=8,ALPHAS_CSV=0.05,POINTCEPT_TRACE_STEPS=1,EXP_TAG=-h10016-qf1trace' \
  tools/concerto_projection_shortcut/submit_projres_v1_smoke_abciq_qf.sh
```

Run-step forward trace on `rt_QF=1`:

```bash
qsub -l walltime=00:05:00 \
  -v 'GPU_IDS_CSV=0\,1\,2\,3,CONCERTO_MAX_TRAIN_ITER=9,ALPHAS_CSV=0.05,POINTCEPT_TRACE_STEPS=1,EXP_TAG=-h10016-qf1trace9' \
  tools/concerto_projection_shortcut/submit_projres_v1_smoke_abciq_qf.sh
```

Current ABCI-Q defaults in the smoke/qf16 wrappers:
- `POINTCEPT_TRAIN_LAUNCHER=torchrun`
- `NCCL_STABLE_MODE=1`
- `NCCL_P2P_DISABLE=1`
- `NCCL_NET_GDR_LEVEL=0`
- `CONCERTO_NUM_WORKER=1`

Defaults in the fast-gate qsub wrapper:
- `GPU_IDS_CSV=0,1,2,3`
- `MAX_TRAIN_BATCHES=1024`
- `MAX_VAL_BATCHES=256`
- `MAX_ROWS_PER_BATCH=512`
- `EXP_MIRROR_ROOT=data/runs/projres_v1`

Override example for fuller prior fitting:

```bash
qsub -v MAX_TRAIN_BATCHES=4096,MAX_VAL_BATCHES=512,MAX_ROWS_PER_BATCH=512 \
  tools/concerto_projection_shortcut/submit_projres_v1_abciq_qf.sh
```

Monitor:

```bash
qstat | awk -v u="$USER" 'NR==1 || NR==2 || $0 ~ u {print}'
tail -f data/logs/abciq/projres_v1_<jobid>.log
find data/runs/projres_v1 -maxdepth 4 -type f | sort
```

ABCI-Q `qstat` on this system rejects `-u`, so use the plain `qstat | awk`
form above. Keep diagnostic walltime short. The stalled smoke jobs were
intentionally stopped instead of being left to burn allocation points.

## ABCI-Q Expected Runtime

The current chain does not use all 4 GPUs at every stage.

Expected behavior:
- prior fit: currently single GPU
- smoke: one GPU per alpha, so two GPUs for the default two alpha values
- 5-epoch continuation: uses all four GPUs in `GPU_IDS_CSV=0,1,2,3`
- stress: currently first GPU only
- ScanNet linear val-only: currently first GPU only
- fine-tune: currently first GPU only unless script/config is changed

First target:
- single-node smoke/data-loader diagnosis only
- no multi-node launch until longer single-node 4-GPU smoke is repeatable
- reuse the already selected prior; do not refit the prior unless the prior
  artifacts are explicitly removed
- do not launch the 4-node / 16-GPU continuation until the ARKit stall is
  explained

If multi-node is needed later, add a separate `pbsdsh` launcher modeled after
`/groups/qgah50055/ide/VGI/3D-NEPA`, using `PBS_NODEFILE`, `MASTER_ADDR`,
`MASTER_PORT`, and `MACHINE_RANK`, and apply it only to the continuation stage.

Rough estimate:
- 4-GPU launcher checks: 2-10 minutes depending on whether a stall appears
- 5-epoch continuation and ScanNet gate: estimate only after the 4-GPU stall is
  fixed
- fuller prior with `MAX_TRAIN_BATCHES=4096`: not needed unless the prior
  artifacts are deleted or invalidated

## Resume And Skip Logic

The chain skips stages when their output already exists.

Important output files:
- `${EXP_MIRROR_ROOT}/priors/selected_prior.json`
- `${EXP_MIRROR_ROOT}/summaries/selected_smoke.json`
- `${EXP_MIRROR_ROOT}/summaries/*_stress.csv`
- `${EXP_MIRROR_ROOT}/summaries/*_gate.json`
- `exp/concerto/<exp>/model/model_last.pth`

To force refitting the prior, remove:

```bash
rm -rf data/runs/projres_v1/priors
```

To force rerunning an experiment, remove the corresponding symlinked exp dir:

```bash
rm -rf exp/concerto/arkit-full-projres-v1a-alpha005-smoke
rm -rf exp/concerto/arkit-full-projres-v1a-alpha010-smoke
```

Do not delete unrelated `exp/concerto/*` runs; those contain previous baseline
results used by the gate.

## Known Caveats

- `fit_coord_prior.py` currently extracts cache with one process on one GPU.
- `run_projres_v1_chain.sh` can pass multiple GPUs to continuation, but not to
  all stages.
- ABCI-Q multi-GPU should use `POINTCEPT_TRAIN_LAUNCHER=torchrun`, not the
  original Pointcept `mp.spawn` launcher.
- Stable 4-GPU torchrun currently needs `NCCL_P2P_DISABLE=1`; without it,
  communication progressed only briefly and then stalled.
- The current blocker is not just batch size. Short 4-GPU runs can complete, but
  longer ARKit smoke has stalled after 8-16 batches or before the first batch on
  the second alpha arm. Treat this as a DDP/data-loader/runtime stall until
  proven otherwise.
- The batch-only diagnostic passed, so the next evidence should come from a
  narrower model-forward trace rather than more DataLoader-only runs.
- The validation-only ScanNet config intentionally removes `PreciseEvaluator`
  to avoid large full-test `.npy` output and disk failure.
- `no-enc2d-renorm` is useful but not perfectly apples-to-apples with the safe
  trio because its previous run used a different linear config and continuation
  recipe.

## What To Report After ABCI-Q Run

Update these files:
- [CURRENT_STATUS.md](./CURRENT_STATUS.md)
- [results_scannet_proxy_lin.md](./results_scannet_proxy_lin.md), if new linear
  results are comparable
- add a new `results_projres_v1.md` if the fix reaches ScanNet linear gate

Minimum report:
- selected prior type: `linear` or `mlp`
- prior validation cosine loss
- `target_energy`, `pred_energy`, `residual_norm`
- selected alpha
- smoke pass summary
- 5-epoch continuation final metrics
- stress CSV summary
- ScanNet linear last and best mIoU
- gate decision: strong go, weak go, or no-go
