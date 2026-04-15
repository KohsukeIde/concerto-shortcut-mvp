# ProjRes v1 Handoff

Updated: 2026-04-15 JST

This document is the handoff entry point for moving the coordinate projection
residual fix experiment to a larger GPU server.

## Current Decision

- Do not continue the current run on this 4-GPU machine.
- Move the next run to the H200 server.
- The old `projres_v1` chain was stopped manually during Stage 1 before any
  prior cache or checkpoint was written.
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

No `projres_v1` job is currently running on this machine.

Stopped run:
- log: `/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1/logs/run_projres_v1_chain_20260415_131727_setsid.log`
- stage reached: prior fit cache extraction
- output written: logs only
- no useful prior cache, selected prior, smoke checkpoint, continuation
  checkpoint, stress csv, or linear result was produced

Reason for stopping:
- current prior cache extraction is too slow on this server
- H200 8-GPU server is the better next environment

## H200 Server Setup

Preferred layout:

```text
/mnt/urashima/users/minesawa/pointcept_data
/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1
```

If `/mnt/urashima` is mounted on the H200 server, use the defaults.

If the H200 server uses a different data root, override these variables:

```bash
POINTCEPT_DATA_ROOT=/path/to/pointcept_data
ARKIT_FULL_META_ROOT=/path/to/arkitscenes_absmeta
SCANNET_EXTRACT_DIR=/path/to/scannet
EXP_MIRROR_ROOT=/path/to/concerto_shortcut_runs/projres_v1
OFFICIAL_WEIGHT=/path/to/weights/concerto/concerto_base_origin.pth
```

Minimum required inputs:
- ARKit full absolute metadata root:
  `/mnt/urashima/users/minesawa/pointcept_data/arkitscenes/arkitscenes_absmeta`
- ScanNet processed root:
  `/mnt/urashima/users/minesawa/pointcept_data/scannet`
- official Concerto weight:
  `weights/concerto/concerto_base_origin.pth`
- conda env:
  `pointcept-concerto-cu121`

Before launching:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept-concerto-cu121
bash tools/concerto_projection_shortcut/check_setup_status.sh
python tools/concerto_projection_shortcut/preflight.py \
  --check-data --check-batch --check-forward \
  --config pretrain-concerto-v1m1-0-arkit-full-continue
```

Dry-run the chain:

```bash
EXP_MIRROR_ROOT=/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1 \
GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
DRY_RUN=1 \
bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh
```

## H200 Launch Commands

Fast first gate:

```bash
EXP_MIRROR_ROOT=/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1 \
GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
MAX_TRAIN_BATCHES=1024 \
MAX_VAL_BATCHES=256 \
MAX_ROWS_PER_BATCH=512 \
OFFICIAL_WEIGHT=/home/minesawa/ssl/concerto-shortcut-mvp/weights/concerto/concerto_base_origin.pth \
bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh
```

Fuller prior fit:

```bash
EXP_MIRROR_ROOT=/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1 \
GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
MAX_TRAIN_BATCHES=4096 \
MAX_VAL_BATCHES=512 \
MAX_ROWS_PER_BATCH=512 \
OFFICIAL_WEIGHT=/home/minesawa/ssl/concerto-shortcut-mvp/weights/concerto/concerto_base_origin.pth \
bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh
```

Detached launch:

```bash
mkdir -p /mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1/logs
nohup setsid bash -lc '
cd /home/minesawa/ssl/concerto-shortcut-mvp &&
source "$(conda info --base)/etc/profile.d/conda.sh" &&
conda activate pointcept-concerto-cu121 &&
EXP_MIRROR_ROOT=/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1 \
GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
MAX_TRAIN_BATCHES=1024 \
MAX_VAL_BATCHES=256 \
MAX_ROWS_PER_BATCH=512 \
OFFICIAL_WEIGHT=/home/minesawa/ssl/concerto-shortcut-mvp/weights/concerto/concerto_base_origin.pth \
bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh
' > /mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1/logs/run_projres_v1_chain_h200.nohup.log 2>&1 &
```

Monitor:

```bash
tail -f /mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1/logs/run_projres_v1_chain_h200.nohup.log
nvidia-smi
find /mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1 -maxdepth 4 -type f | sort
```

## H200 Expected Runtime

The current chain does not use all 8 GPUs at every stage.

Expected behavior:
- prior fit: currently single GPU
- smoke: one GPU per alpha, so two GPUs for the default two alpha values
- 5-epoch continuation: uses all GPUs in `GPU_IDS_CSV`
- stress: currently first GPU only
- ScanNet linear val-only: currently first GPU only
- fine-tune: currently first GPU only unless script/config is changed

Rough estimate with H200:
- fast first gate with `MAX_TRAIN_BATCHES=1024`: likely under 1 day to ScanNet
  linear gate
- fuller prior with `MAX_TRAIN_BATCHES=4096`: likely longer, but still much
  faster than this server

If full 8-GPU utilization is required, improve these before launch:
- parallelize prior cache extraction across scenes or splits
- let ScanNet linear use multiple GPUs cleanly
- run more alpha smoke arms concurrently across available GPUs

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
rm -rf /mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1/priors
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
- `run_projres_v1_chain.sh` can pass 8 GPUs to continuation, but not to all
  stages.
- The validation-only ScanNet config intentionally removes `PreciseEvaluator`
  to avoid large full-test `.npy` output and disk failure.
- `no-enc2d-renorm` is useful but not perfectly apples-to-apples with the safe
  trio because its previous run used a different linear config and continuation
  recipe.

## What To Report After H200 Run

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
