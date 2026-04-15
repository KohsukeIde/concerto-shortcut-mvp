# Concerto shortcut MVP on Pointcept v1.6.1

This directory contains the runnable Pointcept-side setup for the Concerto
`enc2d` shortcut probe.

For the current experiment status and the shortest path to the latest results,
start with:

- `tools/concerto_projection_shortcut/CURRENT_STATUS.md`
- `tools/concerto_projection_shortcut/HANDOFF_PROJRES_V1.md`

Update rule in this repo:

- `CURRENT_STATUS.md` is the living top-level status page.
- stage-specific finished results should live in `results_*.md` / `results_*.csv`
  and be linked back from `CURRENT_STATUS.md`.

Key choices in this version:

- target repo: `Pointcept v1.6.1`
- target dataset: `ARKitScenes` only
- target loss: `enc2d_loss` only
- wording: `correspondence / coordinate shortcut`
- current fix track: coordinate projection residualization

The original bundle notes are preserved in
`tools/concerto_projection_shortcut/README.bundle.md`.

## Included files

- `configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-baseline.py`
- `configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-zero-appearance.py`
- `configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-coord-mlp.py`
- `configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-jitter.py`
- `configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-cross-scene-target-swap.py`
- `configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-shuffle-corr.py`
- `configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-full-*.py`
- `configs/concerto/pretrain-concerto-v1m1-0-arkit-full-*.py`
- `configs/concerto/semseg-ptv3-base-v1m1-0*-scannet-*-proxy.py`
- `tools/concerto_projection_shortcut/preflight.py`
- `tools/concerto_projection_shortcut/run_smoke.sh`
- `tools/concerto_projection_shortcut/run_mvp.sh`
- `tools/concerto_projection_shortcut/summarize_logs.py`
- `tools/concerto_projection_shortcut/summarize_semseg_logs.py`
- `tools/concerto_projection_shortcut/eval_enc2d_stress.py`
- `tools/concerto_projection_shortcut/prepare_arkit_full_splits.py`
- `tools/concerto_projection_shortcut/setup_arkit_full_assets.sh`
- `tools/concerto_projection_shortcut/run_arkit_full_causal.sh`
- `tools/concerto_projection_shortcut/setup_downstream_assets.sh`
- `tools/concerto_projection_shortcut/run_scannet_proxy.sh`
- `tools/concerto_projection_shortcut/fit_coord_prior.py`
- `tools/concerto_projection_shortcut/run_projres_v1_chain.sh`
- `tools/concerto_projection_shortcut/HANDOFF_PROJRES_V1.md`
- `tools/concerto_projection_shortcut/install_extras.sh`

## Important changes from the original bundle

- The 5 default probe configs now train on `ARKitScenes` only.
- `eval_epoch = 5` is set so Pointcept does not collapse `data.train.loop` to `0`.
- `enable_wandb = False` is set to keep the MVP self-contained.
- Helper scripts default to `python3`, because `python` is not guaranteed to exist.
- The main sanity check is now `cross-scene target swap`; `shuffle-corr` is kept for manual comparison.
- The full follow-up adds `global_target_permutation`, `cross_image_target_swap`,
  and `coord_residual_target`.
- `prepare_arkit_full_splits.py` generates an absolute-path metadata root for the
  full ARKitScenes split JSONs so the repo can keep `data/arkitscenes` pointed
  at the mini subset.
- `setup_arkit_full_assets.sh` expands the locally cached full ARKitScenes
  archive and then generates the absolute-path metadata root used by the full
  causal branch configs.
- `run_arkit_full_causal.sh` targets full ARKitScenes with 2-GPU execution by default.
- `setup_downstream_assets.sh` downloads `concerto_base(_origin).pth` and the
  preprocessed ScanNet dataset from Hugging Face.
- `run_scannet_proxy.sh` now follows the cheaper priority order:
  official ScanNet linear gate -> ARKit continuation trio -> ScanNet linear proxy.
  Fine-tuning is intentionally separated into `gate-ft` / `ft`.
- `fit_coord_prior.py` fits a frozen coordinate prior `g(c)` for the projection
  residual fix.
- `run_projres_v1_chain.sh` runs the auto-stop chain:
  offline prior fit -> projection residual smoke -> 5-epoch continuation ->
  stress -> ScanNet linear validation gate -> optional fine-tune.
- `semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly.py` disables the full
  post-train test writer to avoid `.npy` disk failures.

## Current phase

The current phase is no longer the original ScanNet replacement critique. That
proxy is finished enough to conclude:

- objective-level coordinate shortcut: strong
- downstream shortcut relevance: measurable but limited
- strongest downstream claim, "mostly coordinate shortcut": not supported by
  this continuation proxy

The next experiment is the projection residual fix. Use
[HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md) as the exact runbook. The
local run on this machine was stopped during prior cache extraction and should
be resumed on the H200 server.

## H200 projection residual quick start

On the H200 server, after mounting or copying the data and official weight:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept-concerto-cu121

bash tools/concerto_projection_shortcut/check_setup_status.sh

python tools/concerto_projection_shortcut/preflight.py \
  --check-data --check-batch --check-forward \
  --config pretrain-concerto-v1m1-0-arkit-full-continue

EXP_MIRROR_ROOT=/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1 \
GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
DRY_RUN=1 \
bash tools/concerto_projection_shortcut/run_projres_v1_chain.sh
```

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

## Recommended order on 2 GPUs

1. Run the full ARKit causal branch first.
2. Treat `coord_mlp ~= baseline` plus a strong corruption gap as the go/no-go.
3. Only then download ScanNet and run the downstream proxy.
4. Keep `ScanNet200 / ScanNet++ / S3DIS` and the exact quartet for a later pass.

The helper scripts are aligned with that order:

- `run_arkit_full_causal.sh` defaults to one job per GPU and runs the priority
  list in pairs: `baseline + coord-mlp`, `global-target-permutation + cross-image-target-swap`,
  then `cross-scene-target-swap` plus the optional `coord-residual-target`.
- `resume_arkit_full_after_current.sh` waits for the currently running
  `baseline + coord-mlp` pair to finish, then resumes from the second pair and
  runs stress evaluation at the end.
- `extract_scannet_after_arkit_full.sh` waits for the ARKit full completion
  stamp and only then expands the downloaded ScanNet archive, so extraction
  does not contend with the long ARKit training runs.
- `run_scannet_proxy.sh all` intentionally stops after ScanNet linear probe.
- `run_scannet_proxy.sh gate-ft` and `run_scannet_proxy.sh ft` are separate so
  fine-tuning only happens after the linear result looks promising.

## Quick start

1. Create the device-local Pointcept environment:

```bash
bash tools/concerto_projection_shortcut/create_env.sh
```

2. Install Concerto-specific extras:

```bash
bash tools/concerto_projection_shortcut/install_extras.sh
```

3. Prepare ARKitScenes under `/mnt/urashima/users/minesawa/pointcept_data`
and link it into the repo:

```bash
bash tools/concerto_projection_shortcut/setup_arkit_full_assets.sh
```

4. Validate repo paths, dataset layout, and config import:

```bash
python3 tools/concerto_projection_shortcut/preflight.py --check-data
```

5. If the environment is fully ready, validate one collated batch:

```bash
python3 tools/concerto_projection_shortcut/preflight.py --check-data --check-batch
```

6. If GPU dependencies are ready, validate one forward pass:

```bash
python3 tools/concerto_projection_shortcut/preflight.py --check-data --check-forward
```

7. Run the bounded baseline smoke test:

```bash
bash tools/concerto_projection_shortcut/run_smoke.sh
```

8. Run the 5-experiment MVP:

```bash
bash tools/concerto_projection_shortcut/run_mvp.sh
```

9. For the ARKit full follow-up on 2 GPUs:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept-concerto-cu121
bash tools/concerto_projection_shortcut/run_arkit_full_causal.sh
```

10. If the ARKit full branch is positive, prepare downstream assets and run the
ScanNet linear proxy:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept-concerto-cu121
DOWNLOAD_WEIGHTS=0 DOWNLOAD_SCANNET=1 bash tools/concerto_projection_shortcut/setup_downstream_assets.sh
bash tools/concerto_projection_shortcut/run_scannet_proxy.sh all
```

## Notes

- `scripts/train.sh` still uses `-d concerto` because the configs live under
  `configs/concerto`.
- The smoke script uses `timeout` and is meant to cover only the early part of
  training.
- The summary script expects logs under `exp/concerto/arkit-shortcut-*/train.log`.
- `run_mvp.sh` runs `baseline -> zero-appearance -> coord-mlp -> jitter -> cross-scene-target-swap`.
