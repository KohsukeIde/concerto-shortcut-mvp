# Concerto shortcut MVP on Pointcept v1.6.1

This directory contains the runnable Pointcept-side setup for the Concerto
`enc2d` shortcut probe.

Key choices in this version:

- target repo: `Pointcept v1.6.1`
- target dataset: `ARKitScenes` only
- target loss: `enc2d_loss` only
- wording: `correspondence / coordinate shortcut`

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

1. Install the base Pointcept environment.
2. Install Concerto-specific extras:

```bash
bash tools/concerto_projection_shortcut/install_extras.sh
```

3. Preprocess ARKitScenes and place it at `data/arkitscenes`.
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
source /home/cvrt/miniconda3/etc/profile.d/conda.sh
conda activate pointcept-cu128
bash tools/concerto_projection_shortcut/run_arkit_full_causal.sh
```

10. If the ARKit full branch is positive, prepare downstream assets and run the
ScanNet linear proxy:

```bash
source /home/cvrt/miniconda3/etc/profile.d/conda.sh
conda activate pointcept-cu128
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
