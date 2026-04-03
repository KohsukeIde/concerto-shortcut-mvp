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
- `run_scannet_proxy.sh` stages official-weight gate runs, ARKit continuation
  pretraining, and ScanNet proxy linear / fine-tune evaluation.

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

## Notes

- `scripts/train.sh` still uses `-d concerto` because the configs live under
  `configs/concerto`.
- The smoke script uses `timeout` and is meant to cover only the early part of
  training.
- The summary script expects logs under `exp/concerto/arkit-shortcut-*/train.log`.
- `run_mvp.sh` runs `baseline -> zero-appearance -> coord-mlp -> jitter -> cross-scene-target-swap`.
