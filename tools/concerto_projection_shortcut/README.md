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
- `tools/concerto_projection_shortcut/preflight.py`
- `tools/concerto_projection_shortcut/run_smoke.sh`
- `tools/concerto_projection_shortcut/run_mvp.sh`
- `tools/concerto_projection_shortcut/summarize_logs.py`
- `tools/concerto_projection_shortcut/install_extras.sh`

## Important changes from the original bundle

- The 5 default probe configs now train on `ARKitScenes` only.
- `eval_epoch = 5` is set so Pointcept does not collapse `data.train.loop` to `0`.
- `enable_wandb = False` is set to keep the MVP self-contained.
- Helper scripts default to `python3`, because `python` is not guaranteed to exist.
- The main sanity check is now `cross-scene target swap`; `shuffle-corr` is kept for manual comparison.

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
