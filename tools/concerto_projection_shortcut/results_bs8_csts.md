# Concerto shortcut follow-up on ARKitScenes mini (`batch_size=8`)

## Setup

- repo: standalone `concerto-shortcut-mvp`
- dataset: `ARKitScenes mini` at `data/arkitscenes`
- env: `pointcept-cu128`
- run mode: single GPU per run, 5 epochs
- probes compared:
  - `baseline`
  - `cross-scene-target-swap`
  - `coord-mlp`

## Epoch averages (`enc2d_loss`)

| Run | Completed epoch averages (`enc2d_loss`) | Last completed epoch |
| --- | --- | --- |
| `baseline` | `7.1419, 6.6452, 6.4737, 6.3938, 6.4267` | `6.4267` |
| `cross-scene-target-swap` | `7.1711, 6.6023, 6.5411, 6.4492, 6.4871` | `6.4871` |
| `coord-mlp` | `8.1180, 6.5423, 6.5280, 6.4004, 6.3899` | `6.3899` |

## Readout

- `cross-scene-target-swap` does **not** produce a large degradation relative to `baseline`.
- The final gap is small: `6.4871 - 6.4267 = +0.0604`.
- `coord-mlp` remains fully competitive and slightly beats the `baseline` final epoch on this mini setup.
- This strengthens the current claim:
  - the released `enc2d` objective remains strongly compatible with a coordinate shortcut, even under stronger teacher-target corruption.

## Notes

- `coord-mlp` was resumed from `epoch_2` after a machine crash; the final log contains all 5 completed epoch averages.
- The `cross_scene_target_swap` implementation includes a fallback global target permutation for degenerate batches with only one valid scene worth of teacher targets, so the run does not crash.
