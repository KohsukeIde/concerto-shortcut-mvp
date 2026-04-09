# Concerto Shortcut Current Status

This file is the single entry point for the current state of the shortcut
investigation.

## Current Bottom Line

- Strongest supported claim:
  - The released Concerto `enc2d` objective admits a strong coordinate shortcut.
- Scope of that claim:
  - Objective-level evidence on `ARKitScenes`.
  - Not yet a downstream-level claim about full Concerto.
- Current blocker:
  - The ScanNet downstream gate is not complete yet.
  - The original 2-GPU ScanNet gate crashed in the `mp.spawn` path.
  - As of `2026-04-09`, the machine also reports a GPU driver visibility issue
    (`nvidia-smi` cannot talk to the NVIDIA driver), so no GPU jobs are running.

## Best Documents To Read First

1. Objective-level conclusion:
   - [results_arkit_full_causal.md](./results_arkit_full_causal.md)
2. Geometry-vs-coordinate stress result:
   - [results_arkit_full_stress_corrected.md](./results_arkit_full_stress_corrected.md)
3. Short narrative summary:
   - [results_interim_summary_2026-04-06.md](./results_interim_summary_2026-04-06.md)
4. Reproduction / runner overview:
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
- Not finished.

What was confirmed:
- The dataset path and weights are usable.
- A single-GPU safe smoke run reaches actual training:
  - [exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log](../../exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log)

What failed:
- The original 2-GPU official gate crashed repeatedly in the distributed spawn path:
  - [logs/scannet_gate.launch.log](./logs/scannet_gate.launch.log)

Current interpretation:
- The blocker is not the basic dataset path.
- The blocker is the multi-GPU `mp.spawn` path, plus the current GPU driver issue.

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
  - [logs/scannet_gate.launch.log](./logs/scannet_gate.launch.log)
- ScanNet safe smoke log:
  - [exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log](../../exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log)
- Pipeline status:
  - [scannet_pipeline_status.md](./scannet_pipeline_status.md)

## Immediate Next Step

1. Recover GPU driver visibility so `nvidia-smi` works again.
2. Resume the ScanNet gate with the safe single-GPU path.
3. If the gate succeeds, run the continuation trio and ScanNet linear trio.
