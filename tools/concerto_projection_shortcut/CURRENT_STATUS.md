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
  - The ScanNet downstream replacement test is still in progress.
  - The original 2-GPU ScanNet gate crashed in the `mp.spawn` path, so the
    current path is a safe single-GPU proxy.
  - The official ScanNet linear gate now completes, but the continuation trio
    and downstream linear trio are not all finished yet.

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
3. ScanNet gate status:
   - [results_scannet_gate_2026-04-09.md](./results_scannet_gate_2026-04-09.md)
4. Short narrative summary:
   - [results_interim_summary_2026-04-06.md](./results_interim_summary_2026-04-06.md)
5. Reproduction / runner overview:
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
- In progress.

What was confirmed:
- The dataset path and weights are usable.
- A single-GPU safe smoke run reaches actual training:
  - [exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log](../../exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log)
- The official ScanNet linear gate now completes on the safe single-GPU path:
  - [exp/concerto/scannet-proxy-official-origin-lin/train.log](../../exp/concerto/scannet-proxy-official-origin-lin/train.log)
  - final `Val result: mIoU/mAcc/allAcc = 0.1752 / 0.2467 / 0.6167`
  - checkpoints:
    - [exp/concerto/scannet-proxy-official-origin-lin/model/model_best.pth](../../exp/concerto/scannet-proxy-official-origin-lin/model/model_best.pth)
    - [exp/concerto/scannet-proxy-official-origin-lin/model/model_last.pth](../../exp/concerto/scannet-proxy-official-origin-lin/model/model_last.pth)

What failed:
- The original 2-GPU official gate crashed repeatedly in the distributed spawn path:
  - [logs/scannet_gate.launch.log](./logs/scannet_gate.launch.log)

Current interpretation:
- The blocker is not the basic dataset path.
- The blocker is the multi-GPU `mp.spawn` path.
- The current downstream go/no-go is still pending because the replacement
  comparison has not finished yet.

## Active Downstream Jobs

Running now:
- `no-enc2d` continuation:
  - [exp/concerto/arkit-full-continue-no-enc2d/train.log](../../exp/concerto/arkit-full-continue-no-enc2d/train.log)
- `coord_mlp` continuation:
  - [exp/concerto/arkit-full-continue-coord-mlp-debug2/train.log](../../exp/concerto/arkit-full-continue-coord-mlp-debug2/train.log)

Follow-up orchestration:
- safe follow-up chain service:
  - `systemctl --user status concerto-safe-followup --no-pager -l`
- follow-up chain log:
  - [logs/safe_followup_chain.log](./logs/safe_followup_chain.log)

Expected next stages:
1. Finish `no-enc2d` and `coord_mlp` continuations.
2. Promote `coord_mlp-debug2` checkpoint to the canonical `coord_mlp` path.
3. Launch `concerto_continue` if it is still missing.
4. Launch the ScanNet linear trio after the three continuation checkpoints exist.

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
- ScanNet gate success log:
  - [exp/concerto/scannet-proxy-official-origin-lin/train.log](../../exp/concerto/scannet-proxy-official-origin-lin/train.log)
- ScanNet gate result note:
  - [results_scannet_gate_2026-04-09.md](./results_scannet_gate_2026-04-09.md)
- ScanNet safe smoke log:
  - [exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log](../../exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log)
- Pipeline status:
  - [scannet_pipeline_status.md](./scannet_pipeline_status.md)
- Active follow-up chain:
  - [logs/safe_followup_chain.log](./logs/safe_followup_chain.log)

## Immediate Next Step

1. Finish the continuation trio under the safe single-GPU path.
2. Run the ScanNet linear trio from those continuation checkpoints.
3. Decide downstream go/no-go from the replacement comparison.
