# Interim Summary (2026-04-06)

## Main Takeaway

The released Concerto `enc2d` objective shows a strong coordinate shortcut on full ARKitScenes.
`coord_mlp` does not fully match the original branch, but it closes most of the gap relative to the stronger correspondence corruptions.

## ARKit Full Causal Branch

| experiment | enc2d first | enc2d last | delta vs baseline |
| --- | ---: | ---: | ---: |
| baseline | 9.9837 | 5.9470 | 0.0000 |
| coord_mlp | 10.0974 | 6.4204 | +0.4734 |
| global_target_permutation | 9.9086 | 6.4563 | +0.5093 |
| cross_scene_target_swap | 9.9656 | 6.4526 | +0.5056 |
| cross_image_target_swap | 9.9975 | 6.4873 | +0.5403 |
| coord_residual_target | 9.9929 | 7.2936 | +1.3466 |

Interpretation:
- `coord_mlp` is surprisingly competitive on full ARKitScenes.
- Strong correspondence corruption hurts the objective, but not catastrophically.
- This supports a `correspondence-induced coordinate shortcut / scene-coordinate cache shortcut` reading.
- The current `coord_residual_target` implementation is not yet a successful fix.

## Corrected Stress Test

| checkpoint | clean | local_surface_destroy | z_flip | xy_swap | roll_90_x |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.453537 | 3.897785 | 3.980001 | 3.450245 | 3.845464 |
| coord_mlp | 3.536469 | 3.544549 | 3.620394 | 3.537728 | 3.601248 |
| coord_residual_target | 3.628110 | 4.223210 | 4.181930 | 3.624020 | 4.302245 |

Interpretation:
- The original baseline is more sensitive to `local_surface_destroy`, `z_flip`, and `roll_90_x`.
- `coord_mlp` is much flatter across stresses, which is consistent with weaker dependence on local geometry.
- `xy_swap` barely changes either baseline or `coord_mlp`.

## What Is Done vs Not Done

Done:
- Full ARKit causal branch.
- Strong correspondence corruption at the objective level.
- Geometry-vs-coordinate stress test at the objective level.
- One minimal-fix attempt (`coord_residual_target`).

Not done yet:
- Full downstream replacement test on ScanNet / ScanNet200 / ScanNet++ / S3DIS.
- Language probing stress.
- Stronger corruption variants such as patch-grid permutation and wrong-camera correspondence.

## Current Blocker

The downstream ScanNet gate is blocked by the current dataset root.
`/home/cvrt/datasets/scannet` still exposes only `images/` and `test/`, while Pointcept semseg expects top-level `train/`, `val/`, `splits/`, and `tasks/`.

An in-place re-extraction from the downloaded compressed snapshot is still running.
