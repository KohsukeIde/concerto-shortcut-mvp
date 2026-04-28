# ShapeNetPart Grouping Ablation

Inference-time grouping/patchization ablation on ShapeNetPart.
The checkpoint and segmentation head are fixed; only grouping center/neighborhood construction is changed.

| variant | group mode | clean cls-IoU | Δ clean vs FPS | random keep20 | structured keep20 | part-drop largest | xyz-zero |
|---|---|---:|---:|---:|---:|---:|---:|
| `PointGPT-S official` | `fps_knn` | `0.8335` | `+0.0000` | `0.6869` | `0.6456` | `0.4940` | `0.2585` |
| `PointGPT-S official` | `radius_fps` | `0.8314` | `-0.0021` | `0.6933` | `0.6417` | `0.4955` | `0.2588` |
| `PointGPT-S official` | `random_center_knn` | `0.8049` | `-0.0287` | `0.6588` | `0.6356` | `0.4818` | `0.2588` |
| `PointGPT-S official` | `random_group` | `0.4261` | `-0.4075` | `0.4282` | `0.5935` | `0.3258` | `0.2583` |
| `PointGPT-S official` | `voxel_center_knn` | `0.8257` | `-0.0079` | `0.6781` | `0.6321` | `0.4794` | `0.2583` |
| `PointGPT-S no-mask` | `fps_knn` | `0.8287` | `+0.0000` | `0.6957` | `0.6651` | `0.4805` | `0.2323` |
| `PointGPT-S no-mask` | `radius_fps` | `0.8308` | `+0.0021` | `0.6987` | `0.6600` | `0.4768` | `0.2327` |
| `PointGPT-S no-mask` | `random_center_knn` | `0.8027` | `-0.0260` | `0.6689` | `0.6505` | `0.4656` | `0.2328` |
| `PointGPT-S no-mask` | `random_group` | `0.4435` | `-0.3852` | `0.4361` | `0.5970` | `0.3992` | `0.2322` |
| `PointGPT-S no-mask` | `voxel_center_knn` | `0.8186` | `-0.0101` | `0.6898` | `0.6414` | `0.4744` | `0.2323` |

## Interpretation

- This is a dense-task companion to the ScanObjectNN grouping diagnostic.
- `random_group` sharply degrades dense part segmentation while center-selection variants stay close to FPS. The safe architecture statement is that local patch neighborhoods are essential, but FPS center selection alone is not the source of random-drop robustness.
- This is still an inference-time diagnostic; retrained grouping variants are required for a stronger architecture-causality claim.

- CSV: `tools/concerto_projection_shortcut/results_shapenetpart_grouping_ablation.csv`
