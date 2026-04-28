# PointGPT Grouping Ablation

Inference-time grouping/patchization ablation on ScanObjectNN `obj_bg`.
The checkpoint and downstream head are fixed; only the Group module's center/neighborhood construction is changed.

| variant | group mode | clean acc | Δ clean vs FPS | random keep20 | structured keep20 | structured/random damage |
|---|---|---:|---:|---:|---:|---:|
| `PointGPT-S official` | `fps_knn` | `0.9002` | `+0.0000` | `0.4441` | `0.2048` | `1.52` |
| `PointGPT-S official` | `radius_fps` | `0.9139` | `+0.0138` | `0.4389` | `0.2289` | `1.44` |
| `PointGPT-S official` | `random_center_knn` | `0.8950` | `-0.0052` | `0.3993` | `0.1945` | `1.41` |
| `PointGPT-S official` | `random_group` | `0.0757` | `-0.8244` | `0.0740` | `0.0637` | `7.00` |
| `PointGPT-S official` | `voxel_center_knn` | `0.9036` | `+0.0034` | `0.4423` | `0.2117` | `1.50` |
| `PointGPT-S no-mask` | `fps_knn` | `0.8709` | `+0.0000` | `0.5181` | `0.2169` | `1.85` |
| `PointGPT-S no-mask` | `radius_fps` | `0.8795` | `+0.0086` | `0.5250` | `0.2238` | `1.85` |
| `PointGPT-S no-mask` | `random_center_knn` | `0.8382` | `-0.0327` | `0.4544` | `0.2169` | `1.62` |
| `PointGPT-S no-mask` | `random_group` | `0.0878` | `-0.7831` | `0.0775` | `0.0929` | `-0.50` |
| `PointGPT-S no-mask` | `voxel_center_knn` | `0.8657` | `-0.0052` | `0.5284` | `0.2513` | `1.82` |

## Interpretation

- Changing FPS center selection to random or voxel-distributed centers while keeping kNN neighborhoods has modest effect on clean accuracy.
- Radius neighborhoods with FPS centers are also close to the default row.
- Destroying local neighborhoods with `random_group` collapses clean accuracy to near chance, so the model is not purely context/class-prior driven.
- Structured keep20 remains much harsher than random keep20 for all non-destructive grouping modes.
- This supports a scoped architecture statement: local grouping is essential, but the observed random-drop robustness is not explained by center selection alone. Stronger claims about current point architectures require retrained grouping variants or scene-side grouping ablations.

- CSV: `tools/concerto_projection_shortcut/results_pointgpt_grouping_ablation.csv`
