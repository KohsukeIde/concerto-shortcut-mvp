# Full-Scene Masking Scoring Battery

Adds `full_nn` scoring to the existing retained-subset masking battery. For point-drop variants, retained logits are propagated back to every original voxel by nearest-neighbor assignment before scoring. Clean and feature-zero rows are identical between retained and full-scene scoring because no point is removed.

## Key Summary

| method | score | random keep20 mIoU | Δ | focus IoU | focus->conf | structured keep20 mIoU | Δ | focus IoU | focus->conf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Concerto decoder | `retained` | 0.7632 | -0.0231 | 0.3091 | 0.6323 | 0.7388 | -0.0475 | 0.4380 | 0.4155 |
| Concerto decoder | `full_nn` | 0.7527 | -0.0336 | 0.2948 | 0.6478 | 0.3012 | -0.4851 | 0.1387 | 0.6178 |
| Concerto linear | `retained` | 0.7597 | -0.0098 | 0.3503 | 0.5458 | 0.7409 | -0.0287 | 0.3851 | 0.4571 |
| Concerto linear | `full_nn` | 0.7519 | -0.0177 | 0.3442 | 0.5502 | 0.2907 | -0.4789 | 0.1040 | 0.6651 |
| Sonata linear | `retained` | 0.6951 | -0.0219 | 0.2365 | 0.6858 | 0.6712 | -0.0457 | 0.3197 | 0.5737 |
| Sonata linear | `full_nn` | 0.6865 | -0.0305 | 0.2239 | 0.6972 | 0.2662 | -0.4508 | 0.0975 | 0.6885 |
| PTv3 ScanNet20 | `retained` | 0.7131 | -0.0566 | 0.2396 | 0.7308 | 0.6573 | -0.1124 | 0.4612 | 0.4413 |
| PTv3 ScanNet20 | `full_nn` | 0.6995 | -0.0702 | 0.2301 | 0.7368 | 0.2491 | -0.5206 | 0.0876 | 0.6981 |
| PTv3 ScanNet200 | `retained` | 0.2669 | -0.0789 | 0.2418 | 0.6540 | 0.2425 | -0.1033 | 0.3609 | 0.5433 |
| PTv3 ScanNet200 | `full_nn` | 0.2579 | -0.0879 | 0.2324 | 0.6648 | 0.0772 | -0.2686 | 0.0895 | 0.5036 |
| PTv3 S3DIS | `retained` | 0.4513 | -0.2539 | 0.3333 | 0.5529 | 0.6374 | -0.0678 | 0.7733 | 0.1587 |
| PTv3 S3DIS | `full_nn` | 0.4445 | -0.2607 | 0.3297 | 0.5451 | 0.2881 | -0.4170 | 0.2644 | 0.2423 |

## Interpretation

- `random keep20%` is still a relatively dense voxel input. Example-export summaries show that on ScanNet/ScanNet200 it still leaves roughly `18k` to `33k` points per scene, and on S3DIS `32k` to `98k`. This means random keep20 is not a "near-empty input" regime. A stricter fixed-budget condition such as `fixed_points_4000` is needed when we want to claim genuinely sparse input stress.
- Random keep20 robustness mostly survives full-scene nearest-neighbor scoring on ScanNet20: Concerto decoder `0.7632` retained vs `0.7527` full, Concerto linear `0.7597` vs `0.7519`, Sonata linear `0.6951` vs `0.6865`, and PTv3 ScanNet20 `0.7131` vs `0.6995`. This suggests random retained-subset robustness is not only a scoring-subset artifact.
- Structured block masking changes sharply under full-scene scoring: Concerto decoder `0.7388` retained vs `0.3012` full, Concerto linear `0.7409` vs `0.2907`, Sonata `0.6712` vs `0.2662`, PTv3 ScanNet20 `0.6573` vs `0.2491`, PTv3 ScanNet200 `0.2425` vs `0.0772`, and PTv3 S3DIS `0.6374` vs `0.2881`. Retained-subset scoring therefore hides the cost of missing structured regions.
- Feature-zero remains collapsed across models/datasets, so the masking robustness is not coordinate-only. The more accurate claim is retained/random sparsity redundancy plus weak-class fragility, not pure coordinate shortcut.
- Class-wise keep20 remains close to random keep20, so the random keep result is not mainly class-composition drift.
- Object-style masked-model examples were exported separately for ScanNet/ScanNet200/S3DIS. Those examples are meant to remove whole-object silhouettes rather than just leave a random point subset. See `results_masking_examples.md`.

## Files

- Aggregate CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_fullscene_scoring.csv`
- Concerto decoder: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_fullscene_concerto_decoder.csv`
- Concerto linear: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_fullscene_concerto_linear.csv`
- Sonata linear: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_fullscene_sonata_linear.csv`
- PTv3 ScanNet20: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_fullscene_ptv3_scannet20.csv`
- PTv3 ScanNet200: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_fullscene_ptv3_scannet200.csv`
- PTv3 S3DIS: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_fullscene_ptv3_s3dis.csv`
- Example summary: `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_examples.md`
