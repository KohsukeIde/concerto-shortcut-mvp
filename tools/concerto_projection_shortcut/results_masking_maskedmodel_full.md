# Masked-Model Keep20 Battery

This note extends the masking audit with `masked_model_keep0p2`, meaning that
roughly 20% of object/stuff support is retained after raw-scene object-style
masking, then evaluated with `full_nn` full-scene scoring.

## Setup

- Shared conditions:
  - `random_keep0p2`
  - `masked_model_keep0p2`
  - `fixed_points_4000`
  - `feature_zero1p0`
- Score space reported below: `full_nn`
- Existing checkpoints only; no new FT runs

## Results

| family | clean mIoU | random keep20 | masked-model keep20 | fixed 4k | feature-zero | focus class | clean focus | masked-model focus | observed keep20 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `Concerto decoder / ScanNet20` | 0.7869 | 0.7522 | 0.1982 | 0.3930 | 0.0682 | `picture` | 0.4213 | 0.0950 | 0.2030 |
| `Concerto linear / ScanNet20` | 0.7695 | 0.7516 | 0.1990 | 0.4684 | 0.0389 | `picture` | 0.4197 | 0.0821 | 0.2029 |
| `Sonata linear / ScanNet20` | 0.7167 | 0.6862 | 0.1724 | 0.3850 | 0.0606 | `picture` | 0.3699 | 0.0638 | 0.1938 |
| `PTv3 supervised / ScanNet20` | 0.7699 | 0.6969 | 0.1324 | 0.2065 | 0.0271 | `picture` | 0.3881 | 0.0524 | 0.2028 |
| `PTv3 supervised / ScanNet200` | 0.3447 | 0.2551 | 0.0441 | 0.0419 | 0.0019 | `picture` | 0.3596 | 0.0703 | 0.2035 |
| `PTv3 supervised / S3DIS` | 0.7040 | 0.4434 | 0.1083 | 0.0005 | 0.1150 | `board` | 0.8237 | 0.1048 | 0.1875 |

## Interpretation

- `masked_model_keep0p2` is substantially harsher than `random_keep0p2` across
  all available families and datasets, even when the observed retained fraction
  stays near 20%.
- This means the earlier `random_keep0p2` robustness should not be read as
  “the model solves the task with only 20% of the scene” in a strong semantic
  sense. A random 20% voxel subset still preserves a large amount of partial
  geometry and scene outline.
- Whole-object masking removes that support much more effectively. Under this
  condition, all ScanNet20 models drop to about `0.13-0.20` mIoU, and the
  weak wall-adjacent focus class (`picture`) collapses sharply.
- `fixed_points_4000` is also a strong stress, but it is not equivalent to
  object-style masking. On ScanNet20, `fixed_points_4000` is often less harmful
  than `masked_model_keep0p2`, which is consistent with object removal being a
  different failure mode from uniform point-budget reduction.
- The current masking claim is therefore:

  > retained-subset `keep20%` is a weak sparsity regime, whereas
  > object-style `masked_model_keep20%` and `fixed_points_4000` are much
  > stronger stress tests that expose weak-class fragility and remove much of
  > the apparent robustness.

## Files

- Concerto decoder summary:
  `tools/concerto_projection_shortcut/results_masking_maskedmodel_full_concerto_decoder.md`
- Concerto linear summary:
  `tools/concerto_projection_shortcut/results_masking_maskedmodel_full_concerto_linear.md`
- Sonata linear summary:
  `tools/concerto_projection_shortcut/results_masking_maskedmodel_full_sonata_linear.md`
- PTv3 ScanNet20 summary:
  `tools/concerto_projection_shortcut/results_masking_maskedmodel_full_ptv3_scannet20.md`
- PTv3 ScanNet200 summary:
  `tools/concerto_projection_shortcut/results_masking_maskedmodel_full_ptv3_scannet200.md`
- PTv3 S3DIS summary:
  `tools/concerto_projection_shortcut/results_masking_maskedmodel_full_ptv3_s3dis.md`
