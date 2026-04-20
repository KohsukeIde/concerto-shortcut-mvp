# Class-Wise Keep Masking

Follow-up to the masking/ranking battery. Instead of globally sampling 20% of
the retained voxels, this variant keeps approximately 20% **within each GT
class**. This controls for class-composition drift in the retained subset.

## Setup

- Evaluation space: same retained voxel subset as the existing masking battery.
- Variant: `classwise_keep0p2`
- Compared against `random_keep0p2` in the same run.
- Models:
  - `concerto_decoder_origin`
  - `concerto_linear_origin`
  - `sonata_linear_scannet_downloaded`

## Results

| method | clean mIoU | random keep 20% | class-wise keep 20% | clean picture | random picture | class-wise picture | random p->wall | class-wise p->wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `concerto_decoder_origin` | 0.7865 | 0.7626 | 0.7629 | 0.4246 | 0.3137 | 0.2916 | 0.6098 | 0.6327 |
| `concerto_linear_origin` | 0.7689 | 0.7602 | 0.7603 | 0.4212 | 0.3498 | 0.3569 | 0.5603 | 0.5408 |
| `sonata_linear_scannet_downloaded` | 0.7169 | 0.6951 | 0.6951 | 0.3711 | 0.2285 | 0.2248 | 0.6900 | 0.6911 |

## Interpretation

- Class-wise keep 20% is almost identical to global random keep 20% in overall
  mIoU for all three valid comparators.
- Therefore the high retained-subset mIoU is **not mainly an artifact of random
  masking changing class composition**.
- Weak-class fragility remains. In particular, `picture` still drops sharply
  and `picture -> wall` still increases under class-wise keep. The class-wise
  control therefore strengthens the current interpretation:

  > Retained-subset ScanNet evaluation is redundant under heavy sparsity for
  > dominant/easy classes, but weak wall-adjacent classes remain fragile. This
  > is not explained by coordinate-only baselines or by class-composition drift.

## Files

- Summary CSV:
  `tools/concerto_projection_shortcut/results_masking_classwise_keep.csv`
- Concerto decoder:
  `tools/concerto_projection_shortcut/results_masking_classwise_concerto_decoder.md`
- Concerto linear:
  `tools/concerto_projection_shortcut/results_masking_classwise_concerto_linear.md`
- Sonata linear:
  `tools/concerto_projection_shortcut/results_masking_classwise_sonata_linear.md`
