# Masking Ranking Battery

Same-protocol ScanNet voxel-level masking comparison across available Concerto
readouts and cheap layout baselines.

## Setup

- Primary decoder: `concerto_decoder_origin`
- Linear readout: `concerto_linear_origin`
- External SSL comparator: downloaded Sonata linear probe, merged from released
  `sonata.pth` backbone and `sonata_linear_prob_head_sc.pth`
- Baselines: train-majority wall, coordinate-only MLP, class-balanced
  coordinate-only MLP
- Evaluation space: same input-point voxel-level space used by the clean and
  masked rows
- Mask variants: clean, random keep ratios, structured block keep ratios, and
  full feature zeroing
- Missing comparator: valid fully supervised PTv3 is not currently in this
  local ranking table; downloaded released PTv3 checkpoints were invalid under
  the current repo/data protocol

## Key Results

| method | clean mIoU | random keep 20% | structured keep 20% | feature zero | clean picture | random keep 20% picture | p->wall clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `concerto_decoder_origin` | 0.7874 | 0.7636 | 0.7569 | 0.0680 | 0.4153 | 0.3215 | 0.4220 |
| `concerto_linear_origin` | 0.7691 | 0.7589 | 0.7551 | 0.0390 | 0.4230 | 0.3546 | 0.3956 |
| `sonata_linear_scannet_downloaded` | 0.7169 | 0.6942 | 0.6752 | 0.0607 | 0.3662 | 0.2266 | 0.4588 |
| `coord_mlp` | 0.0726 | 0.0732 | 0.0749 | 0.0726 | 0.0000 | 0.0000 | 0.2162 |
| `coord_mlp_balanced` | 0.0707 | 0.0715 | 0.0702 | 0.0707 | 0.0070 | 0.0070 | 0.1954 |
| `train_majority_wall` | 0.0151 | 0.0150 | 0.0146 | 0.0151 | 0.0000 | 0.0000 | 1.0000 |

## Interpretation

- Heavy random and block sparsity tolerance is real for the available Concerto
  decoder and linear readouts. Decoder clean mIoU drops only `0.7874 -> 0.7636`
  under random keep 20%, and linear drops only `0.7691 -> 0.7589`.
- The tolerance is not explained by a pure coordinate-only or majority-layout
  shortcut. Coordinate-only MLPs remain near `0.07` mIoU and fail on `picture`;
  train-majority wall is near `0.015` mIoU.
- Full feature zeroing collapses both Concerto readouts, so the masked
  tolerance is not evidence that coordinates alone solve ScanNet. The model is
  still using input features strongly.
- The external SSL comparator shows the same qualitative pattern: Sonata linear
  stays reasonably high under random keep 20% (`0.7169 -> 0.6942`) and collapses
  under feature-zero (`0.0607`). This suggests retained-voxel sparsity tolerance
  is not unique to Concerto, and still depends on non-coordinate input features.
- Clean-to-masked ranking shift is weak with the current valid comparators.
  Concerto decoder stays above Concerto linear and Sonata linear for clean,
  random keep 20%, and structured keep 20%. Linear only overtakes decoder under
  the most extreme random keep 10% row.
- Current claim status: this is a useful evaluation-pilot and an anti
  coord-only-baseline result, not shortcut-proof task-level evidence. The next
  meaningful ranking extension is a valid supervised PTv3 comparator under the
  same mask variants.

## Files

- Ranking CSV: `tools/concerto_projection_shortcut/results_masking_ranking_battery.csv`
- Decoder masking: `tools/concerto_projection_shortcut/results_masking_battery_full.md`
- Linear masking: `tools/concerto_projection_shortcut/results_masking_linear_origin_full.md`
- Sonata masking: `tools/concerto_projection_shortcut/results_masking_sonata_linear_full.md`
- Coordinate and majority baselines: `tools/concerto_projection_shortcut/results_masking_coord_baselines_full.md`
- Balanced coordinate baseline: `tools/concerto_projection_shortcut/results_masking_coord_baselines_balanced_full.md`
