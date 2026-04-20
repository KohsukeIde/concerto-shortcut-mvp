# Masking Ranking Battery

Same-protocol ScanNet voxel-level masking comparison across available Concerto
readouts, Sonata, a valid supervised PTv3 compatibility row, and cheap layout
baselines.

## Setup

- Primary decoder: `concerto_decoder_origin`
- Linear readout: `concerto_linear_origin`
- External SSL comparator: downloaded Sonata linear probe, merged from released
  `sonata.pth` backbone and `sonata_linear_prob_head_sc.pth`
- Supervised comparator: downloaded PTv3 v1.5.1 checkpoint evaluated through
  the official Pointcept v1.5.1 model / transform implementation, while reading
  the current `.npy` ScanNet scenes
- Baselines: train-majority wall, coordinate-only MLP, class-balanced
  coordinate-only MLP
- Evaluation space: same input-point voxel-level space used by the clean and
  masked rows
- Mask variants: clean, random keep ratios, structured block keep ratios, and
  full feature zeroing
- Note: released PTv3 rows under the current repo/data protocol were invalid.
  The `ptv3_supervised_v151_compat` row fixes that by using the official
  v1.5.1 code path.

## Key Results

| method | clean mIoU | random keep 20% | structured keep 20% | feature zero | clean picture | random keep 20% picture | p->wall clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `concerto_decoder_origin` | 0.7874 | 0.7636 | 0.7569 | 0.0680 | 0.4153 | 0.3215 | 0.4220 |
| `concerto_linear_origin` | 0.7691 | 0.7589 | 0.7551 | 0.0390 | 0.4230 | 0.3546 | 0.3956 |
| `ptv3_supervised_v151_compat` | 0.7697 | 0.7143 | 0.6521 | 0.0269 | 0.3917 | 0.2264 | 0.4694 |
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
- The valid supervised PTv3 compatibility row is also robust to retained random
  sparsity but less so than the Concerto readouts: clean `0.7697`, random keep
  20% `0.7143`, structured keep 20% `0.6521`, feature-zero `0.0269`. This
  removes the earlier missing-supervised-comparator caveat and shows that
  heavy-sparsity tolerance is not SSL-specific, while Concerto decoder/linear
  retain more random-sparsity mIoU in this retained-subset protocol.
- Class-wise keep 20% was added as a control for retained-subset class
  composition. It is nearly identical to global random keep 20% in overall mIoU:
  Concerto decoder `0.7626` random vs `0.7629` class-wise, Concerto linear
  `0.7602` vs `0.7603`, and Sonata linear `0.6951` vs `0.6951`. This rules out
  class-composition drift as the primary explanation for the high retained-subset
  mIoU. It does not rescue `picture`; weak-class fragility remains.
- Clean-to-masked ranking shift is present but not yet a full protocol critique.
  Under random keep 20%, Concerto decoder remains highest (`0.7636`) and PTv3
  drops below Concerto linear and Sonata (`0.7143`). Under structured keep 20%,
  PTv3 falls further (`0.6521`) while Concerto decoder/linear remain near
  `0.7569/0.7551`. This suggests retained-subset sparsity can change
  robustness ranking, but not enough by itself to claim a broken benchmark.
- Current claim status: this is a useful evaluation-pilot and an anti
  coord-only-baseline result, not shortcut-proof task-level evidence.

## Files

- Ranking CSV: `tools/concerto_projection_shortcut/results_masking_ranking_battery.csv`
- Decoder masking: `tools/concerto_projection_shortcut/results_masking_battery_full.md`
- Linear masking: `tools/concerto_projection_shortcut/results_masking_linear_origin_full.md`
- Sonata masking: `tools/concerto_projection_shortcut/results_masking_sonata_linear_full.md`
- PTv3 v1.5.1 compatibility masking:
  `tools/concerto_projection_shortcut/results_ptv3_v151_masking_compat_full.md`
- Class-wise keep control: `tools/concerto_projection_shortcut/results_masking_classwise_keep.md`
- Coordinate and majority baselines: `tools/concerto_projection_shortcut/results_masking_coord_baselines_full.md`
- Balanced coordinate baseline: `tools/concerto_projection_shortcut/results_masking_coord_baselines_balanced_full.md`
