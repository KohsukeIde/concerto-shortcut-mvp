# Masking Example Export

This document records the concrete input sparsity regimes behind the masking battery. The key interpretation is that `keep20%` is still a fairly dense input after voxelization, while `fixed_points_4000` is the first regime that reliably pushes the input down to a few percent of the original points.

## Example Root

- ScanNet: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_examples/scannet`
- ScanNet200: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_examples/scannet200`
- S3DIS: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/masking_examples/s3dis`

Each dataset is organized as:

`dataset / scene / condition / {input_clean,input_masked}.{npz,ply}`

The exported conditions are:

- `clean_voxel`
- `random_keep0p2`
- `random_keep0p1`
- `fixed_points_4000`
- `masked_model_keep0p2`

The masked-model condition drops whole object instances when available and randomly subsamples stuff/background points at the same keep ratio, so the original object silhouette is not preserved by sparse point leftovers alone.

## Scene Set

- ScanNet / ScanNet200: `scene0011_00`, `scene0011_01`, `scene0015_00`, `scene0019_00`, `scene0019_01`
- S3DIS Area_5: `WC_1`, `WC_2`, `conferenceRoom_1`, `conferenceRoom_2`, `conferenceRoom_3`

## Point-Count Summary

### ScanNet

- `random_keep0p2`: `18,525` to `33,049` points, mean `27,396` (`0.2006` keep)
- `random_keep0p1`: `9,277` to `16,447` points, mean `13,631` (`0.0999` keep)
- `fixed_points_4000`: always `4,000` points, which is only `2.43%` to `4.31%`
- `masked_model_keep0p2`: `10,639` to `58,235` points, mean `38,664` (`0.0737` to `0.4701`)

### ScanNet200

- `random_keep0p2`: `18,320` to `32,977` points, mean `27,327` (`0.2000` keep)
- `random_keep0p1`: `9,250` to `16,406` points, mean `13,637` (`0.0999` keep)
- `fixed_points_4000`: always `4,000` points, which is only `2.43%` to `4.31%`
- `masked_model_keep0p2`: `13,577` to `60,842` points, mean `28,098` (`0.0891` to `0.3244`)

### S3DIS

- `random_keep0p2`: `32,544` to `98,117` points, mean `59,312` (`0.2006` keep)
- `random_keep0p1`: `16,428` to `49,124` points, mean `29,650` (`0.1004` keep)
- `fixed_points_4000`: always `4,000` points, which is only `0.82%` to `2.47%`
- `masked_model_keep0p2`: `10,743` to `66,877` points, mean `44,423` (`0.0402` to `0.3722`)

## Interpretation

- `random_keep20%` is not a near-empty regime. On ScanNet-like scenes it still leaves roughly `18k` to `33k` voxelized points, and on S3DIS it leaves `32k` to `98k`.
- This explains why `random_keep20%` can still retain substantial performance even under full-scene scoring: the model is still seeing a large amount of geometry.
- `fixed_points_4000` is the cleaner stress regime for the claim "the input is now genuinely sparse." On these scenes it corresponds to roughly `0.8%` to `4.3%` of the voxelized input.
- `masked_model_keep20%` is not point-budget matched. It is intentionally a different stress family: whole-object removal plus background thinning. Its observed keep fraction varies because some scenes contain large object instances while others are dominated by stuff/background points.
- Therefore the masking interpretation should explicitly distinguish:
  1. random retained-subset sparsity,
  2. fixed-budget sparsity,
  3. object/instance-style removal.

## Source CSVs

- `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_examples_scannet.csv`
- `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_examples_scannet200.csv`
- `/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_masking_examples_s3dis.csv`
