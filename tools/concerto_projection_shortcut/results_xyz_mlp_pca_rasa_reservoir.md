# XYZ-MLP PCA RASA Pilot

## Setup
- config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- train points: `1200000`
- val points: `2000000`
- xyz hidden dim: `128`
- xyz epochs: `30`

## Predictability

- train R2: `0.4751` (dim0 `0.5833`, dim1 `0.3668`)
- val R2: `0.4925` (dim0 `0.5827`, dim1 `0.4070`)
- nuisance basis rank: `2`

## Baselines

- base decoder logits: mIoU `0.7683`, picture `0.7052`, picture->wall `0.1307`
- xyz-only MLP: mIoU `0.0576`, picture `0.0064`
- refit linear on Concerto features: mIoU `0.7317`, picture `0.5278`

## Best Variants

- best mIoU: `base_decoder_logits` mIoU `0.7683`, picture `0.7052`, picture->wall `0.1307`
- best picture: `base_decoder_logits` mIoU `0.7683`, picture `0.7052`, picture->wall `0.1307`

## Interpretation

- The task-conditioned coordinate nuisance target is linearly visible in
  Concerto decoder features: val R2 is `0.4925` for the 2D PCA target.
- This predictability is not concentrated in the problematic picture class:
  `picture_all` R2 is negative (`-0.3470`), while projection energy is higher
  for picture points than average (`0.0421` vs `0.0250`).
- RASA-style rank-2 removal does not improve downstream readout. Removing the
  full estimated subspace changes mIoU `0.7317 -> 0.7303` for the refit
  classifier and leaves `picture -> wall` essentially unchanged.
- The add-back branch also does not recover a positive variant. Current
  reading: Concerto contains a task-conditioned coordinate-derived factor, but
  this factor is entangled with useful layout/readout information rather than a
  clean removable harmful subspace.

## Output Files

- `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_rasa_reservoir/xyz_mlp_pca_rasa_variants.csv`
- `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_rasa_reservoir/xyz_mlp_pca_r2_by_subset.csv`
- `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_rasa_reservoir/xyz_mlp_pca_energy_by_subset.csv`
- `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_rasa_reservoir/metadata.json`
