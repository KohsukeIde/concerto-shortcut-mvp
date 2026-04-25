# XYZ-MLP PCA Error-Conditioned Energy

## Setup
- config: `/groups/qgah50055/ide/concerto-shortcut-mvp/configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- state: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_rasa_reservoir/xyz_mlp_pca_rasa_state.pt`
- val batches: `312`
- val points: `35881054`

## Key Subsets

| subset | count | energy mean | energy p90 | uhat norm mean | R2 |
|---|---:|---:|---:|---:|---:|
| `all` | `35881054` | `0.0251` | `0.0459` | `1.0705` | `0.4935` |
| `hard_correct_all` | `3692348` | `0.0095` | `0.0198` | `0.6125` | `0.2646` |
| `hard_error_to_majority` | `403221` | `0.0234` | `0.0480` | `0.8898` | `0.3101` |
| `hard_error_any` | `652600` | `0.0194` | `0.0426` | `0.8213` | `0.2751` |
| `majority_correct_all` | `25877731` | `0.0279` | `0.0487` | `1.1553` | `0.5192` |
| `picture_correct` | `112009` | `0.0376` | `0.0552` | `1.4662` | `-0.2250` |
| `picture_to_wall` | `90510` | `0.0413` | `0.0618` | `1.2319` | `-0.0517` |
| `counter_to_cabinet` | `20764` | `0.0076` | `0.0166` | `0.5416` | `-0.1403` |
| `sink_to_cabinet` | `8903` | `0.0065` | `0.0149` | `0.4979` | `-0.1145` |
| `desk_to_table` | `20839` | `0.0131` | `0.0227` | `0.7326` | `0.0869` |
| `door_to_wall` | `110559` | `0.0207` | `0.0376` | `0.8056` | `0.1611` |
| `shower_curtain_to_wall` | `8840` | `0.0154` | `0.0256` | `0.7007` | `-0.4650` |

## Interpretation

- Error-conditioned energy gives a more nuanced result than the first rank-2 RASA removal pilot.
- Hard-class errors to majority classes have higher projection energy than correct hard-class points (`0.0234` vs. `0.0095`) and slightly higher R2 (`0.3101` vs. `0.2646`). Thus the coordinate-derived factor is not entirely unrelated to hard-class errors.
- However, correct majority-class points still have the strongest coordinate-target signature (`energy=0.0279`, `R2=0.5192`). The dominant signal is ordinary majority/layout prediction, not a clean hard-failure-only shortcut.
- `picture` is the key negative case: `picture_to_wall` has high energy (`0.0413`) but near/under-zero R2 (`-0.0517`), and `picture_correct` is also high-energy (`0.0376`) with negative R2 (`-0.2250`). This does not support a clean linearly predictable xyz-MLP PCA factor as the cause of the picture failure.
- Current interpretation: task-conditioned coordinate factors are present in Concerto decoder features and are somewhat elevated on hard-to-majority errors, but they are entangled with useful layout/majority readout and do not form a clean harmful subspace. This remains a diagnostic negative result, not a positive method direction.

## Output Files

- `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_error_energy/xyz_mlp_pca_error_energy.csv`
- `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/xyz_mlp_pca_error_energy/xyz_mlp_pca_error_confusions.csv`
