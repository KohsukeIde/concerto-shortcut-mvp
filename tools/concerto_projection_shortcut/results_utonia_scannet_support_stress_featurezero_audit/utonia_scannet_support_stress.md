# Utonia ScanNet Support Stress

## Setup
- utonia weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia.pth`
- seg head weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia_linear_prob_head_sc.pth`
- data root: `data/scannet`
- val scenes: `312`
- scoring: full-scene nearest-neighbor propagation from retained support logits

## Results

| condition | keep frac | mIoU | delta | weak mean | picture | wall | counter | cabinet | sink | desk | door |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | `1.0000` | `0.7576` | `0.0000` | `0.6551` | `0.3718` | `0.8558` | `0.6590` | `0.7154` | `0.6806` | `0.6741` | `0.7439` |
| `random_keep0p2` | `0.1999` | `0.7469` | `-0.0107` | `0.6407` | `0.3332` | `0.8516` | `0.6474` | `0.7092` | `0.6606` | `0.6619` | `0.7264` |
| `structured_b1p28m_keep0p2` | `0.1940` | `0.2834` | `-0.4742` | `0.2106` | `0.1193` | `0.4981` | `0.2132` | `0.2721` | `0.1754` | `0.2453` | `0.2725` |
| `masked_model_keep0p2` | `0.2045` | `0.2360` | `-0.5216` | `0.1948` | `0.0593` | `0.4449` | `0.2336` | `0.1673` | `0.2030` | `0.1997` | `0.2293` |
| `fixed_points_4000` | `0.0342` | `0.4190` | `-0.3386` | `0.3342` | `0.0200` | `0.6533` | `0.3811` | `0.3749` | `0.3822` | `0.3425` | `0.3078` |
| `feature_zero` | `1.0000` | `0.7472` | `-0.0104` | `0.6413` | `0.3319` | `0.8442` | `0.6509` | `0.7082` | `0.6621` | `0.6618` | `0.7096` |
| `feat_zero_color_normal` | `1.0000` | `0.7475` | `-0.0101` | `0.6417` | `0.3349` | `0.8456` | `0.6488` | `0.7117` | `0.6592` | `0.6608` | `0.7114` |
| `feat_zero_coord` | `1.0000` | `0.7586` | `0.0010` | `0.6577` | `0.3866` | `0.8561` | `0.6619` | `0.7158` | `0.6770` | `0.6751` | `0.7464` |
| `raw_wo_color` | `1.0000` | `0.7557` | `-0.0019` | `0.6511` | `0.3518` | `0.8517` | `0.6514` | `0.7083` | `0.6791` | `0.6610` | `0.7344` |
| `raw_wo_normal` | `1.0000` | `0.7526` | `-0.0050` | `0.6495` | `0.3522` | `0.8513` | `0.6570` | `0.7109` | `0.6655` | `0.6752` | `0.7437` |
| `raw_wo_color_normal` | `1.0000` | `0.7477` | `-0.0099` | `0.6411` | `0.3340` | `0.8448` | `0.6487` | `0.7103` | `0.6585` | `0.6595` | `0.7099` |

## Interpretation

- This battery tests whether Utonia's cleaner fixed readout also changes the support-stress profile.
- Random/fixed/structured/object-style rows should be read as full-scene missing-support stress, not retained-subset scoring.
