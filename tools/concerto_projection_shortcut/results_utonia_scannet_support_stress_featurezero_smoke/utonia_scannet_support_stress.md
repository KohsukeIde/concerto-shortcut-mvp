# Utonia ScanNet Support Stress

## Setup
- utonia weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia.pth`
- seg head weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia_linear_prob_head_sc.pth`
- data root: `data/scannet`
- val scenes: `5`
- scoring: full-scene nearest-neighbor propagation from retained support logits

## Results

| condition | keep frac | mIoU | delta | weak mean | picture | wall | counter | cabinet | sink | desk | door |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | `1.0000` | `0.4229` | `0.0000` | `0.3916` | `0.0000` | `0.8331` | `0.7587` | `0.5112` | `0.5841` | `0.0000` | `0.8874` |
| `random_keep0p2` | `0.1999` | `0.4113` | `-0.0116` | `0.3701` | `0.0000` | `0.8281` | `0.7594` | `0.3993` | `0.5342` | `0.0000` | `0.8977` |
| `structured_b1p28m_keep0p2` | `0.2049` | `0.1649` | `-0.2580` | `0.1082` | `0.0000` | `0.4640` | `0.1813` | `0.1369` | `0.0000` | `0.0000` | `0.4392` |
| `masked_model_keep0p2` | `0.1484` | `0.1238` | `-0.2991` | `0.0512` | `0.0000` | `0.3997` | `0.1469` | `0.1035` | `0.0000` | `0.0000` | `0.1083` |
| `fixed_points_4000` | `0.0211` | `0.1915` | `-0.2314` | `0.1165` | `0.0000` | `0.6071` | `0.4837` | `0.1625` | `0.0000` | `0.0000` | `0.1691` |
| `feature_zero` | `1.0000` | `0.4080` | `-0.0149` | `0.3896` | `0.0000` | `0.8380` | `0.7150` | `0.5240` | `0.6521` | `0.0000` | `0.8359` |
| `feat_zero_color_normal` | `1.0000` | `0.4161` | `-0.0068` | `0.4024` | `0.0000` | `0.8520` | `0.7251` | `0.5614` | `0.6658` | `0.0000` | `0.8644` |
| `feat_zero_coord` | `1.0000` | `0.4178` | `-0.0051` | `0.3847` | `0.0000` | `0.8326` | `0.7575` | `0.5073` | `0.5361` | `0.0000` | `0.8918` |
| `raw_wo_color` | `1.0000` | `0.4145` | `-0.0084` | `0.3849` | `0.0000` | `0.8277` | `0.7314` | `0.5811` | `0.6426` | `0.0000` | `0.7389` |
| `raw_wo_normal` | `1.0000` | `0.4116` | `-0.0113` | `0.3872` | `0.0000` | `0.8298` | `0.7402` | `0.4928` | `0.5871` | `0.0000` | `0.8902` |
| `raw_wo_color_normal` | `1.0000` | `0.4137` | `-0.0092` | `0.3988` | `0.0000` | `0.8508` | `0.7241` | `0.5504` | `0.6685` | `0.0000` | `0.8483` |

## Interpretation

- This battery tests whether Utonia's cleaner fixed readout also changes the support-stress profile.
- Random/fixed/structured/object-style rows should be read as full-scene missing-support stress, not retained-subset scoring.
