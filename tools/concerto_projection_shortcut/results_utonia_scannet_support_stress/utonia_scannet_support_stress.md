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
| `clean` | `1.0000` | `0.7586` | `0.0000` | `0.6572` | `0.3796` | `0.8562` | `0.6562` | `0.7150` | `0.6809` | `0.6755` | `0.7465` |
| `random_keep0p2` | `0.1999` | `0.7464` | `-0.0122` | `0.6390` | `0.3199` | `0.8507` | `0.6451` | `0.7084` | `0.6657` | `0.6611` | `0.7237` |
| `structured_b1p28m_keep0p2` | `0.1940` | `0.2837` | `-0.4748` | `0.2122` | `0.1188` | `0.4976` | `0.2116` | `0.2729` | `0.1799` | `0.2445` | `0.2741` |
| `masked_model_keep0p2` | `0.2045` | `0.2340` | `-0.5246` | `0.1910` | `0.0544` | `0.4435` | `0.2246` | `0.1665` | `0.2032` | `0.1967` | `0.2279` |
| `fixed_points_4000` | `0.0342` | `0.4163` | `-0.3423` | `0.3339` | `0.0211` | `0.6542` | `0.3891` | `0.3771` | `0.3775` | `0.3417` | `0.3096` |
| `feature_zero` | `1.0000` | `0.7466` | `-0.0120` | `0.6402` | `0.3250` | `0.8434` | `0.6528` | `0.7052` | `0.6614` | `0.6602` | `0.7091` |

## Interpretation

- This battery tests whether Utonia's cleaner fixed readout also changes the support-stress profile.
- Random/fixed/structured/object-style rows should be read as full-scene missing-support stress, not retained-subset scoring.
