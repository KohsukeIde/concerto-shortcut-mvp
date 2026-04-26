# Six-Dataset Coord-MLP Rival Calibration

This table recalibrates the existing coordinate-only rival against the completed six-dataset main-variant causal battery.

Definitions:
- `relative_position = (L_coord - L_clean) / (L_corrupt - L_clean)`.
- `closure_fraction = 1 - relative_position`; higher means the coordinate-only rival is closer to the clean reference than the corrupted reference.
- The main summary uses the mean of global permutation, cross-image target swap, and cross-scene target swap as `L_corrupt`.

| dataset | clean | coord MLP | mean corrupt | rel. position | closure | global closure | cross-image closure | cross-scene closure | hint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arkit` | `5.8340` | `6.6960` | `6.8493` | `0.8490` | `15.1%` | `7.9%` | `17.3%` | `19.3%` | `weak` |
| `scannet` | `5.6468` | `6.7353` | `7.3048` | `0.6565` | `34.4%` | `34.0%` | `26.1%` | `41.2%` | `partial` |
| `scannetpp` | `5.0148` | `6.3425` | `7.2798` | `0.5862` | `41.4%` | `39.4%` | `39.6%` | `44.8%` | `partial` |
| `s3dis` | `5.8761` | `6.3956` | `6.1764` | `1.7295` | `-73.0%` | `-74.8%` | `-33.8%` | `-140.9%` | `worse_than_mean_corruption` |
| `hm3d` | `5.3425` | `6.3728` | `7.2349` | `0.5444` | `45.6%` | `40.2%` | `46.4%` | `49.4%` | `partial` |
| `structured3d` | `4.9125` | `6.3822` | `6.8330` | `0.7653` | `23.5%` | `19.8%` | `18.6%` | `30.7%` | `weak` |

## Aggregate

- Mean closure against mean corruption: `14.5%`.
- Min / max closure: `-73.0%` / `45.6%`.
- Positive-closure datasets: `5/6`.

## Interpretation

- The six-dataset target-corruption sensitivity is positive, but the coordinate-only rival is not uniformly strong.
- The coordinate-only rival closes a nonzero fraction of the clean-to-corrupted gap on most datasets, but it is weak on ARKit and worse than the mean corruption reference on S3DIS.
- Therefore, the safe paper claim is not that coordinate-only explains the six-dataset average. The safe claim is that the objective has a coordinate-satisfiable component whose strength is dataset-dependent.
