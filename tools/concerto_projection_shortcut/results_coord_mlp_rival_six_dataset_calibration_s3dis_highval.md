# Six-Dataset Coord-MLP Rival Calibration with S3DIS High-Val Follow-Up

This table keeps the same six-dataset causal references as the canonical calibration, but replaces the original tiny-cache S3DIS coord-MLP loss with the S3DIS-only high-validation follow-up.

| dataset | clean | coord MLP | mean corrupt | rel. position | closure | global closure | cross-image closure | cross-scene closure | hint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arkit` | `5.8340` | `6.6960` | `6.8493` | `0.8490` | `15.1%` | `7.9%` | `17.3%` | `19.3%` | `weak` |
| `scannet` | `5.6468` | `6.7353` | `7.3048` | `0.6565` | `34.4%` | `34.0%` | `26.1%` | `41.2%` | `partial` |
| `scannetpp` | `5.0148` | `6.3425` | `7.2798` | `0.5862` | `41.4%` | `39.4%` | `39.6%` | `44.8%` | `partial` |
| `s3dis` | `5.8761` | `6.1365` | `6.1764` | `0.8668` | `13.3%` | `12.4%` | `32.9%` | `-20.7%` | `weak_highval` |
| `hm3d` | `5.3425` | `6.3728` | `7.2349` | `0.5444` | `45.6%` | `40.2%` | `46.4%` | `49.4%` | `partial` |
| `structured3d` | `4.9125` | `6.3822` | `6.8330` | `0.7653` | `23.5%` | `19.8%` | `18.6%` | `30.7%` | `weak` |

## Aggregate

- Mean closure against mean corruption: `28.9%`.
- Min / max closure: `13.3%` / `45.6%`.
- Positive-closure datasets: `6/6`.

## Interpretation

- The original S3DIS negative closure was mainly a small validation-cache artifact.
- With the high-val S3DIS follow-up, all six datasets have positive closure, but S3DIS remains weak at `13.3%`.
- The safe claim is still dataset-dependent coordinate-satisfiable signal, not a uniformly strong coordinate-only explanation.
