# Main-Variant Six-Dataset Causal Battery: Paper Table

This table reformats the completed main-variant head-refit causal battery into a single paper-facing row per dataset.

| dataset | baseline loss | global perm delta | cross-image delta | cross-scene delta | max delta | relative B_pre |
|---|---:|---:|---:|---:|---:|---:|
| `arkit` | `5.8340` | `0.9358` | `1.0420` | `1.0680` | `1.0680` | `0.1831` |
| `scannet` | `5.6468` | `1.6490` | `1.4724` | `1.8526` | `1.8526` | `0.3281` |
| `scannetpp` | `5.0148` | `2.1901` | `2.1987` | `2.4061` | `2.4061` | `0.4798` |
| `s3dis` | `5.8761` | `0.2972` | `0.3881` | `0.2157` | `0.3881` | `0.0661` |
| `hm3d` | `5.3425` | `1.7223` | `1.9208` | `2.0342` | `2.0342` | `0.3808` |
| `structured3d` | `4.9125` | `1.8330` | `1.8061` | `2.1224` | `2.1224` | `0.4320` |

## Interpretation

- The target-swap sensitivity is positive on all six indoor datasets.
- Magnitude is dataset-dependent: `s3dis` is much weaker than ScanNet++ / HM3D / Structured3D, which should be reported rather than hidden.
- Use this as the main scene-level train-side counterfactual table.
