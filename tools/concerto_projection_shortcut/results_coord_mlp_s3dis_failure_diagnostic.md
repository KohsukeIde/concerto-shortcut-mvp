# S3DIS Coord-MLP Rival Failure Diagnostic

This diagnostic checks whether the negative S3DIS coordinate-rival closure is a genuine no-coordinate signal or an evaluation/cache artifact.

| dataset | train rows | val rows | coord mean shift | target mean cosine | shared MLP train | shared MLP val | train-mean val | clean | mean corrupt | closure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arkit` | `49513` | `15239` | `0.162608` | `0.964833` | `6.614874` | `6.695971` | `6.508246` | `5.834015` | `6.849298` | `15.1%` |
| `scannet` | `101996` | `25986` | `0.109071` | `0.975024` | `6.556741` | `6.735274` | `6.642395` | `5.646850` | `7.304839` | `34.4%` |
| `scannetpp` | `108970` | `28452` | `0.123736` | `0.937122` | `6.290183` | `6.342475` | `6.367269` | `5.014833` | `7.279796` | `41.4%` |
| `s3dis` | `61527` | `449` | `0.448939` | `0.853645` | `6.285331` | `6.395550` | `6.209538` | `5.876104` | `6.176442` | `-73.0%` |
| `hm3d` | `108345` | `26830` | `0.043729` | `0.994620` | `6.377903` | `6.372810` | `6.227578` | `5.342510` | `7.234931` | `45.6%` |
| `structured3d` | `104971` | `26343` | `0.140104` | `0.980486` | `6.511524` | `6.382244` | `6.139919` | `4.912487` | `6.832953` | `23.5%` |

## S3DIS-only coord MLP

| epoch | train loss | val loss |
|---:|---:|---:|
| `1` | `7.927153` | `7.092888` |
| `5` | `6.098450` | `6.340835` |
| `10` | `5.967692` | `6.197619` |
| `80` | `5.885549` | `6.234728` |

- Final S3DIS-only train loss: `5.870842`.
- Final S3DIS-only val loss: `6.234728`.
- S3DIS-only closure against mean corruption: `-19.4%`.

## Interpretation

- The S3DIS coord-rival validation cache is extremely small compared with the other five datasets.
- The S3DIS clean-to-corruption gap is also small, so normalized closure is unstable and can become strongly negative from modest absolute loss differences.
- If the S3DIS-only MLP improves train loss but not validation loss, the safest reading is train/val or sample-selection mismatch rather than a clean six-dataset coordinate-only closure.
