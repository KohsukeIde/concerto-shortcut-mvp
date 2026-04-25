# Cross-Model CV Stacker Pilot

Two-fold scene-level validation pilot. This intentionally uses ScanNet val labels to test whether a learned logit/probability stacker can extract the cross-model oracle headroom; it is not a final train-split baseline.

- models: Concerto decoder, Sonata linear, Utonia, PTv3_supervised
- sampled train points per fold: [3072, 3072]
- sample points per scene: `1024`
- epochs: `2`
- class weight powers: `0.0,0.5`

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `single::Concerto decoder` | `0.5481` | `0.9207` | `0.5288` | `0.6667` | `0.0266` | `0.7305` | `0.4579` | `0.7842` |
| 2 | `cv_oracle_router_soft` | `0.5408` | `0.9254` | `0.4546` | `0.0000` | `0.9923` | `0.8168` | `0.3539` | `0.8225` |
| 3 | `avgprob_all` | `0.5369` | `0.9274` | `0.4605` | `0.0000` | `0.9935` | `0.8028` | `0.3911` | `0.8464` |
| 4 | `cv_oracle_router_hard` | `0.4953` | `0.9102` | `0.3998` | `0.0000` | `0.9917` | `0.5891` | `0.3017` | `0.7015` |
| 5 | `cv_linear_stacker_w0.5` | `0.2393` | `0.8274` | `0.0996` | `0.0000` | `0.9941` | `0.0000` | `0.0898` | `0.6076` |
| 6 | `cv_linear_stacker_w0` | `0.1548` | `0.6954` | `0.1135` | `0.0000` | `0.9965` | `0.0000` | `0.0300` | `0.7642` |

## Interpretation Gate

- If this CV stacker is clearly above simple averaging, learned fusion is worth promoting to a train-split method.
- If it is near or below simple averaging, the high oracle complementarity is not easily actionable even with a learned logit stacker.
