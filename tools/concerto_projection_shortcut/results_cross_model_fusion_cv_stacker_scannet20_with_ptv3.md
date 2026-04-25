# Cross-Model CV Stacker Pilot

Two-fold scene-level validation pilot. This intentionally uses ScanNet val labels to test whether a learned logit/probability stacker can extract the cross-model oracle headroom; it is not a final train-split baseline.

- models: Concerto decoder, Sonata linear, Utonia, PTv3_supervised
- sampled train points per fold: [638976, 638976]
- sample points per scene: `4096`
- epochs: `60`
- class weight powers: `0.0,0.5`

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `cv_linear_stacker_w0` | `0.7971` | `0.9283` | `0.6988` | `0.3525` | `0.5398` | `0.7053` | `0.7311` | `0.7880` |
| 2 | `cv_linear_stacker_w0.5` | `0.7961` | `0.9277` | `0.6949` | `0.3912` | `0.3691` | `0.7006` | `0.7310` | `0.7838` |
| 3 | `avgprob_all` | `0.7961` | `0.9278` | `0.6978` | `0.3996` | `0.5029` | `0.7198` | `0.7500` | `0.7757` |
| 4 | `single::Concerto decoder` | `0.7782` | `0.9198` | `0.6815` | `0.4054` | `0.4394` | `0.6888` | `0.7218` | `0.7599` |

## Interpretation Gate

- If this CV stacker is clearly above simple averaging, learned fusion is worth promoting to a train-split method.
- If it is near or below simple averaging, the high oracle complementarity is not easily actionable even with a learned logit stacker.
