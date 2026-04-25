# Cross-Model CV Stacker Pilot

Two-fold scene-level validation pilot. This intentionally uses ScanNet val labels to test whether a learned logit/probability stacker can extract the cross-model oracle headroom; it is not a final train-split baseline.

- models: Concerto decoder, Sonata linear, Utonia
- sampled train points per fold: [638976, 638976]
- sample points per scene: `4096`
- epochs: `60`
- class weight powers: `0.0,0.5`

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `cv_linear_stacker_w0.5` | `0.7910` | `0.9237` | `0.6903` | `0.3836` | `0.3714` | `0.6897` | `0.7387` | `0.7729` |
| 2 | `cv_linear_stacker_w0` | `0.7883` | `0.9239` | `0.6830` | `0.3468` | `0.5481` | `0.6923` | `0.7316` | `0.7734` |
| 3 | `avgprob_all` | `0.7845` | `0.9230` | `0.6853` | `0.4039` | `0.4746` | `0.7013` | `0.7374` | `0.7678` |
| 4 | `single::Concerto decoder` | `0.7796` | `0.9203` | `0.6821` | `0.4092` | `0.4399` | `0.6912` | `0.7198` | `0.7593` |

## Interpretation Gate

- If this CV stacker is clearly above simple averaging, learned fusion is worth promoting to a train-split method.
- If it is near or below simple averaging, the high oracle complementarity is not easily actionable even with a learned logit stacker.
