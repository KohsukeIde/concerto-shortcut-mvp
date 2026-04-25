# Cross-Model CV Stacker Pilot

Two-fold scene-level validation pilot. This intentionally uses ScanNet val labels to test whether a learned logit/probability stacker can extract the cross-model oracle headroom; it is not a final train-split baseline.

- models: Concerto decoder, Sonata linear, Utonia
- sampled train points per fold: [3072, 3072]
- sample points per scene: `1024`
- epochs: `2`

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `single::Concerto decoder` | `0.5460` | `0.9205` | `0.5240` | `0.6156` | `0.0236` | `0.7405` | `0.4509` | `0.8011` |
| 2 | `avgprob_all` | `0.5380` | `0.9262` | `0.4756` | `0.0460` | `0.8530` | `0.7773` | `0.4778` | `0.8092` |
| 3 | `cv_linear_stacker` | `0.2411` | `0.8277` | `0.1040` | `0.0000` | `0.9970` | `0.0000` | `0.0224` | `0.7058` |

## Interpretation Gate

- If this CV stacker is clearly above simple averaging, learned fusion is worth promoting to a train-split method.
- If it is near or below simple averaging, the high oracle complementarity is not easily actionable even with a learned logit stacker.
