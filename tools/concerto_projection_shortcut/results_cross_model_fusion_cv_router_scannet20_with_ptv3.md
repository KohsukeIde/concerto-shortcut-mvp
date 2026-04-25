# Cross-Model CV Stacker Pilot

Two-fold scene-level validation pilot. This intentionally uses ScanNet val labels to test whether a learned logit/probability stacker can extract the cross-model oracle headroom; it is not a final train-split baseline.

- models: Concerto decoder, Sonata linear, Utonia, PTv3_supervised
- sampled train points per fold: [638976, 638976]
- sample points per scene: `4096`
- epochs: `60`
- class weight powers: `0.0,0.5`

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `cv_linear_stacker_w0` | `0.7975` | `0.9284` | `0.6984` | `0.3558` | `0.5371` | `0.7045` | `0.7307` | `0.7875` |
| 2 | `cv_linear_stacker_w0.5` | `0.7962` | `0.9277` | `0.6957` | `0.3965` | `0.3669` | `0.6998` | `0.7305` | `0.7831` |
| 3 | `avgprob_all` | `0.7959` | `0.9278` | `0.6985` | `0.4032` | `0.4993` | `0.7191` | `0.7503` | `0.7772` |
| 4 | `cv_oracle_router_soft` | `0.7812` | `0.9221` | `0.6772` | `0.3890` | `0.5022` | `0.7042` | `0.7111` | `0.7404` |
| 5 | `single::Concerto decoder` | `0.7788` | `0.9201` | `0.6814` | `0.4105` | `0.4324` | `0.6872` | `0.7233` | `0.7581` |
| 6 | `cv_oracle_router_hard` | `0.7676` | `0.9172` | `0.6621` | `0.3842` | `0.4980` | `0.6945` | `0.6889` | `0.7251` |

## Interpretation Gate

- If this CV stacker is clearly above simple averaging, learned fusion is worth promoting to a train-split method.
- If it is near or below simple averaging, the high oracle complementarity is not easily actionable even with a learned logit stacker.
