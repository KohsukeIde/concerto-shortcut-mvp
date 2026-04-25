# Cross-Model CV Stacker Pilot

Two-fold scene-level validation pilot. This intentionally uses ScanNet val labels to test whether a learned logit/probability stacker can extract the cross-model oracle headroom; it is not a final train-split baseline.

- models: Concerto decoder, Sonata linear, Utonia, Concerto_fullFT, PTv3_supervised
- sampled train points per fold: [638976, 638976]
- sample points per scene: `4096`
- epochs: `60`
- class weight powers: `0.0,0.5`

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `avgprob_all` | `0.8064` | `0.9321` | `0.7114` | `0.4221` | `0.4645` | `0.7218` | `0.7713` | `0.7909` |
| 2 | `cv_linear_stacker_w0.5` | `0.8056` | `0.9302` | `0.7121` | `0.4140` | `0.3598` | `0.7109` | `0.7460` | `0.7962` |
| 3 | `cv_linear_stacker_w0` | `0.8051` | `0.9308` | `0.7083` | `0.3747` | `0.5288` | `0.7090` | `0.7432` | `0.7997` |
| 4 | `cv_oracle_router_soft` | `0.8015` | `0.9304` | `0.7054` | `0.4086` | `0.4574` | `0.7093` | `0.7771` | `0.7798` |
| 5 | `cv_oracle_router_hard` | `0.7924` | `0.9268` | `0.6923` | `0.3947` | `0.4650` | `0.6996` | `0.7710` | `0.7681` |
| 6 | `single::Concerto decoder` | `0.7786` | `0.9200` | `0.6818` | `0.4067` | `0.4381` | `0.6882` | `0.7216` | `0.7615` |

## Interpretation Gate

- If this CV stacker is clearly above simple averaging, learned fusion is worth promoting to a train-split method.
- If it is near or below simple averaging, the high oracle complementarity is not easily actionable even with a learned logit stacker.
