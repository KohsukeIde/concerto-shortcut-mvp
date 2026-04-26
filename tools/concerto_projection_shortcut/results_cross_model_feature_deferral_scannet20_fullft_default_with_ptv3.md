# Feature-Level Selective Deferral Diagnostic

Default expert is Concerto full-FT. `logit` uses probability/confidence features only; `feature` adds fixed random projections of raw point features from the default and auxiliary experts.

This is a two-fold scene-level validation diagnostic, not a publishable train-split method result.

## Deferral Predictability

| mode | expert | fold | pos rate | PR-AUC | R@P80 | R@P90 | R@P95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `logit` | `Concerto decoder` | `0` | `0.0201` | `0.4778` | `0.0002` | `0.0002` | `0.0002` |
| `feature` | `Concerto decoder` | `0` | `0.0201` | `0.4685` | `0.0042` | `0.0033` | `0.0023` |
| `logit` | `Sonata linear` | `0` | `0.0214` | `0.3643` | `0.0006` | `0.0006` | `0.0006` |
| `feature` | `Sonata linear` | `0` | `0.0214` | `0.3537` | `0.0041` | `0.0023` | `0.0012` |
| `logit` | `Utonia` | `0` | `0.0227` | `0.4447` | `0.0029` | `0.0015` | `0.0008` |
| `feature` | `Utonia` | `0` | `0.0227` | `0.4335` | `0.0040` | `0.0010` | `0.0010` |
| `logit` | `PTv3_supervised` | `0` | `0.0182` | `0.4401` | `0.0000` | `0.0000` | `0.0000` |
| `logit` | `Concerto decoder` | `1` | `0.0218` | `0.4591` | `0.0000` | `0.0000` | `0.0000` |
| `feature` | `Concerto decoder` | `1` | `0.0218` | `0.4733` | `0.0000` | `0.0000` | `0.0000` |
| `logit` | `Sonata linear` | `1` | `0.0200` | `0.3527` | `0.0014` | `0.0014` | `0.0000` |
| `feature` | `Sonata linear` | `1` | `0.0200` | `0.3453` | `0.0013` | `0.0002` | `0.0002` |
| `logit` | `Utonia` | `1` | `0.0235` | `0.4457` | `0.0000` | `0.0000` | `0.0000` |
| `feature` | `Utonia` | `1` | `0.0235` | `0.4494` | `0.0001` | `0.0001` | `0.0001` |
| `logit` | `PTv3_supervised` | `1` | `0.0201` | `0.4108` | `0.0000` | `0.0000` | `0.0000` |

## Sample Conservative Router

| variant | sample mIoU | allAcc | picture | p->wall |
|---|---:|---:|---:|---:|
| `fold0_default::Concerto fullFT` | `0.8117` | `0.9321` | `0.4371` | `0.4048` |
| `fold0_logit_defer_p80` | `0.8117` | `0.9321` | `0.4371` | `0.4048` |
| `fold0_logit_defer_p90` | `0.8117` | `0.9321` | `0.4371` | `0.4048` |
| `fold0_logit_defer_p95` | `0.8117` | `0.9321` | `0.4371` | `0.4048` |
| `fold0_feature_defer_p80` | `0.8123` | `0.9323` | `0.4380` | `0.4048` |
| `fold0_feature_defer_p90` | `0.8120` | `0.9322` | `0.4378` | `0.4048` |
| `fold0_feature_defer_p95` | `0.8118` | `0.9321` | `0.4373` | `0.4048` |
| `fold1_default::Concerto fullFT` | `0.7926` | `0.9263` | `0.4951` | `0.3190` |
| `fold1_logit_defer_p80` | `0.7931` | `0.9264` | `0.4975` | `0.3190` |
| `fold1_logit_defer_p90` | `0.7926` | `0.9263` | `0.4951` | `0.3190` |
| `fold1_logit_defer_p95` | `0.7926` | `0.9263` | `0.4951` | `0.3190` |
| `fold1_feature_defer_p80` | `0.7939` | `0.9266` | `0.4942` | `0.3226` |
| `fold1_feature_defer_p90` | `0.7929` | `0.9263` | `0.4956` | `0.3190` |
| `fold1_feature_defer_p95` | `0.7929` | `0.9263` | `0.4956` | `0.3190` |
