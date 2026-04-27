# Recovery Positive Controls

Synthetic class-prior/logit-bias sanity check over cached PTv3 raw probabilities.
We induce a known class-prior/logit-bias direction, then evaluate whether class-prior correction variants can recover the induced error.

| variant | mIoU | delta vs biased | fraction to clean | picture | picture delta | picture fraction to clean |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | `0.75736531` | `0.32327169` | `1.00000000` | `0.38242643` | `0.10026437` | `1.00000000` |
| `biased` | `0.43409362` | `0.00000000` | `0.00000000` | `0.28216206` | `0.00000000` | `0.00000000` |
| `learned_bias` | `0.40731109` | `-0.02678253` | `-0.08284835` | `0.13643666` | `-0.14572540` | `-1.45341162` |
| `balanced_bias` | `0.41541941` | `-0.01867421` | `-0.05776629` | `0.03015715` | `-0.25200491` | `-2.51340445` |
| `known_direction_recover_alpha0` | `0.43409362` | `0.00000000` | `0.00000000` | `0.28216206` | `0.00000000` | `0.00000000` |
| `known_direction_recover_alpha0p25` | `0.56240093` | `0.12830731` | `0.39690239` | `0.33990942` | `0.05774736` | `0.57595100` |
| `known_direction_recover_alpha0p5` | `0.70364970` | `0.26955609` | `0.83383757` | `0.36074008` | `0.07857802` | `0.78370829` |
| `known_direction_recover_alpha0p75` | `0.74814654` | `0.31405293` | `0.97148291` | `0.37096108` | `0.08879902` | `0.88564880` |
| `known_direction_recover_alpha1` | `0.75736531` | `0.32327169` | `1.00000000` | `0.38242643` | `0.10026437` | `1.00000000` |
| `known_direction_recover_alpha1p25` | `0.75446835` | `0.32037474` | `0.99103863` | `0.38444493` | `0.10228287` | `1.02013182` |
| `known_direction_recover_alpha1p5` | `0.72647001` | `0.29237639` | `0.90442930` | `0.37757932` | `0.09541726` | `0.95165673` |
| `biased_oracle_top2` | `0.55903068` | `0.12493707` | `0.38647698` | `0.55140052` | `0.26923846` | `2.68528553` |
