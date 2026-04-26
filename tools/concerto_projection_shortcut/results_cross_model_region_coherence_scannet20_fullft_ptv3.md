# Cross-Model Region Coherence Diagnostic

This uses labels only as an oracle diagnostic. It asks whether expert choice is spatially coherent enough that region-smoothed expert selection could plausibly recover cross-model complementarity.

## Oracle / Region Expert Choice

| variant | mIoU | allAcc | weak mIoU | picture | p->wall |
|---|---:|---:|---:|---:|---:|
| `point_oracle_all` | `0.9006` | `0.9687` | `0.8414` | `0.6427` | `0.2839` |
| `region_oracle_all_s4` | `0.8893` | `0.9643` | `0.8263` | `0.6197` | `0.2977` |
| `region_oracle_all_s8` | `0.8846` | `0.9626` | `0.8196` | `0.6091` | `0.3031` |
| `region_oracle_all_s16` | `0.8789` | `0.9606` | `0.8121` | `0.5988` | `0.3080` |
| `region_defer_oracle_s4::Utonia` | `0.8497` | `0.9487` | `0.7732` | `0.5542` | `0.3262` |
| `region_defer_oracle_s4::Concerto decoder` | `0.8479` | `0.9479` | `0.7693` | `0.5242` | `0.3387` |
| `region_defer_oracle_s8::Utonia` | `0.8464` | `0.9474` | `0.7686` | `0.5478` | `0.3298` |
| `region_defer_oracle_s8::Concerto decoder` | `0.8453` | `0.9470` | `0.7665` | `0.5198` | `0.3417` |
| `region_defer_oracle_s16::Utonia` | `0.8428` | `0.9459` | `0.7630` | `0.5406` | `0.3321` |
| `region_defer_oracle_s16::Concerto decoder` | `0.8416` | `0.9459` | `0.7622` | `0.5140` | `0.3439` |
| `region_defer_oracle_s4::Sonata linear` | `0.8409` | `0.9468` | `0.7601` | `0.4962` | `0.3835` |
| `region_defer_oracle_s4::PTv3_supervised` | `0.8383` | `0.9458` | `0.7564` | `0.5089` | `0.3770` |
| `region_defer_oracle_s8::Sonata linear` | `0.8379` | `0.9455` | `0.7558` | `0.4906` | `0.3866` |
| `region_defer_oracle_s8::PTv3_supervised` | `0.8368` | `0.9451` | `0.7542` | `0.5056` | `0.3781` |
| `region_defer_oracle_s16::PTv3_supervised` | `0.8346` | `0.9442` | `0.7510` | `0.5008` | `0.3801` |
| `region_defer_oracle_s16::Sonata linear` | `0.8341` | `0.9440` | `0.7508` | `0.4838` | `0.3887` |
| `single::Concerto fullFT` | `0.7969` | `0.9276` | `0.7014` | `0.4296` | `0.4015` |

## Target Expert Region Purity

| region size | weighted purity | points in purity>=0.8 regions |
|---:|---:|---:|
| `4` | `0.9104` | `0.8018` |
| `8` | `0.8722` | `0.7217` |
| `16` | `0.8304` | `0.6309` |
