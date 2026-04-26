# Cross-Model Region Coherence Diagnostic

This uses labels only as an oracle diagnostic. It asks whether expert choice is spatially coherent enough that region-smoothed expert selection could plausibly recover cross-model complementarity.

## Oracle / Region Expert Choice

| variant | mIoU | allAcc | weak mIoU | picture | p->wall |
|---|---:|---:|---:|---:|---:|
| `point_oracle_all` | `0.9004` | `0.9686` | `0.8410` | `0.6395` | `0.2877` |
| `region_oracle_all_s4` | `0.8890` | `0.9642` | `0.8259` | `0.6166` | `0.3023` |
| `region_oracle_all_s8` | `0.8842` | `0.9625` | `0.8191` | `0.6066` | `0.3067` |
| `region_oracle_all_s16` | `0.8785` | `0.9604` | `0.8113` | `0.5957` | `0.3128` |
| `region_defer_oracle_s4::Utonia` | `0.8496` | `0.9487` | `0.7725` | `0.5481` | `0.3324` |
| `region_defer_oracle_s4::Concerto decoder` | `0.8478` | `0.9478` | `0.7687` | `0.5235` | `0.3409` |
| `region_defer_oracle_s8::Utonia` | `0.8464` | `0.9474` | `0.7679` | `0.5421` | `0.3357` |
| `region_defer_oracle_s8::Concerto decoder` | `0.8451` | `0.9469` | `0.7659` | `0.5200` | `0.3437` |
| `region_defer_oracle_s16::Utonia` | `0.8425` | `0.9459` | `0.7621` | `0.5348` | `0.3380` |
| `region_defer_oracle_s16::Concerto decoder` | `0.8414` | `0.9458` | `0.7615` | `0.5148` | `0.3458` |
| `region_defer_oracle_s4::Sonata linear` | `0.8409` | `0.9467` | `0.7601` | `0.4970` | `0.3832` |
| `region_defer_oracle_s4::PTv3_supervised` | `0.8383` | `0.9458` | `0.7564` | `0.5089` | `0.3770` |
| `region_defer_oracle_s8::Sonata linear` | `0.8378` | `0.9454` | `0.7558` | `0.4914` | `0.3856` |
| `region_defer_oracle_s8::PTv3_supervised` | `0.8368` | `0.9451` | `0.7542` | `0.5056` | `0.3781` |
| `region_defer_oracle_s16::PTv3_supervised` | `0.8346` | `0.9442` | `0.7510` | `0.5008` | `0.3801` |
| `region_defer_oracle_s16::Sonata linear` | `0.8339` | `0.9439` | `0.7509` | `0.4848` | `0.3882` |
| `single::Concerto fullFT` | `0.7969` | `0.9276` | `0.7014` | `0.4296` | `0.4015` |

## Target Expert Region Purity

| region size | weighted purity | points in purity>=0.8 regions |
|---:|---:|---:|
| `4` | `0.9104` | `0.8019` |
| `8` | `0.8724` | `0.7222` |
| `16` | `0.8305` | `0.6311` |
