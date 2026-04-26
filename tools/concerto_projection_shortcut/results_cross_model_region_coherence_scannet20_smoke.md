# Cross-Model Region Coherence Diagnostic

This uses labels only as an oracle diagnostic. It asks whether expert choice is spatially coherent enough that region-smoothed expert selection could plausibly recover cross-model complementarity.

## Oracle / Region Expert Choice

| variant | mIoU | allAcc | weak mIoU | picture | p->wall |
|---|---:|---:|---:|---:|---:|
| `point_oracle_all` | `0.4538` | `0.9775` | `0.4885` | `0.9864` | `0.0053` |
| `region_oracle_all_s4` | `0.4491` | `0.9742` | `0.4818` | `0.9617` | `0.0189` |
| `region_oracle_all_s8` | `0.4476` | `0.9731` | `0.4792` | `0.9518` | `0.0277` |
| `region_oracle_all_s16` | `0.4461` | `0.9718` | `0.4772` | `0.9547` | `0.0254` |
| `region_defer_oracle_s4::Concerto fullFT` | `0.4380` | `0.9657` | `0.4708` | `0.9485` | `0.0177` |
| `region_defer_oracle_s8::Concerto fullFT` | `0.4371` | `0.9651` | `0.4696` | `0.9411` | `0.0213` |
| `region_defer_oracle_s16::Concerto fullFT` | `0.4362` | `0.9641` | `0.4688` | `0.9425` | `0.0171` |
| `region_defer_oracle_s4::PTv3_supervised` | `0.4291` | `0.9669` | `0.4338` | `0.7329` | `0.0407` |
| `region_defer_oracle_s8::PTv3_supervised` | `0.4284` | `0.9662` | `0.4332` | `0.7318` | `0.0354` |
| `region_defer_oracle_s16::PTv3_supervised` | `0.4270` | `0.9654` | `0.4312` | `0.7305` | `0.0390` |
| `region_defer_oracle_s4::Utonia` | `0.4172` | `0.9591` | `0.4180` | `0.7325` | `0.0360` |
| `region_defer_oracle_s8::Utonia` | `0.4156` | `0.9578` | `0.4164` | `0.7301` | `0.0354` |
| `region_defer_oracle_s16::Utonia` | `0.4138` | `0.9562` | `0.4143` | `0.7282` | `0.0331` |
| `region_defer_oracle_s4::Sonata linear` | `0.4127` | `0.9554` | `0.4161` | `0.7329` | `0.0360` |
| `region_defer_oracle_s8::Sonata linear` | `0.4115` | `0.9544` | `0.4154` | `0.7318` | `0.0384` |
| `region_defer_oracle_s16::Sonata linear` | `0.4099` | `0.9533` | `0.4131` | `0.7300` | `0.0390` |
| `single::Concerto decoder` | `0.3884` | `0.9379` | `0.3822` | `0.7170` | `0.0301` |

## Target Expert Region Purity

| region size | weighted purity | points in purity>=0.8 regions |
|---:|---:|---:|
| `4` | `0.9135` | `0.8086` |
| `8` | `0.8770` | `0.7265` |
| `16` | `0.8380` | `0.6651` |
