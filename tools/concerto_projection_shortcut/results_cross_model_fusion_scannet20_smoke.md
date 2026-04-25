# Cross-Model Complementarity / Fusion Audit

Raw-point aligned ScanNet20 audit. Current-Pointcept models use the validation `inverse` mapping to original scene points; Utonia uses its released inverse-restored raw logits.

## Results

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle::Concerto decoder+Sonata linear+Utonia` | `0.4517` | `0.9544` | `0.4103` | `0.7031` | `0.0248` | `0.0000` | `0.8019` | `0.8824` |
| 2 | `oracle::Concerto decoder+Sonata linear` | `0.4420` | `0.9475` | `0.4000` | `0.7006` | `0.0248` | `0.0000` | `0.7819` | `0.8749` |
| 3 | `oracle::Concerto decoder+Utonia` | `0.3988` | `0.9187` | `0.3297` | `0.7031` | `0.0248` | `0.0000` | `0.2583` | `0.8739` |
| 4 | `oracle::Sonata linear+Utonia` | `0.3965` | `0.9271` | `0.2834` | `0.0000` | `0.5909` | `0.0000` | `0.6998` | `0.8428` |
| 5 | `avgprobT2::Concerto decoder+Sonata linear` | `0.3753` | `0.8997` | `0.2947` | `0.6175` | `0.1948` | `0.0000` | `0.3047` | `0.7898` |
| 6 | `avgprob::Concerto decoder+Sonata linear` | `0.3718` | `0.8974` | `0.2877` | `0.5913` | `0.1482` | `0.0000` | `0.2853` | `0.7875` |
| 7 | `maxmargin::Concerto decoder+Sonata linear` | `0.3692` | `0.8955` | `0.2826` | `0.5677` | `0.1293` | `0.0000` | `0.2726` | `0.7899` |
| 8 | `maxconf::Concerto decoder+Sonata linear` | `0.3692` | `0.8953` | `0.2817` | `0.5695` | `0.1399` | `0.0000` | `0.2646` | `0.7910` |
| 9 | `maxconf::Concerto decoder+Sonata linear+Utonia` | `0.3630` | `0.8966` | `0.2507` | `0.3333` | `0.5065` | `0.0000` | `0.2148` | `0.7902` |
| 10 | `single::Concerto decoder` | `0.3601` | `0.8871` | `0.2868` | `0.6710` | `0.0248` | `0.0000` | `0.2270` | `0.7621` |
| 11 | `avgprob::Concerto decoder+Utonia` | `0.3565` | `0.8909` | `0.2480` | `0.3439` | `0.5224` | `0.0000` | `0.1845` | `0.7754` |
| 12 | `maxmargin::Concerto decoder+Utonia` | `0.3548` | `0.8895` | `0.2444` | `0.3312` | `0.5431` | `0.0000` | `0.1771` | `0.7762` |
| 13 | `maxconf::Concerto decoder+Utonia` | `0.3544` | `0.8888` | `0.2446` | `0.3429` | `0.5272` | `0.0000` | `0.1751` | `0.7760` |
| 14 | `avgprobT2::Concerto decoder+Utonia` | `0.3543` | `0.8896` | `0.2423` | `0.3090` | `0.5667` | `0.0000` | `0.1844` | `0.7750` |
| 15 | `avgprob::Concerto decoder+Sonata linear+Utonia` | `0.3514` | `0.8942` | `0.2120` | `0.0384` | `0.8595` | `0.0000` | `0.1711` | `0.8437` |
| 16 | `avgprob::Sonata linear+Utonia` | `0.3513` | `0.8907` | `0.2144` | `0.0000` | `0.9250` | `0.0000` | `0.2831` | `0.8234` |
| 17 | `maxmargin::Sonata linear+Utonia` | `0.3509` | `0.8902` | `0.2149` | `0.0000` | `0.9174` | `0.0000` | `0.2842` | `0.8223` |
| 18 | `maxconf::Sonata linear+Utonia` | `0.3500` | `0.8894` | `0.2135` | `0.0000` | `0.9067` | `0.0000` | `0.2696` | `0.8219` |
| 19 | `avgprobT2::Sonata linear+Utonia` | `0.3497` | `0.8894` | `0.2099` | `0.0000` | `0.9238` | `0.0000` | `0.2506` | `0.8244` |
| 20 | `single::Sonata linear` | `0.3427` | `0.8748` | `0.2339` | `0.0000` | `0.5909` | `0.0000` | `0.5912` | `0.8008` |
| 21 | `single::Utonia` | `0.3366` | `0.8738` | `0.1931` | `0.0000` | `0.9900` | `0.0000` | `0.1160` | `0.8211` |
| 22 | `weakgate::Concerto decoder->Utonia` | `0.3303` | `0.8767` | `0.1931` | `0.0000` | `0.7273` | `0.0000` | `0.1160` | `0.8211` |
| 23 | `picturewall_top2_gate::Concerto decoder->Utonia` | `0.3246` | `0.8668` | `0.1861` | `0.0000` | `0.9900` | `0.0000` | `0.1603` | `0.7987` |

## Pairwise Complementarity

| pair | both correct | first only | second only | both wrong | oracle correct frac |
|---|---:|---:|---:|---:|---:|
| `Concerto decoder+Sonata linear` | `0.4231` | `0.0378` | `0.0314` | `0.5077` | `0.4923` |
| `Concerto decoder+Utonia` | `0.4376` | `0.0233` | `0.0164` | `0.5227` | `0.4773` |
| `Sonata linear+Utonia` | `0.4269` | `0.0276` | `0.0272` | `0.5183` | `0.4817` |

## Interpretation Gate

- If `oracle::Concerto decoder+Utonia` is far above both singles but simple `avgprob`/confidence gates do not move, complementarity exists but requires learned fusion.
- If simple fusion already beats the Concerto full-FT reference (`~0.8075` in this repo, `80.7` in the paper), this becomes a concrete SOTA-method line.
- If two-model oracle is only marginally above the best single model, cross-model fusion is unlikely to be the right SOTA direction.
