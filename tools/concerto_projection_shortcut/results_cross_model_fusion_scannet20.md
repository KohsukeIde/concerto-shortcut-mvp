# Cross-Model Complementarity / Fusion Audit

Raw-point aligned ScanNet20 audit. Current-Pointcept models use the validation `inverse` mapping to original scene points; Utonia uses its released inverse-restored raw logits.

## Results

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle::Concerto decoder+Sonata linear+Utonia` | `0.8556` | `0.9528` | `0.7834` | `0.5690` | `0.3277` | `0.7975` | `0.8289` | `0.8502` |
| 2 | `oracle::Concerto decoder+Sonata linear` | `0.8331` | `0.9438` | `0.7537` | `0.5242` | `0.3516` | `0.7613` | `0.8016` | `0.8190` |
| 3 | `oracle::Concerto decoder+Utonia` | `0.8301` | `0.9426` | `0.7484` | `0.5008` | `0.3774` | `0.7638` | `0.7898` | `0.8256` |
| 4 | `oracle::Sonata linear+Utonia` | `0.8174` | `0.9367` | `0.7285` | `0.4975` | `0.3716` | `0.7443` | `0.7756` | `0.8013` |
| 5 | `avgprobT2::Concerto decoder+Utonia` | `0.7867` | `0.9235` | `0.6906` | `0.4170` | `0.4465` | `0.7007` | `0.7375` | `0.7782` |
| 6 | `avgprob::Concerto decoder+Utonia` | `0.7862` | `0.9233` | `0.6902` | `0.4165` | `0.4466` | `0.7006` | `0.7375` | `0.7777` |
| 7 | `maxconf::Concerto decoder+Utonia` | `0.7850` | `0.9229` | `0.6880` | `0.4146` | `0.4457` | `0.6982` | `0.7362` | `0.7762` |
| 8 | `maxmargin::Concerto decoder+Utonia` | `0.7848` | `0.9229` | `0.6882` | `0.4143` | `0.4470` | `0.6985` | `0.7370` | `0.7761` |
| 9 | `avgprob::Concerto decoder+Sonata linear+Utonia` | `0.7839` | `0.9229` | `0.6851` | `0.4067` | `0.4712` | `0.6977` | `0.7378` | `0.7681` |
| 10 | `maxconf::Concerto decoder+Sonata linear+Utonia` | `0.7804` | `0.9218` | `0.6837` | `0.4034` | `0.4677` | `0.7012` | `0.7339` | `0.7648` |
| 11 | `avgprobT2::Concerto decoder+Sonata linear` | `0.7797` | `0.9211` | `0.6797` | `0.4027` | `0.4683` | `0.6968` | `0.7278` | `0.7439` |
| 12 | `avgprob::Concerto decoder+Sonata linear` | `0.7794` | `0.9210` | `0.6792` | `0.4022` | `0.4669` | `0.6968` | `0.7272` | `0.7464` |
| 13 | `single::Concerto decoder` | `0.7782` | `0.9199` | `0.6802` | `0.4075` | `0.4369` | `0.6841` | `0.7204` | `0.7601` |
| 14 | `maxconf::Concerto decoder+Sonata linear` | `0.7782` | `0.9207` | `0.6777` | `0.4008` | `0.4636` | `0.6957` | `0.7254` | `0.7477` |
| 15 | `maxmargin::Concerto decoder+Sonata linear` | `0.7777` | `0.9205` | `0.6773` | `0.4006` | `0.4645` | `0.6958` | `0.7242` | `0.7473` |
| 16 | `weakgate::Concerto decoder->Utonia` | `0.7685` | `0.9174` | `0.6555` | `0.3780` | `0.4730` | `0.6575` | `0.7131` | `0.7461` |
| 17 | `picturewall_top2_gate::Concerto decoder->Utonia` | `0.7647` | `0.9141` | `0.6598` | `0.3780` | `0.4821` | `0.6702` | `0.7135` | `0.7505` |
| 18 | `avgprobT2::Sonata linear+Utonia` | `0.7591` | `0.9124` | `0.6612` | `0.3871` | `0.4930` | `0.6763` | `0.7065` | `0.7379` |
| 19 | `avgprob::Sonata linear+Utonia` | `0.7579` | `0.9121` | `0.6600` | `0.3847` | `0.4972` | `0.6757` | `0.7050` | `0.7352` |
| 20 | `single::Utonia` | `0.7570` | `0.9105` | `0.6555` | `0.3780` | `0.4821` | `0.6575` | `0.7131` | `0.7461` |
| 21 | `maxconf::Sonata linear+Utonia` | `0.7563` | `0.9116` | `0.6585` | `0.3855` | `0.4895` | `0.6724` | `0.7010` | `0.7354` |
| 22 | `maxmargin::Sonata linear+Utonia` | `0.7559` | `0.9114` | `0.6579` | `0.3838` | `0.4940` | `0.6731` | `0.7003` | `0.7334` |
| 23 | `single::Sonata linear` | `0.7093` | `0.8899` | `0.6054` | `0.3607` | `0.4755` | `0.6279` | `0.6132` | `0.6536` |

## Pairwise Complementarity

| pair | both correct | first only | second only | both wrong | oracle correct frac |
|---|---:|---:|---:|---:|---:|
| `Concerto decoder+Sonata linear` | `0.6272` | `0.0391` | `0.0173` | `0.3164` | `0.6836` |
| `Concerto decoder+Utonia` | `0.6431` | `0.0232` | `0.0164` | `0.3173` | `0.6827` |
| `Sonata linear+Utonia` | `0.6255` | `0.0190` | `0.0339` | `0.3215` | `0.6785` |

## Interpretation Gate

- If `oracle::Concerto decoder+Utonia` is far above both singles but simple `avgprob`/confidence gates do not move, complementarity exists but requires learned fusion.
- If simple fusion already beats the Concerto full-FT reference (`~0.8075` in this repo, `80.7` in the paper), this becomes a concrete SOTA-method line.
- If two-model oracle is only marginally above the best single model, cross-model fusion is unlikely to be the right SOTA direction.
