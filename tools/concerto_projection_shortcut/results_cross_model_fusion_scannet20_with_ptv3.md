# Cross-Model Complementarity / Fusion Audit

Raw-point aligned ScanNet20 audit. Current-Pointcept models use the validation `inverse` mapping to original scene points; Utonia uses its released inverse-restored raw logits.

## Results

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle::Concerto decoder+Sonata linear+Utonia+PTv3_supervised` | `0.8843` | `0.9641` | `0.8221` | `0.6095` | `0.3081` | `0.8323` | `0.8802` | `0.8855` |
| 2 | `oracle::Utonia+PTv3_supervised` | `0.8495` | `0.9503` | `0.7718` | `0.5274` | `0.3896` | `0.7866` | `0.8418` | `0.8506` |
| 3 | `oracle::Concerto decoder+PTv3_supervised` | `0.8485` | `0.9506` | `0.7738` | `0.5403` | `0.3514` | `0.7786` | `0.8245` | `0.8472` |
| 4 | `oracle::Concerto decoder+Sonata linear` | `0.8331` | `0.9438` | `0.7537` | `0.5242` | `0.3515` | `0.7613` | `0.8016` | `0.8190` |
| 5 | `oracle::Concerto decoder+Utonia` | `0.8301` | `0.9426` | `0.7484` | `0.5008` | `0.3773` | `0.7638` | `0.7898` | `0.8256` |
| 6 | `oracle::Sonata linear+PTv3_supervised` | `0.8274` | `0.9432` | `0.7469` | `0.4809` | `0.4080` | `0.7672` | `0.8033` | `0.8106` |
| 7 | `oracle::Sonata linear+Utonia` | `0.8174` | `0.9367` | `0.7285` | `0.4976` | `0.3716` | `0.7442` | `0.7756` | `0.8013` |
| 8 | `avgprob::Concerto decoder+Sonata linear+Utonia+PTv3_supervised` | `0.7959` | `0.9278` | `0.6987` | `0.4043` | `0.5001` | `0.7186` | `0.7497` | `0.7774` |
| 9 | `avgprob::Concerto decoder+PTv3_supervised` | `0.7898` | `0.9254` | `0.6906` | `0.3966` | `0.5120` | `0.7092` | `0.7400` | `0.7625` |
| 10 | `avgprobT2::Concerto decoder+PTv3_supervised` | `0.7895` | `0.9249` | `0.6900` | `0.3985` | `0.5083` | `0.7106` | `0.7393` | `0.7609` |
| 11 | `maxmargin::Concerto decoder+PTv3_supervised` | `0.7890` | `0.9251` | `0.6896` | `0.3965` | `0.5105` | `0.7083` | `0.7362` | `0.7611` |
| 12 | `maxconf::Concerto decoder+PTv3_supervised` | `0.7887` | `0.9249` | `0.6891` | `0.3973` | `0.5094` | `0.7082` | `0.7363` | `0.7603` |
| 13 | `maxconf::Concerto decoder+Sonata linear+Utonia+PTv3_supervised` | `0.7886` | `0.9256` | `0.6885` | `0.3973` | `0.5109` | `0.7043` | `0.7432` | `0.7640` |
| 14 | `avgprobT2::Concerto decoder+Utonia` | `0.7867` | `0.9235` | `0.6906` | `0.4170` | `0.4465` | `0.7008` | `0.7375` | `0.7782` |
| 15 | `avgprob::Concerto decoder+Utonia` | `0.7862` | `0.9233` | `0.6902` | `0.4165` | `0.4466` | `0.7006` | `0.7375` | `0.7777` |
| 16 | `avgprobT2::Utonia+PTv3_supervised` | `0.7856` | `0.9237` | `0.6804` | `0.3957` | `0.5096` | `0.7030` | `0.7283` | `0.7574` |
| 17 | `maxconf::Concerto decoder+Utonia` | `0.7850` | `0.9229` | `0.6880` | `0.4147` | `0.4457` | `0.6982` | `0.7362` | `0.7762` |
| 18 | `maxmargin::Concerto decoder+Utonia` | `0.7848` | `0.9229` | `0.6882` | `0.4143` | `0.4470` | `0.6985` | `0.7370` | `0.7761` |
| 19 | `avgprob::Utonia+PTv3_supervised` | `0.7837` | `0.9232` | `0.6790` | `0.3950` | `0.5121` | `0.7013` | `0.7286` | `0.7564` |
| 20 | `maxconf::Utonia+PTv3_supervised` | `0.7824` | `0.9228` | `0.6773` | `0.3956` | `0.5087` | `0.7011` | `0.7263` | `0.7543` |
| 21 | `maxmargin::Utonia+PTv3_supervised` | `0.7823` | `0.9228` | `0.6773` | `0.3947` | `0.5109` | `0.7006` | `0.7272` | `0.7550` |
| 22 | `avgprobT2::Concerto decoder+Sonata linear` | `0.7797` | `0.9211` | `0.6797` | `0.4027` | `0.4683` | `0.6968` | `0.7278` | `0.7440` |
| 23 | `avgprob::Concerto decoder+Sonata linear` | `0.7794` | `0.9210` | `0.6792` | `0.4022` | `0.4669` | `0.6968` | `0.7272` | `0.7464` |
| 24 | `single::Concerto decoder` | `0.7782` | `0.9199` | `0.6802` | `0.4075` | `0.4369` | `0.6841` | `0.7204` | `0.7601` |
| 25 | `maxconf::Concerto decoder+Sonata linear` | `0.7782` | `0.9207` | `0.6777` | `0.4008` | `0.4636` | `0.6957` | `0.7254` | `0.7477` |
| 26 | `maxmargin::Concerto decoder+Sonata linear` | `0.7777` | `0.9205` | `0.6773` | `0.4006` | `0.4645` | `0.6958` | `0.7242` | `0.7473` |
| 27 | `picturewall_top2_gate::Concerto decoder->PTv3_supervised` | `0.7697` | `0.9156` | `0.6639` | `0.3824` | `0.5068` | `0.6800` | `0.7002` | `0.7419` |
| 28 | `avgprobT2::Sonata linear+PTv3_supervised` | `0.7688` | `0.9173` | `0.6653` | `0.3912` | `0.5055` | `0.7054` | `0.7104` | `0.7249` |
| 29 | `avgprob::Sonata linear+PTv3_supervised` | `0.7681` | `0.9171` | `0.6647` | `0.3920` | `0.5053` | `0.7044` | `0.7105` | `0.7248` |
| 30 | `maxconf::Sonata linear+PTv3_supervised` | `0.7671` | `0.9167` | `0.6637` | `0.3930` | `0.5013` | `0.7021` | `0.7081` | `0.7244` |
| 31 | `maxmargin::Sonata linear+PTv3_supervised` | `0.7670` | `0.9167` | `0.6636` | `0.3927` | `0.5019` | `0.7026` | `0.7088` | `0.7240` |
| 32 | `weakgate::Concerto decoder->PTv3_supervised` | `0.7637` | `0.9161` | `0.6482` | `0.3824` | `0.4958` | `0.6950` | `0.6871` | `0.7161` |
| 33 | `avgprobT2::Sonata linear+Utonia` | `0.7590` | `0.9124` | `0.6612` | `0.3871` | `0.4930` | `0.6763` | `0.7065` | `0.7379` |
| 34 | `avgprob::Sonata linear+Utonia` | `0.7579` | `0.9121` | `0.6600` | `0.3847` | `0.4972` | `0.6757` | `0.7050` | `0.7352` |
| 35 | `single::PTv3_supervised` | `0.7574` | `0.9126` | `0.6482` | `0.3824` | `0.5068` | `0.6950` | `0.6871` | `0.7161` |
| 36 | `single::Utonia` | `0.7570` | `0.9105` | `0.6555` | `0.3780` | `0.4821` | `0.6574` | `0.7131` | `0.7461` |
| 37 | `maxconf::Sonata linear+Utonia` | `0.7563` | `0.9116` | `0.6584` | `0.3855` | `0.4896` | `0.6723` | `0.7010` | `0.7354` |
| 38 | `maxmargin::Sonata linear+Utonia` | `0.7559` | `0.9114` | `0.6579` | `0.3837` | `0.4940` | `0.6731` | `0.7003` | `0.7334` |
| 39 | `single::Sonata linear` | `0.7093` | `0.8899` | `0.6054` | `0.3607` | `0.4755` | `0.6279` | `0.6132` | `0.6536` |

## Pairwise Complementarity

| pair | both correct | first only | second only | both wrong | oracle correct frac |
|---|---:|---:|---:|---:|---:|
| `Concerto decoder+Sonata linear` | `0.6272` | `0.0391` | `0.0173` | `0.3164` | `0.6836` |
| `Concerto decoder+Utonia` | `0.6431` | `0.0232` | `0.0164` | `0.3173` | `0.6827` |
| `Concerto decoder+PTv3_supervised` | `0.6388` | `0.0275` | `0.0222` | `0.3115` | `0.6885` |
| `Sonata linear+Utonia` | `0.6255` | `0.0190` | `0.0339` | `0.3216` | `0.6784` |
| `Sonata linear+PTv3_supervised` | `0.6224` | `0.0221` | `0.0386` | `0.3169` | `0.6831` |
| `Utonia+PTv3_supervised` | `0.6321` | `0.0273` | `0.0288` | `0.3117` | `0.6883` |

## Interpretation Gate

- If `oracle::Concerto decoder+Utonia` is far above both singles but simple `avgprob`/confidence gates do not move, complementarity exists but requires learned fusion.
- If simple fusion already beats the Concerto full-FT reference (`~0.8075` in this repo, `80.7` in the paper), this becomes a concrete SOTA-method line.
- If two-model oracle is only marginally above the best single model, cross-model fusion is unlikely to be the right SOTA direction.
