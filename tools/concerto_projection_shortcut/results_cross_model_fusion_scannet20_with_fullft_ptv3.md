# Cross-Model Complementarity / Fusion Audit

Raw-point aligned ScanNet20 audit. Current-Pointcept models use the validation `inverse` mapping to original scene points; Utonia uses its released inverse-restored raw logits.

## Results

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle::Concerto decoder+Sonata linear+Utonia+Concerto fullFT+PTv3_supervised` | `0.8969` | `0.9684` | `0.8399` | `0.6397` | `0.2873` | `0.8440` | `0.8989` | `0.9023` |
| 2 | `oracle::Utonia+Concerto fullFT` | `0.8547` | `0.9515` | `0.7807` | `0.5692` | `0.3224` | `0.7871` | `0.8474` | `0.8518` |
| 3 | `oracle::Utonia+PTv3_supervised` | `0.8495` | `0.9503` | `0.7718` | `0.5274` | `0.3896` | `0.7866` | `0.8418` | `0.8506` |
| 4 | `oracle::Concerto decoder+Concerto fullFT` | `0.8492` | `0.9499` | `0.7741` | `0.5382` | `0.3328` | `0.7832` | `0.8317` | `0.8405` |
| 5 | `oracle::Concerto decoder+PTv3_supervised` | `0.8485` | `0.9506` | `0.7738` | `0.5403` | `0.3514` | `0.7786` | `0.8245` | `0.8472` |
| 6 | `oracle::Sonata linear+Concerto fullFT` | `0.8480` | `0.9496` | `0.7698` | `0.5103` | `0.3597` | `0.7711` | `0.8360` | `0.8350` |
| 7 | `oracle::Concerto fullFT+PTv3_supervised` | `0.8429` | `0.9478` | `0.7628` | `0.5173` | `0.3705` | `0.7549` | `0.8344` | `0.8446` |
| 8 | `oracle::Concerto decoder+Sonata linear` | `0.8331` | `0.9438` | `0.7537` | `0.5242` | `0.3516` | `0.7613` | `0.8016` | `0.8190` |
| 9 | `oracle::Concerto decoder+Utonia` | `0.8301` | `0.9426` | `0.7484` | `0.5008` | `0.3774` | `0.7638` | `0.7898` | `0.8256` |
| 10 | `oracle::Sonata linear+PTv3_supervised` | `0.8274` | `0.9432` | `0.7469` | `0.4809` | `0.4080` | `0.7672` | `0.8033` | `0.8106` |
| 11 | `oracle::Sonata linear+Utonia` | `0.8174` | `0.9367` | `0.7285` | `0.4976` | `0.3716` | `0.7442` | `0.7756` | `0.8013` |
| 12 | `avgprob::Concerto decoder+Sonata linear+Utonia+Concerto fullFT+PTv3_supervised` | `0.8065` | `0.9321` | `0.7117` | `0.4230` | `0.4655` | `0.7210` | `0.7709` | `0.7905` |
| 13 | `avgprob::Concerto decoder+Concerto fullFT` | `0.8051` | `0.9313` | `0.7106` | `0.4355` | `0.4156` | `0.7177` | `0.7825` | `0.7847` |
| 14 | `avgprobT2::Concerto decoder+Concerto fullFT` | `0.8049` | `0.9311` | `0.7105` | `0.4358` | `0.4126` | `0.7163` | `0.7844` | `0.7840` |
| 15 | `maxconf::Concerto decoder+Concerto fullFT` | `0.8046` | `0.9311` | `0.7097` | `0.4351` | `0.4147` | `0.7178` | `0.7806` | `0.7841` |
| 16 | `maxmargin::Concerto decoder+Concerto fullFT` | `0.8046` | `0.9311` | `0.7097` | `0.4352` | `0.4153` | `0.7180` | `0.7804` | `0.7842` |
| 17 | `avgprobT2::Utonia+Concerto fullFT` | `0.8037` | `0.9307` | `0.7092` | `0.4369` | `0.4090` | `0.7205` | `0.7855` | `0.7851` |
| 18 | `maxconf::Concerto decoder+Sonata linear+Utonia+Concerto fullFT+PTv3_supervised` | `0.8034` | `0.9310` | `0.7095` | `0.4333` | `0.4438` | `0.7202` | `0.7826` | `0.7822` |
| 19 | `avgprob::Utonia+Concerto fullFT` | `0.8032` | `0.9305` | `0.7085` | `0.4372` | `0.4109` | `0.7203` | `0.7849` | `0.7850` |
| 20 | `maxconf::Utonia+Concerto fullFT` | `0.8027` | `0.9304` | `0.7078` | `0.4366` | `0.4091` | `0.7199` | `0.7839` | `0.7845` |
| 21 | `maxmargin::Utonia+Concerto fullFT` | `0.8024` | `0.9303` | `0.7075` | `0.4367` | `0.4103` | `0.7193` | `0.7838` | `0.7846` |
| 22 | `avgprobT2::Concerto fullFT+PTv3_supervised` | `0.8021` | `0.9300` | `0.7085` | `0.4349` | `0.4318` | `0.7202` | `0.7785` | `0.7775` |
| 23 | `avgprob::Concerto fullFT+PTv3_supervised` | `0.8012` | `0.9298` | `0.7072` | `0.4321` | `0.4381` | `0.7206` | `0.7769` | `0.7760` |
| 24 | `maxconf::Concerto fullFT+PTv3_supervised` | `0.8007` | `0.9297` | `0.7065` | `0.4327` | `0.4352` | `0.7196` | `0.7760` | `0.7750` |
| 25 | `maxmargin::Concerto fullFT+PTv3_supervised` | `0.8005` | `0.9296` | `0.7062` | `0.4316` | `0.4373` | `0.7197` | `0.7757` | `0.7748` |
| 26 | `avgprobT2::Sonata linear+Concerto fullFT` | `0.8001` | `0.9295` | `0.7061` | `0.4380` | `0.4098` | `0.7095` | `0.7829` | `0.7776` |
| 27 | `avgprob::Sonata linear+Concerto fullFT` | `0.7992` | `0.9292` | `0.7053` | `0.4386` | `0.4119` | `0.7090` | `0.7824` | `0.7768` |
| 28 | `maxconf::Sonata linear+Concerto fullFT` | `0.7985` | `0.9289` | `0.7044` | `0.4385` | `0.4096` | `0.7087` | `0.7806` | `0.7752` |
| 29 | `maxmargin::Sonata linear+Concerto fullFT` | `0.7980` | `0.9288` | `0.7038` | `0.4387` | `0.4107` | `0.7081` | `0.7799` | `0.7751` |
| 30 | `single::Concerto fullFT` | `0.7969` | `0.9276` | `0.7014` | `0.4296` | `0.4015` | `0.7062` | `0.7760` | `0.7721` |
| 31 | `avgprob::Concerto decoder+PTv3_supervised` | `0.7898` | `0.9254` | `0.6906` | `0.3966` | `0.5120` | `0.7092` | `0.7399` | `0.7625` |
| 32 | `avgprobT2::Concerto decoder+PTv3_supervised` | `0.7895` | `0.9249` | `0.6900` | `0.3985` | `0.5083` | `0.7106` | `0.7393` | `0.7609` |
| 33 | `maxmargin::Concerto decoder+PTv3_supervised` | `0.7890` | `0.9251` | `0.6896` | `0.3965` | `0.5105` | `0.7083` | `0.7362` | `0.7611` |
| 34 | `maxconf::Concerto decoder+PTv3_supervised` | `0.7887` | `0.9249` | `0.6891` | `0.3973` | `0.5094` | `0.7082` | `0.7363` | `0.7604` |
| 35 | `avgprobT2::Concerto decoder+Utonia` | `0.7867` | `0.9235` | `0.6906` | `0.4170` | `0.4465` | `0.7008` | `0.7375` | `0.7782` |
| 36 | `avgprob::Concerto decoder+Utonia` | `0.7862` | `0.9233` | `0.6902` | `0.4164` | `0.4466` | `0.7006` | `0.7375` | `0.7777` |
| 37 | `avgprobT2::Utonia+PTv3_supervised` | `0.7856` | `0.9237` | `0.6804` | `0.3957` | `0.5096` | `0.7030` | `0.7283` | `0.7574` |
| 38 | `maxconf::Concerto decoder+Utonia` | `0.7850` | `0.9229` | `0.6880` | `0.4146` | `0.4457` | `0.6982` | `0.7362` | `0.7762` |
| 39 | `maxmargin::Concerto decoder+Utonia` | `0.7848` | `0.9229` | `0.6882` | `0.4143` | `0.4470` | `0.6985` | `0.7370` | `0.7761` |
| 40 | `avgprob::Utonia+PTv3_supervised` | `0.7837` | `0.9232` | `0.6790` | `0.3950` | `0.5121` | `0.7013` | `0.7286` | `0.7564` |
| 41 | `maxconf::Utonia+PTv3_supervised` | `0.7824` | `0.9228` | `0.6773` | `0.3956` | `0.5087` | `0.7011` | `0.7263` | `0.7543` |
| 42 | `maxmargin::Utonia+PTv3_supervised` | `0.7823` | `0.9228` | `0.6773` | `0.3947` | `0.5109` | `0.7006` | `0.7272` | `0.7550` |
| 43 | `avgprobT2::Concerto decoder+Sonata linear` | `0.7797` | `0.9211` | `0.6797` | `0.4027` | `0.4683` | `0.6968` | `0.7278` | `0.7439` |
| 44 | `avgprob::Concerto decoder+Sonata linear` | `0.7794` | `0.9210` | `0.6792` | `0.4022` | `0.4669` | `0.6968` | `0.7272` | `0.7464` |
| 45 | `single::Concerto decoder` | `0.7782` | `0.9199` | `0.6802` | `0.4075` | `0.4369` | `0.6842` | `0.7204` | `0.7601` |
| 46 | `maxconf::Concerto decoder+Sonata linear` | `0.7782` | `0.9207` | `0.6777` | `0.4008` | `0.4637` | `0.6957` | `0.7254` | `0.7477` |
| 47 | `maxmargin::Concerto decoder+Sonata linear` | `0.7777` | `0.9205` | `0.6773` | `0.4006` | `0.4645` | `0.6958` | `0.7242` | `0.7473` |
| 48 | `picturewall_top2_gate::Concerto decoder->PTv3_supervised` | `0.7697` | `0.9156` | `0.6639` | `0.3824` | `0.5068` | `0.6800` | `0.7002` | `0.7419` |
| 49 | `avgprobT2::Sonata linear+PTv3_supervised` | `0.7688` | `0.9173` | `0.6653` | `0.3912` | `0.5055` | `0.7055` | `0.7104` | `0.7249` |
| 50 | `avgprob::Sonata linear+PTv3_supervised` | `0.7681` | `0.9171` | `0.6647` | `0.3920` | `0.5053` | `0.7044` | `0.7105` | `0.7248` |
| 51 | `maxconf::Sonata linear+PTv3_supervised` | `0.7671` | `0.9167` | `0.6637` | `0.3930` | `0.5013` | `0.7021` | `0.7081` | `0.7244` |
| 52 | `maxmargin::Sonata linear+PTv3_supervised` | `0.7670` | `0.9167` | `0.6636` | `0.3927` | `0.5019` | `0.7026` | `0.7087` | `0.7240` |
| 53 | `weakgate::Concerto decoder->PTv3_supervised` | `0.7637` | `0.9161` | `0.6482` | `0.3824` | `0.4958` | `0.6950` | `0.6871` | `0.7161` |
| 54 | `avgprobT2::Sonata linear+Utonia` | `0.7590` | `0.9124` | `0.6612` | `0.3871` | `0.4930` | `0.6763` | `0.7065` | `0.7379` |
| 55 | `avgprob::Sonata linear+Utonia` | `0.7579` | `0.9121` | `0.6600` | `0.3847` | `0.4972` | `0.6757` | `0.7050` | `0.7352` |
| 56 | `single::PTv3_supervised` | `0.7574` | `0.9126` | `0.6482` | `0.3824` | `0.5068` | `0.6950` | `0.6871` | `0.7161` |
| 57 | `single::Utonia` | `0.7570` | `0.9105` | `0.6555` | `0.3780` | `0.4821` | `0.6574` | `0.7131` | `0.7461` |
| 58 | `maxconf::Sonata linear+Utonia` | `0.7563` | `0.9116` | `0.6584` | `0.3855` | `0.4896` | `0.6723` | `0.7010` | `0.7353` |
| 59 | `maxmargin::Sonata linear+Utonia` | `0.7559` | `0.9114` | `0.6579` | `0.3837` | `0.4940` | `0.6730` | `0.7003` | `0.7334` |
| 60 | `single::Sonata linear` | `0.7093` | `0.8899` | `0.6054` | `0.3607` | `0.4755` | `0.6279` | `0.6132` | `0.6536` |

## Pairwise Complementarity

| pair | both correct | first only | second only | both wrong | oracle correct frac |
|---|---:|---:|---:|---:|---:|
| `Concerto decoder+Sonata linear` | `0.6272` | `0.0391` | `0.0173` | `0.3164` | `0.6836` |
| `Concerto decoder+Utonia` | `0.6431` | `0.0232` | `0.0164` | `0.3173` | `0.6827` |
| `Concerto decoder+Concerto fullFT` | `0.6501` | `0.0162` | `0.0217` | `0.3120` | `0.6880` |
| `Concerto decoder+PTv3_supervised` | `0.6388` | `0.0275` | `0.0222` | `0.3115` | `0.6885` |
| `Sonata linear+Utonia` | `0.6255` | `0.0190` | `0.0339` | `0.3216` | `0.6784` |
| `Sonata linear+Concerto fullFT` | `0.6286` | `0.0160` | `0.0432` | `0.3122` | `0.6878` |
| `Sonata linear+PTv3_supervised` | `0.6224` | `0.0221` | `0.0386` | `0.3169` | `0.6831` |
| `Utonia+Concerto fullFT` | `0.6421` | `0.0173` | `0.0297` | `0.3108` | `0.6892` |
| `Utonia+PTv3_supervised` | `0.6321` | `0.0273` | `0.0288` | `0.3117` | `0.6883` |
| `Concerto fullFT+PTv3_supervised` | `0.6463` | `0.0255` | `0.0147` | `0.3135` | `0.6865` |

## Interpretation Gate

- If `oracle::Concerto decoder+Utonia` is far above both singles but simple `avgprob`/confidence gates do not move, complementarity exists but requires learned fusion.
- If simple fusion already beats the Concerto full-FT reference (`~0.8075` in this repo, `80.7` in the paper), this becomes a concrete SOTA-method line.
- If two-model oracle is only marginally above the best single model, cross-model fusion is unlikely to be the right SOTA direction.
