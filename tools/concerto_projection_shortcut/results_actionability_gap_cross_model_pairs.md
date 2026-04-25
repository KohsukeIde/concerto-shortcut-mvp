# Cross-Model Representation-Readout Actionability Gap

This table separates four quantities that should not be collapsed into a single claim:

- `point_feature_bal_acc`: whether pairwise semantic information is present in the frozen feature.
- `direct_pair_margin_bal_acc`: whether the released fixed 20-way readout realizes that pairwise information.
- `positive_to_negative`: the observed target-to-confusion error rate.
- `oracle_topK_headroom`: candidate-set/actionability headroom, not a realizable method.

| model | pair | point BA | direct BA | direct-point | base IoU | pos->neg | top2 | oracle top2 IoU | top2 headroom | top5 headroom |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Concerto` | `picture_vs_wall` | `0.7041` | `0.6987` | `-0.0054` | `0.4034` | `0.4382` | `0.8929` | `0.8579` | `0.4545` | `0.5393` |
| `Concerto` | `door_vs_wall` | `0.8662` | `0.9518` | `0.0856` | `0.7603` | `0.0682` | `0.9840` | `0.9555` | `0.1953` | `0.2280` |
| `Concerto` | `counter_vs_cabinet` | `0.9276` | `0.9305` | `0.0028` | `0.6861` | `0.0947` | `0.9334` | `0.8985` | `0.2124` | `0.2961` |
| `Sonata` | `picture_vs_wall` | `0.7501` | `0.6452` | `-0.1050` | `0.3582` | `0.4783` | `0.7102` | `0.6972` | `0.3389` | `0.5284` |
| `Sonata` | `door_vs_wall` | `0.9207` | `0.9298` | `0.0091` | `0.6544` | `0.1145` | `0.9615` | `0.9135` | `0.2591` | `0.3374` |
| `Sonata` | `counter_vs_cabinet` | `0.9487` | `0.9453` | `-0.0034` | `0.6269` | `0.0867` | `0.9052` | `0.8568` | `0.2299` | `0.3498` |
| `PTv3 supervised` | `picture_vs_wall` | `0.9626` | `0.8892` | `-0.0735` | `0.4908` | `0.2326` | `0.8791` | `0.8785` | `0.3878` | `0.5045` |
| `PTv3 supervised` | `door_vs_wall` | `0.9361` | `0.9385` | `0.0024` | `0.7431` | `0.1000` | `0.9766` | `0.9460` | `0.2029` | `0.2435` |
| `PTv3 supervised` | `counter_vs_cabinet` | `0.9582` | `0.9709` | `0.0127` | `0.7778` | `0.0220` | `0.9700` | `0.9287` | `0.1509` | `0.2129` |
| `Utonia` | `picture_vs_wall` | `0.8847` | `0.9320` | `0.0472` | `0.2952` | `0.1284` | `0.9994` | `0.9747` | `0.6795` | `0.7048` |
| `Utonia` | `door_vs_wall` | `0.7294` | `0.9624` | `0.2330` | `0.8049` | `0.0586` | `0.9900` | `0.9645` | `0.1596` | `0.1940` |
| `Utonia` | `counter_vs_cabinet` | `0.6740` | `0.9499` | `0.2759` | `0.7230` | `0.0519` | `0.9858` | `0.9311` | `0.2081` | `0.2761` |

## Model-Level Means

| model | mean point BA | mean direct BA | mean pos->neg | mean top2 headroom | picture pos->neg | picture top2 headroom |
|---|---:|---:|---:|---:|---:|---:|
| `Concerto` | `0.8326` | `0.8603` | `0.2004` | `0.2874` | `0.4382` | `0.4545` |
| `Sonata` | `0.8732` | `0.8401` | `0.2265` | `0.2760` | `0.4783` | `0.3389` |
| `PTv3 supervised` | `0.9523` | `0.9328` | `0.1182` | `0.2472` | `0.2326` | `0.3878` |
| `Utonia` | `0.7627` | `0.9481` | `0.0796` | `0.3491` | `0.1284` | `0.6795` |

## Interpretation

- The audited failures are not representation collapse: all rows retain substantial pairwise feature content.
- Concerto/Sonata show large `picture -> wall` confusion and weaker fixed direct margins than the feature/probe evidence would suggest.
- PTv3 and Utonia are cleaner on fixed pairwise readout/confusion, which argues against a universal ScanNet-only artifact.
- Utonia should be described as a constructive comparator: its released stack reduces the audited wall-confusion/fixed-margin gap, but oracle headroom remains and the cause cannot be attributed to a specific Utonia design component without ablation checkpoints.
