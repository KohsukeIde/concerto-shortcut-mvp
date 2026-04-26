# Recoverability Table: R_rec^max

Definition: `R_rec^max = (best non-oracle score - base score) / (oracle score - base score)`.
This table separates available oracle headroom from the fraction recovered by pre-specified non-oracle suites.

| model | base mIoU | oracle@2 | oracle@5 | frozen Δ mIoU | frozen R_rec@2 | adaptation Δ mIoU | adaptation R_rec@2 | base picture | picture oracle@2 | frozen Δ picture | frozen picture R_rec@2 | adaptation Δ picture | adaptation picture R_rec@2 | suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Concerto decoder` | `0.7778` | `0.9197` | `0.9775` | `0.0002` | `0.2%` | `0.0187` | `13.2%` | `0.4034` | `0.8579` | `0.0017` | `0.4%` | `0.0198` | `4.4%` | `Concerto structural test battery` |
| `Sonata linear` | `0.7086` | `0.8747` | `0.9670` | `0.0000` | `0.0%` | `` | `` | `0.3582` | `0.6972` | `0.0000` | `0.0%` | `` | `` | `oracle-analysis prior variants only; adaptation suite not run` |
| `Utonia released stack` | `0.7574` | `0.9367` | `0.9908` | `0.0000` | `0.0%` | `` | `` | `0.2952` | `0.9747` | `0.0000` | `0.0%` | `` | `` | `oracle-analysis prior/pair variants only; adaptation suite not run` |
| `PTv3 supervised` | `0.7745` | `0.9038` | `0.9690` | `0.0000` | `0.0%` | `` | `` | `0.4908` | `0.8785` | `0.0000` | `0.0%` | `` | `` | `oracle-analysis prior/bias variants only; adaptation suite not run` |

## Interpretation

- Concerto has large top-2/top-5 oracle headroom, but frozen/cached-feature families recover essentially none of it.
- Full fine-tuning recovers a nonzero but still small fraction of the oracle headroom; it improves aggregate accuracy but does not close the actionability gap.
- For Sonata, Utonia, and PTv3, the table intentionally reports only the recovery suites that have actually been run. Missing adaptation recovery is a gap to label as `not run`, not a zero.
