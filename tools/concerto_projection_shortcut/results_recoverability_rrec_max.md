# Recoverability Table: R_rec^max

Definition: `R_rec^max = (best non-oracle score - base score) / (oracle score - base score)`.
This table separates available oracle headroom from the fraction recovered by the pre-specified six-family recovery suite.
`R_rec` is computed only within protocol-matched base/readout rows; LP-FT belongs to the Concerto linear-head family and is not mixed into the decoder-probe denominator.

Main-suite recovery families:

1. Class-prior correction / decoupled classifier.
2. Nonparametric feature readout: prototype or kNN.
3. Candidate-set reranking: constrained Top-K.
4. Capacity-limited adaptation: fixed-rank LoRA.
5. LP-FT warm-start adaptation.
6. Full fine-tuning.

Exploratory CoDA/CIDA/region/proposal/subgroup attempts are intentionally not included in `R_rec^max`; they belong in the appendix.

| model | base mIoU | oracle@2 | oracle@5 | frozen Δ mIoU | frozen R_rec@2 | adaptation Δ mIoU | adaptation R_rec@2 | base picture | picture oracle@2 | frozen Δ picture | frozen picture R_rec@2 | adaptation Δ picture | adaptation picture R_rec@2 | suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Concerto decoder` | `0.7778` | `0.9197` | `0.9775` | `0.0002` | `0.2%` | `0.0187` | `13.2%` | `0.4034` | `0.8579` | `0.0013` | `0.3%` | `0.0198` | `4.4%` | `6-family suite; decoder-compatible families used here: class-prior, prototype/kNN, constrained Top-K, decoder-matched LoRA, full FT. LP-FT is tracked separately for the linear-head base.` |
| `Concerto linear` | `0.7615` | `0.9171` | `0.9839` | `` | `` | `0.0166` | `10.7%` | `0.4014` | `0.8013` | `` | `` | `0.0289` | `7.2%` | `6-family suite; linear-head-compatible adaptation families used here: fixed-rank LoRA and LP-FT. Frozen suite pending for this base row.` |
| `Sonata linear` | `0.7086` | `0.8747` | `0.9670` | `0.0017` | `1.0%` | `0.0684` | `41.2%` | `0.3582` | `0.6972` | `0.0006` | `0.2%` | `-0.0074` | `0.0%` | `6-family suite partially run: class-prior and constrained Top-K frozen rows complete, prototype/kNN pending, full-FT adaptation integrated. LoRA/LP-FT not run.` |
| `Utonia released stack` | `0.7574` | `0.9367` | `0.9908` | `` | `` | `` | `` | `0.2952` | `0.9747` | `` | `` | `` | `` | `6-family recovery suite not run in a protocol-matched way; oracle/actionability diagnostics only` |
| `PTv3 supervised` | `0.7745` | `0.9038` | `0.9690` | `` | `` | `` | `` | `0.4908` | `0.8785` | `` | `` | `` | `` | `6-family recovery suite not run in a protocol-matched way; oracle/actionability diagnostics only` |

## Interpretation

- Concerto has large top-2/top-5 oracle headroom, but the fixed frozen suite recovers essentially none of it.
- Full fine-tuning recovers a nonzero but still small fraction of the oracle headroom; it improves aggregate accuracy but does not close the actionability gap.
- LP-FT is a linear-head-family adaptation row. It should be reported as protocol-matched to the Concerto linear base, not as recovery for the decoder-probe oracle denominator.
- Sonata now has partial protocol-matched recovery coverage: class-prior and constrained Top-K frozen rows are complete, prototype/kNN is pending, and full fine-tuning provides the high-budget adaptation row. Aggregate recovery is possible under full FT, but picture recovery remains poor.
- For Utonia and PTv3, the six-family recovery suite has not been run in a protocol-matched way. Keep their recovery interpretation limited to oracle/actionability comparisons unless custom recovery paths are added.
