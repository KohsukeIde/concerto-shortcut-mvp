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
| `Sonata linear` | `0.7086` | `0.8747` | `0.9670` | `0.0017` | `1.0%` | `0.0684` | `41.2%` | `0.3582` | `0.6972` | `0.0006` | `0.2%` | `-0.0074` | `0.0%` | `6-family suite partially run: frozen class-prior, prototype/kNN, and constrained Top-K rows complete; full-FT adaptation integrated. LoRA/LP-FT not run.` |
| `Utonia released stack` | `0.7573` | `0.9116` | `0.9821` | `0.0001` | `0.1%` | `` | `` | `0.3749` | `0.8043` | `0.0021` | `0.5%` | `` | `` | `protocol-matched frozen recovery suite complete: class-prior, prototype/multiprototype, and candidate-set pair rerank. Adaptation suite not interpreted for this row.` |
| `PTv3 supervised` | `0.7716` | `0.8889` | `0.9647` | `0.0013` | `1.1%` | `` | `` | `0.3910` | `0.6763` | `0.0081` | `2.8%` | `` | `` | `protocol-matched frozen recovery suite complete: class-prior, prototype/multiprototype, and candidate-set pair rerank. Adaptation suite not interpreted for this row.` |

## Interpretation

- Concerto has large top-2/top-5 oracle headroom, but the fixed frozen suite recovers essentially none of it.
- Full fine-tuning recovers a nonzero but still small fraction of the oracle headroom; it improves aggregate accuracy but does not close the actionability gap.
- LP-FT is a linear-head-family adaptation row. It should be reported as protocol-matched to the Concerto linear base, not as recovery for the decoder-probe oracle denominator.
- Sonata now has protocol-matched frozen recovery coverage for class-prior, prototype/kNN, and constrained Top-K families, plus full fine-tuning as a high-budget adaptation row. Aggregate recovery is possible under full FT, but picture recovery remains poor.
- Utonia/PTv3 recovery rows are included only when the protocol-matched frozen recovery output exists. Otherwise the row remains explicitly scoped as oracle/actionability-only.
