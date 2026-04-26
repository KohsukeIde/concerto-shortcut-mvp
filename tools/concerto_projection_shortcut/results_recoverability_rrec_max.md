# Recoverability Table: R_rec^max

Definition: `R_rec^max = (best non-oracle score - base score) / (oracle score - base score)`.
This table separates available oracle headroom from the fraction recovered by the pre-specified five-method recovery suite.

Main-suite methods:

1. Decoupled classifier / class-prior correction.
2. Prototype or kNN readout.
3. Constrained Top-K reranking.
4. Fixed-rank LoRA.
5. Full fine-tuning.

Exploratory CoDA/CIDA/region/proposal/subgroup attempts are intentionally not included in `R_rec^max`; they belong in the appendix.

| model | base mIoU | oracle@2 | oracle@5 | frozen Δ mIoU | frozen R_rec@2 | adaptation Δ mIoU | adaptation R_rec@2 | base picture | picture oracle@2 | frozen Δ picture | frozen picture R_rec@2 | adaptation Δ picture | adaptation picture R_rec@2 | suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Concerto decoder` | `0.7778` | `0.9197` | `0.9775` | `0.0002` | `0.2%` | `0.0187` | `13.2%` | `0.4034` | `0.8579` | `0.0013` | `0.3%` | `0.0198` | `4.4%` | `5-method fixed suite: decoupled classifier, prototype/kNN, constrained Top-K, fixed-rank LoRA, full FT` |
| `Sonata linear` | `0.7086` | `0.8747` | `0.9670` | `` | `` | `` | `` | `0.3582` | `0.6972` | `` | `` | `` | `` | `5-method fixed suite not run; oracle/actionability diagnostics only` |
| `Utonia released stack` | `0.7574` | `0.9367` | `0.9908` | `` | `` | `` | `` | `0.2952` | `0.9747` | `` | `` | `` | `` | `5-method fixed suite not run; oracle/actionability diagnostics only` |
| `PTv3 supervised` | `0.7745` | `0.9038` | `0.9690` | `` | `` | `` | `` | `0.4908` | `0.8785` | `` | `` | `` | `` | `5-method fixed suite not run; oracle/actionability diagnostics only` |

## Interpretation

- Concerto has large top-2/top-5 oracle headroom, but the fixed frozen suite recovers essentially none of it.
- Full fine-tuning recovers a nonzero but still small fraction of the oracle headroom; it improves aggregate accuracy but does not close the actionability gap.
- For Sonata, Utonia, and PTv3, the fixed recovery suite has not been run. The main claim should therefore stay Concerto-centric for recovery, while external models support oracle/actionability comparisons.
