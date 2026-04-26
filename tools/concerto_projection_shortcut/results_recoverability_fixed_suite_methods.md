# Fixed Recovery Suite: Method-Level Rows

These are the only methods included in the main-paper `R_rec^max` suite.

| model | suite | method | best ΔmIoU | best Δpicture | source | notes |
|---|---|---|---:|---:|---|---|
| `Concerto decoder` | `frozen` | `decoupled classifier / class-prior correction` | `0.0002` | `-0.0006` | `results_decoupled_classifier_readout.md` | tests long-tail / class-prior miscalibration; no meaningful recovery |
| `Concerto decoder` | `frozen` | `prototype or kNN readout` | `0.0002` | `0.0008` | `results_knn_readout_small.md; results_prototype_readout.md` | tests nonparametric/metric readout geometry; no meaningful recovery |
| `Concerto decoder` | `frozen` | `constrained Top-K reranking` | `0.0002` | `0.0013` | `results_topk_pairwise_rerank_decoder.md; results_constrained_topk_set_decoder.md` | tests whether oracle candidate-set headroom is recoverable by reranking; no meaningful recovery |
| `Concerto decoder` | `adaptation` | `fixed-rank LoRA` | `-0.0028` | `-0.0013` | `results_scannet_dec_lora_origin_perclass.md` | decoder-capacity-matched LoRA; same-head linear LoRA is positive but head-capacity confounded |
| `Concerto decoder` | `adaptation` | `full fine-tuning` | `0.0187` | `0.0198` | `results_scannet_origin_fullft.md; results_scannet_origin_fullft_oracle_actionability/` | maximum practical adaptation budget; improves aggregate but leaves large oracle headroom |
| `Sonata linear` | `fixed suite` | `5-method fixed recovery suite` | `` | `` | `` | not run; external rows report oracle/actionability diagnostics only |
| `Utonia released stack` | `fixed suite` | `5-method fixed recovery suite` | `` | `` | `` | not run; external rows report oracle/actionability diagnostics only |
| `PTv3 supervised` | `fixed suite` | `5-method fixed recovery suite` | `` | `` | `` | not run; external rows report oracle/actionability diagnostics only |

## Appendix-only exploratory families

CoDA, CIDA, latent subgroup readout, region/superpoint smoothing, PHRD/PVD, proposal boosting, and other process-driven variants should be reported as exploratory recovery attempts, not as part of the pre-specified main suite.
