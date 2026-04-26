# Fixed Recovery Suite: Family-Level Rows

These are the only recovery families included in the main-paper `R_rec^max` suite.

| model | base row | suite | family | best ΔmIoU | best Δpicture | source | notes |
|---|---|---|---|---:|---:|---|---|
| `Concerto decoder` | `frozen encoder + decoder probe` | `frozen` | `class-prior correction / decoupled classifier` | `0.0002` | `-0.0006` | `results_decoupled_classifier_readout.md` | tests long-tail / class-prior miscalibration; no meaningful recovery |
| `Concerto decoder` | `frozen encoder + decoder probe` | `frozen` | `nonparametric feature readout: prototype or kNN` | `0.0002` | `0.0008` | `results_knn_readout_small.md; results_prototype_readout.md` | tests nonparametric/metric readout geometry; no meaningful recovery |
| `Concerto decoder` | `frozen encoder + decoder probe` | `frozen` | `candidate-set reranking: constrained Top-K` | `0.0002` | `0.0013` | `results_topk_pairwise_rerank_decoder.md; results_constrained_topk_set_decoder.md` | tests whether oracle candidate-set headroom is recoverable by reranking; no meaningful recovery |
| `Concerto decoder` | `frozen encoder + decoder probe` | `adaptation` | `capacity-limited adaptation: fixed-rank LoRA` | `-0.0028` | `-0.0013` | `results_scannet_dec_lora_origin_perclass.md` | decoder-capacity-matched LoRA; same-head linear LoRA is positive but head-capacity confounded |
| `Concerto linear` | `frozen encoder + linear head` | `adaptation` | `capacity-limited adaptation: fixed-rank LoRA` | `0.0134` | `0.0289` | `results_scannet_lora_origin_perclass.md; results_scannet_linear_origin_oracle_actionability/` | protocol-matched to the linear-head family; strongest picture recovery among linear-head adaptation rows |
| `Concerto linear` | `frozen encoder + linear head` | `adaptation` | `LP-FT warm-start adaptation` | `0.0166` | `0.0125` | `results_scannet_lora_lpft_classsafe.md; results_scannet_lora_lpft_plain_oracle_actionability/` | protocol-matched to the linear-head family; strongest mIoU recovery among linear-head adaptation rows |
| `Concerto decoder` | `frozen encoder + decoder probe` | `adaptation` | `full fine-tuning` | `0.0187` | `0.0198` | `results_scannet_origin_fullft.md; results_scannet_origin_fullft_oracle_actionability/` | maximum practical adaptation budget; improves aggregate but leaves large oracle headroom |
| `Sonata linear` | `released backbone + linear head` | `frozen` | `class-prior correction / decoupled classifier` | `0.0017` | `-0.0001` | `results_sonata_recovery_decoupled_classifier.md` | small aggregate gain; picture does not recover |
| `Sonata linear` | `released backbone + linear head` | `frozen` | `candidate-set reranking: constrained Top-K` | `0.0000` | `0.0006` | `data/runs/sonata_recovery_topk/topk_pairwise_rerank_decoder.md` | no aggregate recovery; tiny picture-only movement |
| `Sonata full FT` | `released backbone + full fine-tuned head` | `adaptation` | `full fine-tuning` | `0.0684` | `-0.0074` | `results_sonata_fullft_oracle_actionability/oracle_actionability_analysis.md` | full-FT improves aggregate relative to Sonata linear under the oracle evaluator, but does not improve picture |
| `Utonia released stack` | `released stack / protocol-specific head` | `frozen/adaptation` | `6-family recovery suite` | `` | `` | `` | not run in a protocol-matched way yet; external rows report oracle/actionability diagnostics only |
| `PTv3 supervised` | `released stack / protocol-specific head` | `frozen/adaptation` | `6-family recovery suite` | `` | `` | `` | not run in a protocol-matched way yet; external rows report oracle/actionability diagnostics only |

## Appendix-only exploratory families

CoDA, CIDA, latent subgroup readout, region/superpoint smoothing, PHRD/PVD, proposal boosting, and other process-driven variants should be reported as exploratory recovery attempts, not as part of the pre-specified main suite.
