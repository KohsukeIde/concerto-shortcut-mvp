# Readout/Adaptation Structural Test Battery

This table is not a method leaderboard. It is a structural test for the claim that the oracle/actionability gap is not a trivial readout artifact.

| family | method | best delta mIoU | best delta picture | picture->wall effect | decision | source |
|---|---|---:|---:|---|---|---|
| `fixed-logit residual` | `confusion residual readout` | `0.00020` | `0.00055` | reduced slightly | below gate | `results_confusion_residual_readout.md` |
| `top-k rerank` | `top-K pairwise reranker` | `0.00022` | `0.00066` | reduced at small lambda | below gate | `results_topk_pairwise_rerank_decoder.md` |
| `validation-aware set rerank` | `constrained top-K set decoder` | `0.00012` | `0.00130` | 0.4386 -> 0.4277 | below gate | `results_constrained_topk_set_decoder.md` |
| `feature-conditioned adapter` | `CoDA residual decoder adapter` | `0.00024` | `0.00168` | 0.4436 -> 0.4232 | below gate / transfer failure | `results_coda_decoder_adapter.md` |
| `in-loop decoder adaptation` | `CIDA` | `-0.00312` | `-0.02026` | reduced slightly but collateral damage | no-go | `results_cida_inloop_decoder_adaptation.md` |
| `nonparametric retrieval` | `kNN readout` | `0.00020` | `0.00030` | minor | no-go | `results_knn_readout_small.md` |
| `prototype metric readout` | `prototype / multi-prototype readout` | `0.00020` | `0.00080` | minor | no-go | `results_prototype_readout.md` |
| `decoupled classifier` | `tau / cRT / balanced softmax` | `0.00020` | `-0.00060` | can reduce confusion but not IoU | no-go | `results_decoupled_classifier_readout.md` |
| `region readout` | `purity-aware hybrid region decoder` | `0.00020` | `0.00000` | no useful movement | no-go | `results_purity_aware_region_readout.md` |
| `proposal/readout` | `proposal-verify decoder` | `0.00000` | `0.00000` | base remains best | no-go | `results_proposal_verify_decoder.md` |
| `encoder adaptation` | `linear-head LoRA` | `0.01320` | `0.02250` | 0.4151 -> 0.3867 | positive only in low-capacity linear-head family | `results_scannet_lora_origin_perclass.md` |
| `decoder-capacity encoder adaptation` | `decoder + LoRA` | `-0.00280` | `-0.00130` | 0.4310 -> 0.4387 | no-go under decoder-capacity matching | `results_scannet_dec_lora_origin_perclass.md` |
| `full fine-tuning` | `Concerto origin official-like full FT` | `0.01870` | `0.01980` | residual 0.3956 in audit row | improves aggregate but does not close oracle headroom | `results_scannet_origin_fullft.md; results_scannet_origin_fullft_oracle_actionability/` |

## Interpretation

- Pairwise information and top-k candidate headroom exist, but fixed-logit, cached-feature, validation-aware rerank, nonparametric, decoupled-classifier, region, proposal, and simple adapter families recover almost none of the oracle headroom.
- The linear-head LoRA row shows the target confusion can move when the low-capacity head family is changed, but the gain disappears under decoder-capacity matching.
- Full fine-tuning improves aggregate mIoU and `picture`, but leaves large residual oracle headroom. This supports the phrasing `representation-readout actionability gap`, not `readout problem only`.
