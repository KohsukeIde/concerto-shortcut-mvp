# Top-K Pairwise Re-ranking Decoder

## Summary

This is a same-checkpoint offline readout-correction pilot on the
`concerto_base_origin.pth` decoder-probe line.

It tests a stronger readout correction than the naive antisymmetric residual
expert. The base 20-way decoder logits are kept fixed. At prediction time, the
method forms a candidate set from the base top-K classes plus neighbors from a
small confusion graph, then applies a tiny candidate-local reranker. Corrections
are centered within the candidate set so the edit acts as local reranking rather
than a global class-prior shift.

Result: **weak positive, below gate**.

The reranker fits the train candidate problem. The first sweep over larger
lambda values overcorrected, but a follow-up small-lambda sweep found a tiny
positive: `topk2_lam0p05` gives `+0.00022` mIoU and `+0.00066` `picture` IoU.
This is far below a paper-relevant same-checkpoint positive, but it confirms
that the direction has a measurable, very small effect.

## Setup

Base checkpoint:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`

Config:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/config.py`

Implementation:

- `tools/concerto_projection_shortcut/fit_topk_pairwise_rerank_decoder.py`
- `tools/concerto_projection_shortcut/submit_topk_pairwise_rerank_decoder_abciq_qf.sh`

Run output:

- `data/runs/scannet_decoder_probe_origin/topk_pairwise_rerank_decoder/topk_pairwise_rerank_decoder.md`
- `data/runs/scannet_decoder_probe_origin/topk_pairwise_rerank_decoder/topk_pairwise_rerank_summary.csv`
- `data/runs/scannet_decoder_probe_origin/topk_pairwise_rerank_decoder/topk_pairwise_rerank_class_metrics.csv`
- `data/runs/scannet_decoder_probe_origin/topk_pairwise_rerank_decoder/topk_pairwise_reranker.pt`
- `data/runs/scannet_decoder_probe_origin/topk_pairwise_rerank_decoder_small_lambda/topk_pairwise_rerank_decoder.md`
- `data/runs/scannet_decoder_probe_origin/topk_pairwise_rerank_decoder_small_lambda/topk_pairwise_rerank_summary.csv`
- `data/runs/scannet_decoder_probe_origin/topk_pairwise_rerank_decoder_small_lambda/topk_pairwise_rerank_class_metrics.csv`
- `data/runs/scannet_decoder_probe_origin/topk_pairwise_rerank_decoder_small_lambda/topk_pairwise_reranker.pt`
- `data/logs/abciq/133349.qjcm.OU`
- `data/logs/abciq/133350.qjcm.OU`

This is **not** a retrained Concerto model. It is an offline same-checkpoint
readout correction over frozen decoder features and fixed base 20-way logits.

## Job

| run | job | resource | walltime used | status |
| --- | --- | --- | ---: | --- |
| top-K pairwise reranker | `133349.qjcm` | `rt_QF=1` | `00:02:17` | `Exit_status=0` |
| top-K pairwise reranker, small lambda sweep | `133350.qjcm` | `rt_QF=1` | `00:03:00` | `Exit_status=0` |

Walltime request was `00:20:00`.

## Training Diagnostics

Training cache:

- train batches seen: `18`
- train points collected: `600000`
- candidate coverage: `0.9983`
- valid candidate-train points: `598967`

Reranker trace:

| step | loss | candidate base acc | candidate rerank acc | delta RMS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1384 | 0.9579 | 0.9580 | 0.1191 |
| 200 | 0.0643 | 0.9569 | 0.9766 | 0.8777 |
| 400 | 0.0520 | 0.9594 | 0.9796 | 1.1175 |
| 600 | 0.0393 | 0.9618 | 0.9854 | 1.2550 |
| 800 | 0.0355 | 0.9637 | 0.9835 | 1.2520 |
| 1000 | 0.0367 | 0.9626 | 0.9849 | 1.2407 |
| 1200 | 0.0330 | 0.9565 | 0.9862 | 1.2659 |
| 1400 | 0.0304 | 0.9604 | 0.9873 | 1.2494 |
| 1600 | 0.0289 | 0.9601 | 0.9894 | 1.2645 |
| 1800 | 0.0296 | 0.9620 | 0.9885 | 1.2959 |
| 2000 | 0.0259 | 0.9600 | 0.9868 | 1.3779 |

The train candidate problem is learnable. The failure is therefore not a basic
wiring failure; it is a train-to-val / calibration problem for this correction
form.

## Full Val Result

Important caveat: this deterministic offline evaluator's base is `0.7789` mIoU,
whereas the precise decoder-probe eval recorded `0.7888`. Compare deltas within
this script, not absolute mIoU against the precise evaluator.

### Large Lambda Sweep

| criterion | variant | mIoU | delta mIoU | picture IoU | delta picture IoU | picture->wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | `base` | 0.7789 | +0.0000 | 0.4096 | +0.0000 | 0.4324 |
| best non-base mIoU | `topk2_lam0p25` | 0.7741 | -0.0047 | 0.3948 | -0.0148 | 0.3640 |

Top non-base variants by mIoU:

| variant | mIoU | delta | picture IoU | picture delta | picture->wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `topk2_lam0p25` | 0.7741 | -0.0047 | 0.3948 | -0.0148 | 0.3640 |
| `topk3_lam0p25` | 0.7738 | -0.0051 | 0.3951 | -0.0144 | 0.3639 |
| `topk5_lam0p25` | 0.7737 | -0.0051 | 0.3953 | -0.0142 | 0.3639 |

### Small Lambda Sweep

| criterion | variant | mIoU | delta mIoU | picture IoU | delta picture IoU | picture->wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | `base` | 0.7789 | +0.0000 | 0.4096 | +0.0000 | 0.4324 |
| best mIoU | `topk2_lam0p05` | 0.7791 | +0.0002 | 0.4102 | +0.0007 | 0.4181 |
| best picture | `topk3_lam0p05` | 0.7791 | +0.0002 | 0.4103 | +0.0007 | 0.4181 |

Top small-lambda variants:

| variant | mIoU | delta | picture IoU | picture delta | picture->wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `topk2_lam0p05` | 0.7791 | +0.0002 | 0.4102 | +0.0007 | 0.4181 |
| `topk3_lam0p05` | 0.7791 | +0.0002 | 0.4103 | +0.0007 | 0.4181 |
| `topk5_lam0p05` | 0.7791 | +0.0002 | 0.4103 | +0.0007 | 0.4181 |
| `topk2_lam0p02` | 0.7790 | +0.0002 | 0.4099 | +0.0004 | 0.4268 |

## Interpretation

- The reranker reduces `picture -> wall` relative to base. The small-lambda
  sweep shows that this can be done without immediate global damage, but the
  improvement is only around `+0.0002` mIoU and `+0.0007` `picture` IoU.
- Larger lambda values overcorrect badly and quickly hurt both `picture` and
  global mIoU.
- The method is therefore not a useful same-checkpoint positive yet. It is best
  read as evidence that readout-side correction has a real but currently tiny
  lever.

## Takeaway

The readout/calibration bottleneck diagnosis remains plausible, but this learned
top-K reranker is not yet the needed positive method. The next readout-side
attempt should add stricter validation-aware constraints, for example:

- train/validation split inside ScanNet train for lambda/model selection;
- stronger residual regularization or monotonicity constraints;
- class-prior calibrated correction instead of free candidate residual scores;
- or a lightweight decoder fine-tune objective rather than post-hoc logit
  surgery.
