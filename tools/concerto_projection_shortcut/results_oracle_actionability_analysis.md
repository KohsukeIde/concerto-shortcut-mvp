# Oracle Actionability Analysis

## Summary

This analysis asks whether the `picture -> wall` and related weak-class errors
are reachable by readout-family methods, or whether the errors are structurally
outside the trained 20-way decoder's candidate set.

Result: **large oracle headroom, but current learned/calibrated readout proxies
fail**.

The key result is that ground truth is usually already in the decoder's candidate
set:

- `picture` top-2 hit rate: `0.8929` with 95% CI `[0.8916, 0.8943]`
- `picture` top-5 hit rate: `0.9599` with 95% CI `[0.9590, 0.9607]`
- `picture` top-1 plus confusion-graph-neighbor hit rate: `0.9748`

Thus, the `picture` label is usually present near the top of the 20-way logits.
In principle, a better local decision rule could rescue it. However, the
train-fitted pair probe and simple train-derived calibration variants do not
produce a useful validation improvement.

## Setup

Base checkpoint:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`

Config:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/config.py`

Implementation:

- `tools/concerto_projection_shortcut/eval_oracle_actionability_analysis.py`
- `tools/concerto_projection_shortcut/submit_oracle_actionability_analysis_abciq_qf.sh`

Run output:

- `data/runs/scannet_decoder_probe_origin/oracle_actionability_analysis/oracle_actionability_analysis.md`
- `data/runs/scannet_decoder_probe_origin/oracle_actionability_analysis/oracle_topk_hit_rates.csv`
- `data/runs/scannet_decoder_probe_origin/oracle_actionability_analysis/oracle_variants.csv`
- `data/runs/scannet_decoder_probe_origin/oracle_actionability_analysis/oracle_confusion_distribution.csv`
- `data/runs/scannet_decoder_probe_origin/oracle_actionability_analysis/oracle_top3_distribution.csv`
- `data/runs/scannet_decoder_probe_origin/oracle_actionability_analysis/oracle_feature_geometry.csv`
- `data/runs/scannet_decoder_probe_origin/oracle_actionability_analysis/oracle_pair_probe_train.csv`
- `data/logs/abciq/133352.qjcm.OU`

The first run `133351.qjcm` completed successfully but was superseded by
`133352.qjcm`, which adds confidence intervals to top-K hit rates.

## Job

| run | job | resource | walltime used | status |
| --- | --- | --- | ---: | --- |
| oracle/actionability analysis with top-K CI | `133352.qjcm` | `rt_QF=1` | `00:01:45` | `Exit_status=0` |

Walltime request was `00:25:00`.

## Candidate Headroom

Base deterministic offline eval:

- mIoU: `0.7778`
- `picture` IoU: `0.4034`
- `picture -> wall`: `0.4382`

Top-K hit rates:

| class | top-1 | top-2 | top-5 | graph top-1 |
| --- | ---: | ---: | ---: | ---: |
| picture | 0.5366 | 0.8929 | 0.9599 | 0.9748 |
| counter | 0.8090 | 0.9334 | 0.9854 | 0.9037 |
| desk | 0.8584 | 0.9581 | 0.9896 | 0.8911 |
| sink | 0.8490 | 0.9161 | 0.9776 | 0.9351 |
| cabinet | 0.8266 | 0.9459 | 0.9916 | 0.8478 |
| shower curtain | 0.8708 | 0.9310 | 0.9954 | 0.9289 |
| door | 0.9073 | 0.9840 | 0.9985 | 0.9755 |

Oracle variants:

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| oracle top-2 | 0.9197 | +0.1418 | 0.8579 | +0.4545 |
| oracle top-5 | 0.9775 | +0.1997 | 0.9427 | +0.5393 |
| oracle graph top-1 | 0.8426 | +0.0647 | 0.9455 | +0.5421 |
| oracle graph top-5 | 0.9829 | +0.2051 | 0.9840 | +0.5806 |

Interpretation:

- This is **Scenario A for oracle headroom**: the candidate set usually contains
  the correct weak class, especially for `picture`.
- Therefore, readout-side improvement is not ruled out by candidate selection.

## Learned / Calibrated Proxy Results

Train-fitted pair probes are strong on the train pair tasks:

| pair | train balanced acc |
| --- | ---: |
| picture_vs_wall | 1.0000 |
| counter_vs_cabinet | 0.9988 |
| desk_vs_table | 0.9967 |
| sink_vs_cabinet | 0.9877 |
| door_vs_wall | 0.9920 |
| shower_curtain_vs_wall | 1.0000 |

But the validation application is bad:

| variant | mIoU | delta mIoU | picture IoU | picture delta |
| --- | ---: | ---: | ---: | ---: |
| pair_probe_top2 | 0.7567 | -0.0212 | 0.1722 | -0.2312 |
| prior_alpha0p25 | 0.7759 | -0.0019 | 0.4003 | -0.0031 |
| bias_unweighted | 0.7673 | -0.0105 | 0.4005 | -0.0029 |
| bias_balanced | 0.7449 | -0.0329 | 0.3193 | -0.0842 |

Interpretation:

- The failure of the previous Top-K reranker is not because the correct class is
  absent from the logits.
- The current learned readout proxies overfit or miscalibrate; they do not yet
  convert the large oracle headroom into a usable method.
- Simple class-prior or bias calibration is not the answer.

## Confusion Structure

Base `picture` prediction distribution:

| target | prediction | fraction |
| --- | --- | ---: |
| picture | picture | 0.5366 |
| picture | wall | 0.4382 |
| picture | sofa | 0.0073 |
| picture | otherfurniture | 0.0043 |
| picture | bookshelf | 0.0037 |

Other weak classes are more distributed but still structured:

- `counter`: mainly `cabinet`, `wall`, `sink`
- `desk`: mainly `table`, `wall`, `bookshelf`, `cabinet`
- `sink`: mainly `cabinet`, `counter`, `wall`
- `door`: mainly `wall`, `window`

## Feature Geometry

Nearest centroid relations among decoder features:

| class | nearest class | centroid cosine |
| --- | --- | ---: |
| picture | wall | 0.8517 |
| counter | sink | 0.6217 |
| desk | table | 0.7331 |
| sink | counter | 0.6217 |
| cabinet | counter | 0.6092 |
| shower curtain | curtain | 0.4970 |
| door | wall | 0.6663 |

Interpretation:

- The confusion graph is reflected in feature geometry.
- `picture` is especially close to `wall`, which explains why unconstrained
  readout corrections overcorrect easily.

## Decision

Do **not** pivot to shifted evaluation solely because readout headroom is small.
The oracle headroom is actually large.

But also do **not** keep trying unconstrained post-hoc rerankers. The current
learned readout proxies fail despite the headroom.

The next readout-side method should be validation-aware and strongly constrained,
for example:

- fit a held-out-train calibrated pair/graph correction and select lambda on
  held-out train, not val;
- add monotonic or margin-bounded corrections that can only flip when the target
  class is already competitive;
- optimize a weak-class-aware decoder fine-tune rather than post-hoc logit
  surgery;
- or treat this as evidence that a small supervised decoder adaptation is needed,
  not a purely offline calibration.
