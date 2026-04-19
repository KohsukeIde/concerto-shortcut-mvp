# CoDA Transfer-Failure Analysis

## Summary

This is the minimal follow-up analysis for the CoDA no-go result. It explains
why the adapter looked good on the heldout selection cache but did not transfer
to ScanNet val.

Result: **the failure is a transfer / calibration failure, not simply a lack of
adapter capacity**.

Two facts matter most:

1. `picture` has a much larger train/val feature shift than stable classes such
   as `wall`.
2. The aggressive heldout-selected adapter does improve target-`picture`
   ordering on val, but it also overpredicts `picture` and harms other classes,
   so `picture` IoU and mIoU drop.

This means future methods should not use the current class-balanced cached
heldout sweep as the main selection mechanism. Any next readout-side method
needs representative validation, in-loop training with real augmentation, or a
stricter calibration constraint.

## Setup

Base checkpoint:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`

CoDA adapter:

- `data/runs/scannet_decoder_probe_origin/coda_decoder_adapter_fullscan/coda_adapter.pt`

Implementation:

- `tools/concerto_projection_shortcut/eval_coda_transfer_failure_analysis.py`
- `tools/concerto_projection_shortcut/submit_coda_transfer_failure_analysis_abciq_qf.sh`

Job:

- `133363.qjcm`
- `rt_QF=1`
- walltime used: `00:02:53`
- `Exit_status=0`

Run outputs:

- `data/runs/scannet_decoder_probe_origin/coda_transfer_failure_analysis/coda_transfer_failure_analysis.md`
- `data/runs/scannet_decoder_probe_origin/coda_transfer_failure_analysis/coda_transfer_class_iou.csv`
- `data/runs/scannet_decoder_probe_origin/coda_transfer_failure_analysis/coda_transfer_pair_confusion.csv`
- `data/runs/scannet_decoder_probe_origin/coda_transfer_failure_analysis/coda_transfer_feature_shift.csv`
- `data/runs/scannet_decoder_probe_origin/coda_transfer_failure_analysis/coda_transfer_pair_centroids.csv`
- `data/runs/scannet_decoder_probe_origin/coda_transfer_failure_analysis/coda_transfer_picture_rank_drift.csv`
- `data/logs/abciq/133363.qjcm.OU`

## Feature Shift

`picture` is the outlier.

| class | cos train-heldout | cos train-val | cos heldout-val | var train | var heldout | var val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| picture | 0.9960 | 0.9615 | 0.9371 | 60.36 | 50.49 | 95.33 |
| wall | 0.9998 | 0.9998 | 0.9994 | 63.31 | 62.76 | 69.25 |
| counter | 0.9988 | 0.9968 | 0.9984 | 66.39 | 72.45 | 76.58 |
| desk | 0.9996 | 0.9984 | 0.9980 | 67.18 | 68.14 | 79.88 |
| sink | 0.9995 | 0.9994 | 0.9987 | 77.98 | 78.11 | 88.10 |
| cabinet | 0.9988 | 0.9980 | 0.9992 | 68.08 | 68.40 | 90.41 |

Interpretation:

- Heldout `picture` is very close to train `picture`.
- Val `picture` is much farther from both train and heldout, and its variance
  trace is much larger.
- `wall` remains nearly identical across train, heldout, and val.

This makes the heldout split a weak proxy for target `picture` behavior on val.

## Candidate Ordering Drift

The base decoder already treats heldout `picture` as easy, but val `picture` is
hard and wall-dominated.

| domain | variant | GT top1 | GT top2 | GT top5 | wall top1 | mean picture-wall margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| heldout | base | 0.9259 | 0.9937 | 0.9990 | 0.0727 | +4.0820 |
| val | base | 0.5357 | 0.8888 | 0.9576 | 0.4388 | -0.3420 |
| val | heldout-selected `lam1_tau1` | 0.6212 | 0.9031 | 0.9598 | 0.3486 | +1.3363 |
| val | best-safe `lam0p2_tau1` | 0.5550 | 0.8947 | 0.9587 | 0.4187 | -0.0063 |

Interpretation:

- The adapter does push val `picture` points in the intended direction.
- The aggressive variant reduces target `picture -> wall` from `0.4388` to
  `0.3486`.
- But this is not enough for IoU because the same correction changes the
  predicted class distribution elsewhere.

## Overcorrection

The heldout-selected aggressive adapter increases target `picture` accuracy on
val, but `picture` IoU drops because false positives and other class damage grow.

| metric | base | heldout-selected `lam1_tau1` |
| --- | ---: | ---: |
| val `picture` IoU | 0.4022 | 0.3776 |
| val `picture` target accuracy | 0.5357 | 0.6212 |
| val `picture` prediction count | 179,966 | 262,691 |
| val `picture -> wall` | 0.4388 | 0.3486 |
| val `wall` IoU delta | 0.0000 | -0.0162 |
| val `desk` IoU delta | 0.0000 | -0.0254 |
| val `sink` IoU delta | 0.0000 | -0.0171 |

So the adapter improves the binary-looking target-picture ordering while
damaging the multiclass calibration.

## Selection Caveat

The CoDA heldout selection cache was class-balanced / capped. That is useful for
training signal, but it is not a representative selection objective. In this
full-distribution analysis, the same aggressive correction already shows
multiclass side effects on the heldout split, and the side effects become worse
on val.

This explains why the heldout-selected `lam1_tau1` looked promising in
`results_coda_decoder_adapter.md` but failed on ScanNet val.

## Decision

This analysis is sufficient. Do not expand into a broader diagnosis sweep now.

Actionable takeaways:

- Do not select readout corrections on the current class-balanced cached
  heldout objective alone.
- Cached-feature post-hoc adapters are still failing to recover oracle headroom.
- The next readout-side attempt must materially change the protocol:
  representative full-distribution validation, in-loop decoder adaptation with
  real augmentation, or a strictly constrained rare-class calibration objective.
