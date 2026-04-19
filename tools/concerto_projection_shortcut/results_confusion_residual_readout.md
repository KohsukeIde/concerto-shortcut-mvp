# Confusion-Graph Residual Readout

## Summary

This is a same-checkpoint offline readout-correction pilot on the
`concerto_base_origin.pth` decoder-probe line.

It tests whether the pairwise separability found in the origin decoder feature
can be converted into a better 20-way ScanNet decision by adding small
antisymmetric residual logit corrections for known confusion pairs.

Result: **no meaningful positive**.

- Multi-pair residual readout:
  - best mIoU delta: `+0.0002`
  - best `picture` IoU: base remains best
- `picture:wall` only, small-lambda sweep:
  - best mIoU delta: `+0.00001`
  - best `picture` IoU delta: `+0.00055`

This confirms that the naive form
`logit_a += lambda * expert_ab(x)`, `logit_b -= lambda * expert_ab(x)` is too
weak / too poorly calibrated to become the needed positive method, despite the
experts being strong on their pairwise training task.

## Setup

Base checkpoint:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`

Config:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/config.py`

Implementation:

- `tools/concerto_projection_shortcut/fit_confusion_residual_readout.py`
- `tools/concerto_projection_shortcut/submit_confusion_residual_readout_abciq_qf.sh`

This is **not** a retrained Concerto model. It is an offline same-checkpoint
readout correction over frozen decoder features and fixed base 20-way logits.

## Jobs

| run | job | resource | walltime used | status |
| --- | --- | --- | ---: | --- |
| multi-pair residual readout | `133346.qjcm` | `rt_QF=1` | `00:01:53` | `Exit_status=0` |
| picture-only refined sweep | `133347.qjcm` | `rt_QF=1` | `00:01:21` | `Exit_status=0` |

Outputs:

- `data/runs/scannet_decoder_probe_origin/confusion_residual_readout/confusion_residual_readout.md`
- `data/runs/scannet_decoder_probe_origin/confusion_residual_readout/confusion_residual_summary.csv`
- `data/runs/scannet_decoder_probe_origin/confusion_residual_readout/confusion_residual_class_metrics.csv`
- `data/runs/scannet_decoder_probe_origin/confusion_residual_readout_picture_only/confusion_residual_readout.md`
- `data/runs/scannet_decoder_probe_origin/confusion_residual_readout_picture_only/confusion_residual_summary.csv`
- `data/runs/scannet_decoder_probe_origin/confusion_residual_readout_picture_only/confusion_residual_class_metrics.csv`

Logs:

- `data/logs/abciq/133346.qjcm.OU`
- `data/logs/abciq/133347.qjcm.OU`

## Multi-Pair Run

Pairs:

- `picture:wall`
- `counter:cabinet`
- `desk:table`
- `sink:cabinet`
- `door:wall`
- `shower curtain:wall`

Expert train diagnostics:

| pair | train balanced acc | train AUC proxy |
| --- | ---: | ---: |
| picture_vs_wall | 0.9918 | 0.9995 |
| counter_vs_cabinet | 0.9916 | 0.9994 |
| desk_vs_table | 0.9850 | 0.9993 |
| sink_vs_cabinet | 0.9713 | 0.9957 |
| door_vs_wall | 0.9920 | 0.9996 |
| shower_curtain_vs_wall | 0.9983 | 1.0000 |

Best variants:

| criterion | variant | mIoU | delta mIoU | picture IoU | delta picture IoU | picture->wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| best mIoU | `residual_lam0p1_top1_pair` | 0.7782 | +0.0002 | 0.4038 | -0.0001 | 0.4279 |
| best picture | `base` | 0.7779 | +0.0000 | 0.4039 | +0.0000 | 0.4404 |

Interpretation:

- The residual experts reduce `picture -> wall` for some variants, but they add
  enough new false positives or other class damage that `picture` IoU and mIoU
  do not meaningfully improve.
- Adding all pair corrections at once is not a viable positive method in this
  naive form.

## Picture-Only Refined Sweep

This run repeats the experiment with only `picture:wall` and smaller lambdas.

Best variant:

| criterion | variant | mIoU | delta mIoU | picture IoU | delta picture IoU | picture->wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| best mIoU / picture | `residual_lam0p02_top2_both` | 0.7780 | +0.0000 | 0.4045 | +0.0006 | 0.4232 |

Interpretation:

- The refined sweep confirms that the effect is real but extremely small.
- A simple binary residual margin can slightly reduce `picture -> wall`, but it
  does not produce a paper-relevant same-checkpoint positive.

## Takeaway

The diagnosis still points toward readout / multiclass decision geometry, but
the first obvious implementation is too naive. A stronger next readout method
would need one of:

- calibrated class-prior correction rather than raw antisymmetric logit shifts;
- explicit optimization of 20-way validation-like loss on a held-out train
  split;
- a constrained confusion-aware head retraining step instead of post-hoc logit
  surgery;
- or a proper decoder/LoRA fine-tune objective that conditions on weak
  confusion classes.
