# Constrained Top-K Set Decoder

## Summary

This is the validation-aware structured readout pilot motivated by the oracle
actionability analysis. The base `concerto_base_origin.pth` decoder-probe
checkpoint is fixed. A small candidate-set decoder is trained on a ScanNet train
scene split, selected on a held-out train split, and then applied once to
ScanNet val.

Result: **weak positive, below gate**.

The full-scan valid run selects `k=2, lambda=0.2, tau=0.5, trust_gap=999`. On
ScanNet val it gives:

- mIoU: `0.77865983 -> 0.77878256`, delta `+0.00012274`
- `picture` IoU: `0.40257231 -> 0.40386984`, delta `+0.00129753`
- `picture -> wall`: `0.43864309 -> 0.42774366`

This is a real but very small movement. It does not meet the planned weak gate
(`picture +0.005` without mIoU loss).

## Setup

Base checkpoint:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`

Config:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/config.py`

Implementation:

- `tools/concerto_projection_shortcut/fit_constrained_topk_set_decoder.py`
- `tools/concerto_projection_shortcut/submit_constrained_topk_set_decoder_abciq_qf.sh`

Run outputs:

- `data/runs/scannet_decoder_probe_origin/constrained_topk_set_decoder/constrained_topk_set_decoder.md`
- `data/runs/scannet_decoder_probe_origin/constrained_topk_set_decoder_fullscan/constrained_topk_set_decoder.md`
- `data/runs/scannet_decoder_probe_origin/constrained_topk_set_decoder_fullscan/constrained_topk_heldout_sweep.csv`
- `data/runs/scannet_decoder_probe_origin/constrained_topk_set_decoder_fullscan/constrained_topk_val_selected.csv`
- `data/runs/scannet_decoder_probe_origin/constrained_topk_set_decoder_fullscan/constrained_topk_set_decoder.pt`
- `data/logs/abciq/133354.qjcm.OU`
- `data/logs/abciq/133355.qjcm.OU`

## Jobs

| run | job | resource | walltime used | status | note |
| --- | --- | --- | ---: | --- | --- |
| initial capped run | `133354.qjcm` | `rt_QF=1` | `00:01:41` | `Exit_status=0` | invalid heldout: `picture=0`, `counter=0` |
| full-scan valid run | `133355.qjcm` | `rt_QF=1` | `00:03:34` | `Exit_status=0` | valid heldout: `picture=60000`, `counter=60000` |

The first run is recorded for reproducibility but should not be used for
method selection because the heldout split hit point caps after only 26 scenes
and missed rare weak classes.

## Full-Scan Training / Selection

Full-scan collection:

- reranker train points: `1200000`
- heldout train points: `1168197`
- train scenes scanned: `1201`
- heldout `picture`: `60000`
- heldout `counter`: `60000`

Training trace:

| step | loss | candidate base acc | candidate train acc | delta RMS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1850 | 0.9503 | 0.9506 | 0.0832 |
| 250 | 0.0894 | 0.9482 | 0.9724 | 0.8309 |
| 500 | 0.0761 | 0.9487 | 0.9777 | 0.8505 |
| 750 | 0.0660 | 0.9498 | 0.9796 | 0.8660 |
| 1000 | 0.0581 | 0.9529 | 0.9830 | 0.8804 |
| 1250 | 0.0557 | 0.9500 | 0.9851 | 0.9031 |
| 1500 | 0.0550 | 0.9460 | 0.9839 | 0.9208 |
| 1750 | 0.0565 | 0.9460 | 0.9817 | 0.9069 |
| 2000 | 0.0482 | 0.9524 | 0.9871 | 0.9025 |
| 2250 | 0.0502 | 0.9482 | 0.9860 | 0.9265 |
| 2500 | 0.0427 | 0.9539 | 0.9885 | 0.8972 |

Heldout-selected setting:

- `k2_lam0p2_tau0p5_gap999`
- heldout mIoU: `0.91233062`, delta `+0.00267545`
- heldout `picture` IoU: `0.92185277`, delta `+0.00320848`
- heldout `picture -> wall`: `0.06996667`

## ScanNet Val

| variant | mIoU | delta | picture IoU | picture delta | picture->wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.7787 | +0.0000 | 0.4026 | +0.0000 | 0.4386 |
| selected `k2_lam0p2_tau0p5_gap999` | 0.7788 | +0.0001 | 0.4039 | +0.0013 | 0.4277 |

## Interpretation

- The structured decoder improves the candidate train objective and also
  improves heldout train.
- It transfers only weakly to ScanNet val.
- The direction is consistent with oracle/actionability: `picture -> wall`
  decreases and `picture` IoU rises slightly.
- The magnitude is far below what is needed for a paper-relevant positive.

## Takeaway

The constrained top-K method is better disciplined than the free reranker, but
it still does not recover meaningful oracle headroom. This suggests that the
next positive method should move beyond offline post-hoc logit correction:

- lightweight decoder fine-tuning with weak-class-aware validation;
- explicit train/heldout calibration over the original decoder objective;
- or encoder/decoder adaptation rather than a fixed-logit reranker.
