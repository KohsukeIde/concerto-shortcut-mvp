# CIDA In-Loop Decoder Adaptation

## Summary

CIDA changes the protocol from cached-feature post-hoc correction to in-loop
decoder adaptation on ScanNet train augmentation.

The implementation trains the origin decoder-probe checkpoint with:

- frozen encoder
- trainable PTv3 decoder and segmentation head
- standard weak-class-weighted 20-way CE
- confusion-pair CE for `picture:wall`, `desk:table`, `sink:cabinet`, and
  `counter:cabinet`
- pointwise KL to the frozen origin decoder checkpoint
- batch prediction-distribution anchoring to the frozen origin decoder

Result: the training path is stable, but the two 1200-step pilot conditions are
no-go on full ScanNet val. Both reduce `picture -> wall` slightly, but they
hurt overall mIoU, weak-class mean IoU, and `picture` IoU.

## Jobs

| purpose | job id | resource | status | notes |
| --- | --- | --- | --- | --- |
| AMP smoke | `133365.qjcm` | `rt_QF=1`, `00:12:00` | failed | spconv backward assertion with custom eval-mode frozen encoder path |
| no-AMP smoke | `133366.qjcm` | `rt_QF=1`, `00:12:00` | failed | same spconv assertion |
| batch-2 smoke | `133367.qjcm` | `rt_QF=1`, `00:12:00` | failed | same spconv assertion |
| train-mode smoke, no base anchor | `133369.qjcm` | `rt_QF=1`, `00:12:00` | pass | official Pointcept-style train mode fixes spconv backward |
| anchor smoke | `133371.qjcm` | `rt_QF=1`, `00:12:00` | pass | KL/distribution anchor path works |
| timing smoke | `133372.qjcm` | `rt_QF=1`, `00:20:00` | pass | 20 steps, batch 8 |
| CIDA-base train | `133373.qjcm` | `rt_QF=1`, `01:15:00` | pass | 1200 train steps |
| CIDA-strong train | `133374.qjcm` | `rt_QF=1`, `01:15:00` | pass | 1200 train steps |
| CIDA-base eval | `133376.qjcm` | `rt_QF=1`, `00:20:00` | pass | batch size 1 full val |
| CIDA-strong eval | `133377.qjcm` | `rt_QF=1`, `00:20:00` | pass | batch size 1 full val |

## Implementation Notes

The first CIDA implementation put the frozen encoder submodules in eval mode.
That diverged from the validated Pointcept decoder-probe training path and
caused spconv backward assertions. The script was changed to keep the whole
model in train mode while relying on `requires_grad=False` for the frozen
encoder, matching Pointcept's standard decoder-probe behavior.

The train runs used batch size 8. A batch-size-8 val path produced invalid
absolute mIoU because Pointcept's `inverse` / `origin_segment` handling is not
safe for this custom evaluator with multi-scene val batches. The final reported
numbers below are from a separate eval-only pass with batch size 1.

## Training Stability

Both 1200-step pilots trained without divergence. The last logged training
rows were:

| tag | step | CE | pair | KL | dist | base acc | train acc | hist L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIDA-base | 1200 | 0.2411 | 0.0730 | 0.1005 | 0.0023 | 0.9229 | 0.9269 | 0.0327 |
| CIDA-strong | 1200 | 0.2438 | 0.0703 | 0.1027 | 0.0024 | 0.9230 | 0.9264 | 0.0339 |

The batch-level prediction histogram drift remained modest in the logged
points (`hist_l1` mostly `0.02` to `0.15`).

## Full ScanNet Val

Baseline is the frozen origin decoder-probe checkpoint:
`data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`.

| tag | mIoU | delta mIoU | weak mIoU | weak delta | picture IoU | picture delta | picture -> wall | base picture -> wall | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CIDA-base, 1200 steps | 0.775233 | -0.003120 | 0.629086 | -0.015803 | 0.385599 | -0.020259 | 0.430385 | 0.439414 | no-go |
| CIDA-strong, 1200 steps | 0.774612 | -0.003740 | 0.627961 | -0.016925 | 0.386264 | -0.019583 | 0.430636 | 0.439429 | no-go |

## Interpretation

CIDA is a stricter protocol than cached-feature CoDA: it trains inside the real
ScanNet augmentation loop and changes the decoder mapping itself. Even so, the
two conservative pilots do not recover the oracle/actionability headroom.

The direction of the change is informative:

- `picture -> wall` falls from about `0.439` to about `0.430`
- `picture` IoU still drops by about `0.020`
- weak-class mean IoU drops by about `0.016`

So the model moves some target `picture` points away from `wall`, but the
movement is not class-safe and hurts the broader weak-class decision geometry.

This makes the current CIDA pilot no-go as a positive method. It also strengthens
the conclusion that simply emphasizing a few weak confusion pairs is not enough;
the correction must preserve the full multiclass decision surface, not just the
target pair.

## Artifacts

Train outputs:

- `data/runs/cida_inloop_decoder_adaptation/cida-base-i1200`
- `data/runs/cida_inloop_decoder_adaptation/cida-strong-i1200`

Valid eval outputs:

- `data/runs/cida_inloop_decoder_adaptation/cida-base-i1200-eval-b1`
- `data/runs/cida_inloop_decoder_adaptation/cida-strong-i1200-eval-b1`

CSV:

- `tools/concerto_projection_shortcut/results_cida_inloop_decoder_adaptation.csv`
