# ScanNet Linear Proxy Results

Updated: 2026-04-13 15:35 JST

## Runs

| experiment | status | final/last eval mIoU | final/last eval mAcc | final/last eval allAcc | train best mIoU | eval count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| scannet-proxy-concerto-continue-lin | finished | 0.4794 | 0.6114 | 0.7718 | 0.4552 | 11 |
| scannet-proxy-coord-mlp-continue-lin | finished | 0.4064 | 0.5503 | 0.7422 | 0.3829 | 11 |
| scannet-proxy-no-enc2d-continue-lin | running / interim | 0.3762 | 0.5368 | 0.7236 | 0.3765 | 10 |

## Readout

- `coord_mlp` is below original continuation by 0.0730 final-eval mIoU.
- `coord_mlp` retains 84.8% of the original continuation final-eval mIoU.
- `no-enc2d` linear is still running; the row above is an interim last validation, not a final eval.
- On training-side best mIoU so far, `coord_mlp` is +0.0064 above `no-enc2d` and -0.0723 below original.
- This means `coord_mlp` currently recovers 8.1% of the `no-enc2d` -> original gap by training-side best mIoU.
- Current interpretation: downstream shortcut relevance is measurable but weak in this continuation proxy; this is not enough for the stronger "mostly coordinate shortcut" claim unless the final test/eval shifts substantially.
