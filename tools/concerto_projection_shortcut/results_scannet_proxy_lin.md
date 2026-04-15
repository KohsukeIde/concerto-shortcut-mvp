# ScanNet Linear Proxy Results

Updated: 2026-04-16 04:35 JST

## Runs

| experiment | status | final/last eval mIoU | final/last eval mAcc | final/last eval allAcc | train best mIoU | eval count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| scannet-proxy-concerto-continue-lin | finished | 0.4794 | 0.6114 | 0.7718 | 0.4552 | 11 |
| scannet-proxy-coord-mlp-continue-lin | finished | 0.4064 | 0.5503 | 0.7422 | 0.3829 | 11 |
| scannet-proxy-no-enc2d-continue-lin | finished | 0.4010 | 0.5440 | 0.7391 | 0.3765 | 11 |
| scannet-proxy-no-enc2d-renorm-continue-lin | finished validation / full test aborted (disk) | 0.3794 | 0.5510 | 0.7282 | 0.3802 | 10 |
| scannet-proxy-projres-v1a-alpha005-h10032-qf32-lin | finished no-go | 0.3627 | 0.5083 | 0.7247 | 0.3627 | 10 |
| scannet-proxy-projres-v1b-combo-b075-a001-h10016x4-qf16-lin | finished no strong-go | 0.4220 | 0.5719 | 0.7469 | 0.4220 | 10 |
| scannet-proxy-projres-v1b-resonly-b075-a000-h10016x4-qf16-lin | finished no strong-go | 0.4176 | 0.5625 | 0.7444 | 0.4176 | 10 |
| scannet-proxy-projres-v1b-combo-b050-a002-h10016x4-qf16-lin | finished no strong-go | 0.4129 | 0.5551 | 0.7441 | 0.4129 | 10 |
| scannet-proxy-projres-v1b-penalty-b000-a002-h10016x4-qf16-lin | finished no strong-go | 0.3887 | 0.5308 | 0.7324 | 0.3907 | 10 |

## Readout

- `coord_mlp` is below original continuation by 0.0730 final/last-eval mIoU.
- `coord_mlp` retains 84.8% of the original continuation final/last-eval mIoU.
- `no-enc2d` linear is now finished.
- On final/last eval mIoU, `coord_mlp` is +0.0054 above `no-enc2d` and -0.0730 below original.
- On training-side best mIoU, `coord_mlp` is +0.0064 above `no-enc2d` and -0.0723 below original.
- This means `coord_mlp` recovers 6.9% of the `no-enc2d` -> original gap by final/last eval mIoU and 8.1% by training-side best mIoU.
- `no-enc2d-renorm` finished linear training and validation, but its post-train full test aborted during logging with `No space left on device`.
- Relative to `no-enc2d-renorm`, `coord_mlp` is +0.0270 final/last-eval mIoU and +0.0027 training-side best mIoU.
- Relative to `no-enc2d-renorm`, `coord_mlp` recovers 27.0% of the gap to original by final/last eval mIoU, but only 3.6% by training-side best mIoU.
- Current interpretation: downstream shortcut relevance remains measurable, but the renormalized control further weakens the stronger "mostly coordinate shortcut" claim in this continuation proxy. The objective-level shortcut claim is unchanged.
- ProjRes v1a with `alpha=0.05` did not clear the replacement gate:
  - final/last eval mIoU is 0.1167 below original continuation.
  - training-side best mIoU is 0.0925 below original continuation.
  - it is also 0.0167 / 0.0175 below `no-enc2d-renorm` on last / best mIoU.
- ProjRes v1b factorized the fix into partial residualization (`beta`) and
  alignment penalty (`alpha`). The best tested arm is `combo-b075-a001`
  (`beta=0.75`, `alpha=0.01`):
  - final/last and best mIoU are both 0.4220.
  - this is +0.0593 last / +0.0593 best over `projres_v1a`.
  - this is +0.0426 last / +0.0418 best over `no-enc2d-renorm`.
  - it is still -0.0574 last / -0.0332 best below original continuation, so
    the replacement gate remains no strong-go.
- Among the v1b continuations, target residualization mattered more than the
  alignment penalty alone. `penalty-b000-a002` reached only 0.3887 / 0.3907
  mIoU, whereas the `beta=0.75` residualized arms reached 0.4176 to 0.4220.
